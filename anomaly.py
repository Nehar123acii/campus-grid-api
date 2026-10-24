import pandas as pd, numpy as np, joblib

pack = joblib.load("forecast_models.joblib")
MODELS, QHAT, FEATS = pack["models"], pack["qhat"], pack["feats"]

h = pd.read_csv("clean_hourly.csv", parse_dates=["timestamp"])
b = pd.read_excel("data/buildings.xlsx")
cal = pd.read_excel("data/calendar.xlsx")
cal["date"] = pd.to_datetime(cal["date"])
cal["has_event"] = cal["event"].notna().astype(int)
DT = sorted(cal.day_type.unique())
cal["day_type_code"] = cal.day_type.map({k: i for i, k in enumerate(DT)})
CALINFO = cal.set_index("date")[["day_type", "event"]]
calf = cal[["date", "day_type_code", "has_event", "avg_temp_c", "rainfall_mm", "humidity_pct"]]

b["btype_code"] = b.building_type.astype("category").cat.codes
b["bcode"] = b.building_id.str[1:].astype(int)
h = h.merge(b[["building_id", "bcode", "btype_code", "area_sqm", "has_solar"]], on="building_id")
h = h.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

for m in ["electricity_kwh", "water_litres"]:
    for k in [1, 2, 3, 6, 12, 24, 48, 168]:
        h[f"{m}_lag{k}"] = h.groupby("building_id")[m].shift(k)
    for win in [6, 24]:
        h[f"{m}_roll{win}"] = h.groupby("building_id")[m].transform(
            lambda s: s.shift(1).rolling(win, min_periods=3).mean())
h["footfall_lag1"] = h.groupby("building_id")["footfall"].shift(1)

def build(m, hz):
    d = h.copy()
    d["y"] = d.groupby("building_id")[m].shift(-hz)
    d["target_time"] = d.timestamp + pd.Timedelta(hours=hz)
    d["tt_hour"] = d.target_time.dt.hour
    d["tt_dow"] = d.target_time.dt.dayofweek
    d["date"] = d.target_time.dt.normalize()
    d = d.merge(calf, on="date", how="left")
    return d.dropna(subset=["y"]).reset_index(drop=True)

def band(m, hz, d):
    P = np.sort(np.column_stack([MODELS[(m, hz)][q].predict(d[FEATS]) for q in (0.1, 0.5, 0.9)]), axis=1)
    q = QHAT[(m, hz)]
    return np.clip(P[:, 0] - q, 0, None), P[:, 1], P[:, 2] + q

HELD_OUT = pd.Timestamp("2026-10-04")
events = []
for m in ["electricity_kwh", "water_litres"]:
    d = build(m, 6)
    d["lo"], d["mid"], d["hi"] = band(m, 6, d)
    d["flag"] = d.y > d.hi

    if m == "water_litres":
        print("=== LEAK TEST: B06 water, flagged hours per day ===")
        t = d[(d.building_id == "B06") & (d.target_time >= "2026-07-30") & (d.target_time < "2026-08-16")]
        print(t.groupby(t.target_time.dt.date).flag.sum().to_string(), "\n")

    for bid, g in d.groupby("building_id"):
        f = g[g.flag].sort_values("target_time")
        if f.empty:
            continue
        rid = (f.target_time.diff() > pd.Timedelta(hours=2)).cumsum()
        for _, r in f.groupby(rid):
            if len(r) < 3:
                continue
            day = r.target_time.iloc[0].normalize()
            info = CALINFO.loc[day] if day in CALINFO.index else None
            events.append({
                "building_id": bid, "metric": m,
                "start": r.target_time.iloc[0], "end": r.target_time.iloc[-1],
                "hours": len(r),
                "excess_pct": round(100 * (r.y.sum() / r.mid.sum() - 1), 0),
                "excess_total": round((r.y - r.mid).sum(), 0),
                "night_share": round((r.target_time.dt.hour < 6).mean(), 2),
                "day_type": info["day_type"] if info is not None else "",
                "calendar_event": info["event"] if info is not None else "",
                "held_out": r.target_time.iloc[0] >= HELD_OUT,
            })

ev = pd.DataFrame(events).sort_values("excess_total", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45)
print("=== TOP 20 ANOMALY EVENTS ===")
print(ev.head(20).to_string(index=False))
print("\nTotal events:", len(ev), "| in held-out period (Oct 4+):", int(ev.held_out.sum()))
ev.to_csv("anomalies.csv", index=False)
print("Saved anomalies.csv (for Person B)")