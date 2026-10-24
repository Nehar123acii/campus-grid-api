import pandas as pd, numpy as np, joblib

pack = joblib.load("forecast_models.joblib")
MODELS, QHAT, FEATS = pack["models"], pack["qhat"], pack["feats"]

h = pd.read_csv("clean_hourly.csv", parse_dates=["timestamp"])
b = pd.read_excel("data/buildings.xlsx")
BTYPE = dict(zip(b.building_id, b.building_type))
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

HELD_OUT = pd.Timestamp("2026-10-07")

# Which building types each event affects (my reading of the event text; edit if wrong)
KEYS = {
    "hackathon": {"labs", "food"}, "placement": {"labs", "academic", "admin"},
    "alumni": {"food"}, "feast": {"hostel", "food"}, "check-in": {"hostel", "food"},
    "allotment": {"hostel", "admin"}, "semester begins": {"hostel", "academic", "library", "food"},
    "sports": {"hostel", "food"}, "fest": {"hostel", "food"}, "expo": {"labs", "academic"},
}
def related(event, btype):
    if not isinstance(event, str):
        return False
    e = event.lower()
    return any(k in e and btype in types for k, types in KEYS.items())

events = []
for m in ["electricity_kwh", "water_litres"]:
    d = build(m, 6)
    d["lo"], d["mid"], d["hi"] = band(m, 6, d)
    d["flag"] = d.y > d.hi
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
                "building_id": bid, "btype": BTYPE[bid], "metric": m,
                "start": r.target_time.iloc[0], "end": r.target_time.iloc[-1],
                "hours": len(r),
                "excess_total": round((r.y - r.mid).sum(), 0),
                "excess_pct": round(100 * (r.y.sum() / r.mid.sum() - 1), 0),
                "night_share": round((r.target_time.dt.hour < 6).mean(), 2),
                "day_type": info["day_type"] if info is not None else "",
                "calendar_event": info["event"] if info is not None else "",
                "held_out": r.target_time.iloc[0] >= HELD_OUT,
            })

ev = pd.DataFrame(events)
ev["date"] = ev.start.dt.normalize()

def days_nearby(r):
    s = ev[(ev.building_id == r["building_id"]) & (ev.metric == r["metric"]) &
           (ev.date >= r["date"] - pd.Timedelta(days=3)) & (ev.date <= r["date"] + pd.Timedelta(days=3))]
    return s.date.nunique()
ev["days_nearby"] = ev.apply(days_nearby, axis=1)

cnt = ev.groupby(["metric", "date"]).building_id.nunique()
ev["n_bldgs"] = [cnt[(m, d)] for m, d in zip(ev.metric, ev.date)]

def cause(r):
    if r["metric"] == "electricity_kwh" and r["excess_pct"] >= 300:
        return "Extreme spike, possible sensor or meter fault"
    if related(r["calendar_event"], r["btype"]):
        return "Event-driven surge: " + r["calendar_event"]
    if r["metric"] == "electricity_kwh" and r["day_type"] in ("holiday", "weekend", "semester_break"):
        return "Excess on non-working day, equipment likely left on"
    if r["day_type"] == "exam":
        return "Exam-period demand surge"
    if r["n_bldgs"] >= 3:
        return "Campus-wide shift (several buildings at once), not a single-building fault"
    if r["days_nearby"] >= 3:
        return ("Persistent multi-day excess, probable leak" if r["metric"] == "water_litres"
                else "Persistent multi-day excess, check meter or equipment")
    if r["night_share"] >= 0.25:
        return "Overnight excess, check for equipment left running"
    return "Unexplained daytime excess, manual check"

ev["cause"] = ev.apply(cause, axis=1)
ev["cause_type"] = ev.cause.str.split(":").str[0]
ev["score"] = ev.excess_pct * ev.hours

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
cols = ["building_id", "start", "hours", "excess_pct", "night_share", "cause", "held_out"]
for m in ["water_litres", "electricity_kwh"]:
    print(f"\n=== TOP 10: {m} ===")
    print(ev[ev.metric == m].sort_values("score", ascending=False).head(10)[cols].to_string(index=False))

print("\n=== KNOWN INCIDENT CHECK ===")
KNOWN = [
    ("B06", "water_litres", "2026-08-04", "2026-08-10", "Water leak (notes Aug 5-10)"),
    ("B03", "electricity_kwh", "2026-10-02", "2026-10-02", "AC and lights left on, holiday"),
    ("B02", "electricity_kwh", "2026-09-19", "2026-09-21", "Electricity reading higher than usual"),
    ("B04", "electricity_kwh", "2026-08-27", "2026-08-27", "Power analyzer calibration"),
]
for bid, m, s, e, name in KNOWN:
    s, e = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
    hit = ev[(ev.building_id == bid) & (ev.metric == m) & (ev.start < e) & (ev.end >= s)]
    print(f"{name} [{bid}]: {len(hit)} event(s) flagged", hit.cause.unique().tolist())

print("\n=== CAUSE COUNTS (all events) ===")
print(ev.cause_type.value_counts().to_string())
print("\n=== CAUSE COUNTS (held-out, Oct 7 onward) ===")
print(ev[ev.held_out].cause_type.value_counts().to_string())
print("\nTotal events:", len(ev), "| held-out:", int(ev.held_out.sum()))
ev.to_csv("anomalies.csv", index=False)