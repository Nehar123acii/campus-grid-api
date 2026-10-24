import pandas as pd, numpy as np, os, shutil

WATER_RS_PER_L = 0.03   # ASSUMED tariff, edit to your campus rate
os.makedirs("export_for_B", exist_ok=True)

b = pd.read_excel("data/buildings.xlsx")
NAME = dict(zip(b.building_id, b.building_name))

# ---------- 1. GreenScore: actual vs forecast, per building per day ----------
f = pd.read_csv("forecasts_test.csv", parse_dates=["origin_time", "target_time"])
f = f[f.horizon_h == 6].copy()
f["date"] = f.target_time.dt.normalize()

g = (f.groupby(["building_id", "metric", "date"])
       .agg(actual=("actual", "sum"), forecast=("p50", "sum"), hours=("actual", "size"))
       .reset_index())
g = g[g.hours >= 20]                       # only near-complete days count
g["saving"] = 1 - g.actual / g.forecast    # positive = used less than forecast

print("Bias check (mean saving across all buildings; near 0 means the forecast is unbiased):")
print((g.groupby("metric").saving.mean() * 100).round(1).to_string(), "\n")

d = g.pivot_table(index=["building_id", "date"], columns="metric", values="saving").reset_index()
d = d.rename(columns={"electricity_kwh": "elec_saving", "water_litres": "water_saving"})
d["saving"] = d[["elec_saving", "water_saving"]].mean(axis=1)
d = d.dropna(subset=["saving"])
d["status"] = np.where(d.saving > 0.30, "held for review (too good to be true)", "scored")
d["points"] = (50 + 250 * d.saving.clip(-0.5, 0.5)).clip(0, 100).round(1)
d.loc[d.status != "scored", "points"] = np.nan
d["building_name"] = d.building_id.map(NAME)
d.to_csv("leaderboard_daily.csv", index=False)

rows = []
for bid, x in d.sort_values("date").groupby("building_id"):
    ok = x[x.status == "scored"]
    streak = 0
    for v, st in zip(x.saving.values[::-1], x.status.values[::-1]):
        if v > 0 and st == "scored":
            streak += 1
        else:
            break
    rows.append({"building_id": bid, "building_name": NAME[bid],
                 "green_score": round(ok.points.mean(), 1) if len(ok) else np.nan,
                 "avg_saving_pct": round(100 * x.saving.mean(), 1),
                 "streak_days": streak, "days_scored": len(ok),
                 "days_held": int((x.status != "scored").sum())})
lb = pd.DataFrame(rows).sort_values("green_score", ascending=False)
lb.insert(0, "rank", range(1, len(lb) + 1))
lb.to_csv("leaderboard_summary.csv", index=False)
print("=== LEADERBOARD (held-out period, Oct 7 onward) ===")
print(lb.to_string(index=False))

# ---------- 2. Leak repair: estimated saving ----------
h = pd.read_csv("clean_hourly.csv", parse_dates=["timestamp"])
w = h[h.building_id == "B06"].sort_values("timestamp").set_index("timestamp").water_litres
before, during, after = (w["2026-07-29":"2026-08-03"].mean(),
                         w["2026-08-05":"2026-08-09"].mean(),
                         w["2026-08-11":"2026-08-17"].mean())
lit_day = (during - after) * 24
leak = pd.DataFrame([{
    "building_id": "B06", "before_L_per_h": round(before), "leak_L_per_h": round(during),
    "after_fix_L_per_h": round(after), "excess_L_per_day": round(lit_day),
    "est_rs_per_day": round(lit_day * WATER_RS_PER_L),
    "note": "ESTIMATE: compares leak days with post-fix days; weekday mix differs; tariff assumed"}])
leak.to_csv("leak_savings.csv", index=False)
print("\n=== B06 LEAK REPAIR (estimate) ===")
print(leak.drop(columns="note").to_string(index=False))

# ---------- 3. Export package for Person B ----------
for fn in ["clean_hourly.csv", "forecasts_test.csv", "forecast_metrics.csv", "anomalies.csv",
           "recommendations.csv", "clean_manual_logs.csv", "leaderboard_daily.csv",
           "leaderboard_summary.csv", "leak_savings.csv"]:
    if os.path.exists(fn):
        shutil.copy(fn, os.path.join("export_for_B", fn))
    else:
        print("MISSING:", fn)
b.to_csv("export_for_B/buildings.csv", index=False)
pd.read_excel("data/calendar.xlsx").to_csv("export_for_B/calendar.csv", index=False)
if os.path.exists("waste_events_test_predicted.xlsx"):
    pd.read_excel("waste_events_test_predicted.xlsx").to_csv(
        "export_for_B/waste_events_test_predicted.csv", index=False)
else:
    print("NOTE: waste predictions not included yet. Run waste_model.py, then rerun this script.")

lines = ["REPLAY WINDOW (held-out, never trained on): 2026-10-07 to 2026-10-17", ""]
for fn in sorted(os.listdir("export_for_B")):
    if fn.endswith(".csv"):
        t = pd.read_csv(os.path.join("export_for_B", fn))
        lines.append(f"{fn} ({len(t)} rows): " + ", ".join(t.columns))
open("export_for_B/data_dictionary.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\nDone. Send the whole export_for_B folder to Person B.")