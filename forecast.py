import pandas as pd, numpy as np, lightgbm as lgb, joblib, json, time, os

h = pd.read_csv("clean_hourly.csv", parse_dates=["timestamp"])
b = pd.read_excel("data/buildings.xlsx")
cal = pd.read_excel("data/calendar.xlsx")
cal["date"] = pd.to_datetime(cal["date"])
cal["has_event"] = cal["event"].notna().astype(int)
DT = sorted(cal.day_type.unique())
cal["day_type_code"] = cal.day_type.map({k: i for i, k in enumerate(DT)})
cal = cal[["date", "day_type_code", "has_event", "avg_temp_c", "rainfall_mm", "humidity_pct"]]

b["btype_code"] = b.building_type.astype("category").cat.codes
b["bcode"] = b.building_id.str[1:].astype(int)
h = h.merge(b[["building_id", "bcode", "btype_code", "area_sqm", "has_solar"]], on="building_id")
h = h.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

# ---- sanity check: does the known B06 leak (Aug 5-10) show in the data? ----
w = h[(h.building_id == "B06") & (h.timestamp >= "2026-07-30") & (h.timestamp < "2026-08-16")]
print("B06 daily mean water (litres/hour):")
print(w.groupby(w.timestamp.dt.date).water_litres.mean().round(0).to_string(), "\n")

# ---- features: only past information ----
METRICS = ["electricity_kwh", "water_litres"]
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
lagcols = []
for m in METRICS:
    for k in LAGS:
        c = f"{m}_lag{k}"
        h[c] = h.groupby("building_id")[m].shift(k)
        lagcols.append(c)
    for win in [6, 24]:
        c = f"{m}_roll{win}"
        h[c] = h.groupby("building_id")[m].transform(
            lambda s: s.shift(1).rolling(win, min_periods=3).mean())
        lagcols.append(c)
h["footfall_lag1"] = h.groupby("building_id")["footfall"].shift(1)
lagcols.append("footfall_lag1")

FEATS = lagcols + ["tt_hour", "tt_dow", "day_type_code", "has_event", "avg_temp_c",
                   "rainfall_mm", "humidity_pct", "bcode", "btype_code", "area_sqm", "has_solar"]

def build(m, hz):
    d = h.copy()
    d["y"] = d.groupby("building_id")[m].shift(-hz)
    d["base"] = d.groupby("building_id")[m].shift(24 - hz)   # same hour yesterday
    d["target_time"] = d.timestamp + pd.Timedelta(hours=hz)
    d["tt_hour"] = d.target_time.dt.hour
    d["tt_dow"] = d.target_time.dt.dayofweek
    d["date"] = d.target_time.dt.normalize()
    d = d.merge(cal, on="date", how="left")
    return d.dropna(subset=["y"])

T1, T2 = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-07")
QS = (0.1, 0.5, 0.9)

def predict(models, X):
    P = np.column_stack([models[q].predict(X) for q in QS])
    return np.sort(P, axis=1)

rows, all_models, qhats, outputs = [], {}, {}, []
for m in METRICS:
    for hz in [6, 12]:
        d = build(m, hz)
        tr = d[d.target_time < T1]
        ca = d[(d.target_time >= T1) & (d.target_time < T2)]
        te = d[d.target_time >= T2]
        if m == "water_litres":   # keep the known B06 leak out of training
            leak = (tr.building_id == "B06") & tr.target_time.between("2026-08-05", "2026-08-11")
            tr = tr[~leak]

        mods = {}
        for q in QS:
            mods[q] = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=300, learning_rate=0.05,
                num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                verbose=-1).fit(tr[FEATS], tr.y)

        # conformal calibration (CQR): widen the band so ~80% of actuals fall inside
        Pc = predict(mods, ca[FEATS])
        score = np.maximum(Pc[:, 0] - ca.y.values, ca.y.values - Pc[:, 2])
        n = len(score)
        qhat = np.quantile(score, min(1.0, np.ceil((n + 1) * 0.8) / n), method="higher")

        te = te.dropna(subset=["base"]).copy()
        P = predict(mods, te[FEATS])
        lo, mid, hi = np.clip(P[:, 0] - qhat, 0, None), P[:, 1], P[:, 2] + qhat
        y = te.y.values

        mae_b = np.mean(np.abs(te.base.values - y))
        mae_m = np.mean(np.abs(mid - y))
        rows.append({
            "metric": m, "horizon_h": hz, "test_rows": len(te),
            "MAE_baseline": round(mae_b, 2), "MAE_lightgbm": round(mae_m, 2),
            "improvement_%": round(100 * (1 - mae_m / mae_b), 1),
            "WAPE_%": round(100 * np.abs(mid - y).sum() / np.abs(y).sum(), 1),
            "coverage_raw_%": round(100 * np.mean((y >= P[:, 0]) & (y <= P[:, 2])), 1),
            "coverage_conformal_%": round(100 * np.mean((y >= lo) & (y <= hi)), 1),
        })
        all_models[(m, hz)] = mods
        qhats[(m, hz)] = float(qhat)
        o = te[["building_id", "timestamp", "target_time", "y", "base"]].copy()
        o["metric"], o["horizon_h"] = m, hz
        o["p10"], o["p50"], o["p90"] = lo, mid, hi
        outputs.append(o.rename(columns={"timestamp": "origin_time", "y": "actual", "base": "baseline"}))

res = pd.DataFrame(rows)
print(res.to_string(index=False))

# edge cost: size and time for one forecast (3 quantile models)
x1 = te[FEATS].iloc[[0]]
t = time.perf_counter()
for _ in range(100):
    predict(mods, x1)
ms = (time.perf_counter() - t) / 100 * 1000
joblib.dump({"models": all_models, "qhat": qhats, "feats": FEATS}, "forecast_models.joblib")
kb = os.path.getsize("forecast_models.joblib") / 1024
print(f"\nOne forecast (3 models): {ms:.1f} ms | all 12 models on disk: {kb:.0f} KB")

pd.concat(outputs).to_csv("forecasts_test.csv", index=False)
res.to_csv("forecast_metrics.csv", index=False)
print("Saved forecasts_test.csv (for Person B), forecast_metrics.csv, forecast_models.joblib")