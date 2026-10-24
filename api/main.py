import json, os, time
import numpy as np, pandas as pd, joblib
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
def p(f): return os.path.join(HERE, f)
def rec(df): return json.loads(df.to_json(orient="records", date_format="iso"))

app = FastAPI(title="Campus Resource Grid API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

F = pd.read_csv(p("forecasts_test.csv"), parse_dates=["origin_time", "target_time"])
A = pd.read_csv(p("anomalies.csv"), parse_dates=["start", "end", "date"])
R = pd.read_csv(p("recommendations.csv"), parse_dates=["start", "end"])
LB = pd.read_csv(p("leaderboard_summary.csv"))
M = pd.read_csv(p("forecast_metrics.csv"))
B = pd.read_csv(p("buildings.csv"))
W = joblib.load(p("waste_model.joblib"))
BASE = ["weight_g", "moisture_pct", "metal_signal", "transparency", "acoustic_hardness"]

def feats(d):
    x = d[BASE].copy()
    ts = pd.to_datetime(d.timestamp)
    x["hour"] = ts.dt.hour
    x["bin_num"] = d.bin_id.str.extract(r"BIN(\d+)")[0].astype(float)
    x["bcode"] = d.building_id.str[1:].astype(int)
    x["missing_n"] = d[BASE].isna().sum(axis=1)
    x["log_weight"] = np.log1p(d.weight_g)
    x["moist_x_trans"] = d.moisture_pct * d.transparency
    x["metal_x_hard"] = d.metal_signal * d.acoustic_hardness
    x["hard_minus_trans"] = d.acoustic_hardness - d.transparency
    x["moist_per_g"] = d.moisture_pct / (d.weight_g + 1)
    return x

@app.get("/health")
def health():
    return {"ok": True, "forecast_window": [str(F.target_time.min()), str(F.target_time.max())]}

@app.get("/buildings")
def buildings():
    return rec(B)

@app.get("/metrics")
def metrics():
    return rec(M)

@app.get("/leaderboard")
def leaderboard():
    return rec(LB)

@app.get("/anomalies")
def anomalies(building_id: Optional[str] = None, held_out_only: bool = False):
    d = A
    if building_id: d = d[d.building_id == building_id]
    if held_out_only: d = d[d.held_out]
    return rec(d)

@app.get("/recommendations")
def recommendations(building_id: Optional[str] = None, held_out_only: bool = False, top: int = 20):
    d = R
    if building_id: d = d[d.building_id == building_id]
    if held_out_only: d = d[d.held_out]
    ids = d.drop_duplicates("event_id").sort_values("priority_rank").head(top).event_id
    return rec(d[d.event_id.isin(ids)].sort_values(["priority_rank", "action_rank"]))

@app.get("/forecast")
def forecast(building_id: str, metric: str = "electricity_kwh", as_of: Optional[str] = None):
    d = F[(F.building_id == building_id) & (F.metric == metric)]
    if d.empty:
        raise HTTPException(404, "No forecast for that building/metric")
    t = pd.Timestamp(as_of).floor("h") if as_of else d.origin_time.max()
    d = d[d.origin_time == t]
    if d.empty:
        raise HTTPException(404, "as_of outside the precomputed window (Oct 6 - Oct 17)")
    return {"source": "precomputed by the trained model", "as_of": str(t),
            "forecast": rec(d[["horizon_h", "target_time", "actual", "p10", "p50", "p90"]])}

class Check(BaseModel):
    building_id: str
    metric: str
    target_time: str
    actual: float

@app.post("/analyze")
def analyze(c: Check):
    t = pd.Timestamp(c.target_time).floor("h")
    d = F[(F.building_id == c.building_id) & (F.metric == c.metric) &
          (F.horizon_h == 6) & (F.target_time == t)]
    if d.empty:
        raise HTTPException(404, "No stored band for that hour (use Oct 7 - Oct 17)")
    r = d.iloc[0]
    return {"flagged": bool(c.actual > r.p90), "actual": c.actual,
            "expected": round(float(r.p50), 2), "upper_band": round(float(r.p90), 2),
            "excess_pct": round(100 * (c.actual / r.p50 - 1), 1) if r.p50 > 0 else None}

class Waste(BaseModel):
    weight_g: Optional[float] = None
    moisture_pct: Optional[float] = None
    metal_signal: Optional[float] = None
    transparency: Optional[float] = None
    acoustic_hardness: Optional[float] = None
    timestamp: str = "2026-10-10 12:00:00"
    bin_id: str = "B01-BIN1"
    building_id: str = "B01"

@app.post("/classify")
def classify(w: Waste):
    t0 = time.perf_counter()
    row = pd.DataFrame([w.dict()])
    for c in BASE:
        row[c] = row[c].astype(float)
    x = feats(row)[W["feats"]]
    pr = W["model"].predict_proba(x)[0]
    i = int(pr.argmax())
    return {"category": W["classes"][i], "confidence": round(float(pr[i]), 3),
            "needs_review": bool(pr[i] < 0.7),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}