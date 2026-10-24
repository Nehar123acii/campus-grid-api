import pandas as pd, numpy as np, time, os, joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

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

tr = pd.read_excel("data/waste_events_train.xlsx")
te = pd.read_excel("data/waste_events_test.xlsx")
classes = sorted(tr.category.unique())
y = tr.category.map({c: i for i, c in enumerate(classes)})
X = feats(tr)

# same held-out split as before, so numbers are comparable
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# validation slice for tuning, taken from the training part only
Xt, Xv, yt, yv = train_test_split(Xa, ya, test_size=0.2, stratify=ya, random_state=1)

CONFIGS = [
    dict(num_leaves=15, learning_rate=0.05, n_estimators=300, min_child_samples=20),
    dict(num_leaves=31, learning_rate=0.05, n_estimators=400, min_child_samples=20),
    dict(num_leaves=31, learning_rate=0.03, n_estimators=600, min_child_samples=10),
    dict(num_leaves=63, learning_rate=0.03, n_estimators=500, min_child_samples=10),
]
def make(c):
    return lgb.LGBMClassifier(class_weight="balanced", subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=42, verbose=-1, **c)

best, best_acc = None, -1
for c in CONFIGS:
    a = accuracy_score(yv, make(c).fit(Xt, yt).predict(Xv))
    print("validation accuracy", round(a, 3), c)
    if a > best_acc:
        best, best_acc = c, a

m = make(best).fit(Xa, ya)
P = m.predict_proba(Xb)
pred = P.argmax(1)
print("\nBEST CONFIG:", best)
print("HELD-OUT accuracy:", round(accuracy_score(yb, pred), 3),
      "| macro-F1:", round(f1_score(yb, pred, average="macro"), 3))

conf = P.max(1)
print("\nCONFIDENCE GATE (held-out):")
for t in [0.5, 0.6, 0.7, 0.8]:
    keep = conf >= t
    print(f"  threshold {t}: accuracy on confident events {accuracy_score(yb[keep], pred[keep]):.3f}"
          f" | confident share {keep.mean():.1%} | sent to human review {1 - keep.mean():.1%}")

row = Xb.iloc[[0]]
t0 = time.perf_counter()
for _ in range(200):
    m.predict_proba(row)
print(f"\nLatency per event: {(time.perf_counter() - t0) / 200 * 1000:.2f} ms")

joblib.dump({"model": m, "classes": classes, "feats": list(X.columns)}, "waste_model.joblib")
print("Model size:", round(os.path.getsize("waste_model.joblib") / 1024, 1), "KB")

Pt = m.predict_proba(feats(te))
te["predicted_category"] = [classes[i] for i in Pt.argmax(1)]
te["confidence"] = Pt.max(1).round(3)
te["needs_review"] = te.confidence < 0.7
te.to_excel("waste_events_test_predicted.xlsx", index=False)
print("Saved waste_events_test_predicted.xlsx (with confidence and needs_review)")