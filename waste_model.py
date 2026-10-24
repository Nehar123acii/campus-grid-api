import pandas as pd, numpy as np, time, os, joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

FEATS = ["weight_g", "moisture_pct", "metal_signal", "transparency", "acoustic_hardness"]

tr = pd.read_excel("data/waste_events_train.xlsx")
te = pd.read_excel("data/waste_events_test.xlsx")

classes = sorted(tr.category.unique())
cmap = {c: i for i, c in enumerate(classes)}
X, y = tr[FEATS], tr.category.map(cmap)

Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Baseline: small decision tree (needs gaps filled)
dt = DecisionTreeClassifier(max_depth=6, random_state=42).fit(Xa.fillna(-1), ya)
dt_pred = dt.predict(Xb.fillna(-1))

# Main model: LightGBM (handles missing values itself)
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=31,
                       class_weight="balanced", random_state=42, verbose=-1)
m.fit(Xa, ya)
pred = m.predict(Xb)

print("Decision tree accuracy :", round(accuracy_score(yb, dt_pred), 3))
print("LightGBM accuracy      :", round(accuracy_score(yb, pred), 3))
print("LightGBM macro-F1      :", round(f1_score(yb, pred, average="macro"), 3))
print("\n", classification_report(yb, pred, target_names=classes, digits=3))
print("Confusion matrix (rows=true, cols=predicted):")
print(pd.DataFrame(confusion_matrix(yb, pred), index=classes, columns=classes))

# Edge latency: one event at a time
row = Xb.iloc[[0]]
t = time.perf_counter()
for _ in range(200):
    m.predict_proba(row)
ms = (time.perf_counter() - t) / 200 * 1000
print(f"\nLatency per event: {ms:.2f} ms (requirement: under 2000 ms)")

# Save model and size
joblib.dump({"model": m, "classes": classes, "feats": FEATS}, "waste_model.joblib")
print("Model size:", round(os.path.getsize("waste_model.joblib") / 1024, 1), "KB")

# Predict the unlabeled test file for B's live replay
proba = m.predict_proba(te[FEATS])
te["predicted_category"] = [classes[i] for i in proba.argmax(1)]
te["confidence"] = proba.max(1).round(3)
te.to_excel("waste_events_test_predicted.xlsx", index=False)
print("Saved waste_events_test_predicted.xlsx for Person B")