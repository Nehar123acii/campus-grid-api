import pandas as pd, numpy as np

out = []
def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s); print(s)

# ---------- 1. HOURLY METERS ----------
h = pd.read_excel("data/hourly_streams.xlsx")
h["timestamp"] = pd.to_datetime(h["timestamp"])
h["received_at"] = pd.to_datetime(h["received_at"])
h["delay_h"] = (h.received_at - h.timestamp).dt.total_seconds() / 3600
say("== HOURLY ==")
say("rows before:", len(h))

# real outage episodes: readings that arrived more than 3 hours late
late = h[h.delay_h > 3]
say("late readings (>3h):", len(late))
if len(late):
    say(late.groupby("building_id").timestamp.agg(["count", "min", "max"]).to_string())

# duplicates: do the copies disagree?
d = h[h.duplicated(["timestamp", "building_id"], keep=False)]
conflicts = (d.groupby(["timestamp", "building_id"])["electricity_kwh"]
               .nunique(dropna=False).gt(1).sum())
say("duplicate groups with conflicting values:", conflicts)

# keep the most recently received copy
h = h.sort_values("received_at").drop_duplicates(["timestamp", "building_id"], keep="last")
say("rows after dedupe:", len(h))

# full hourly grid per building, so missing hours become visible
grid = pd.MultiIndex.from_product(
    [sorted(h.building_id.unique()),
     pd.date_range(h.timestamp.min(), h.timestamp.max(), freq="h")],
    names=["building_id", "timestamp"])
h = h.set_index(["building_id", "timestamp"]).reindex(grid).reset_index()
cols = ["electricity_kwh", "water_litres", "solar_generation_kwh", "footfall"]
h["was_missing"] = h["electricity_kwh"].isna()
say("hours missing or blank:", int(h.was_missing.sum()))

# fill only short gaps (up to 3 hours); longer gaps stay blank on purpose
h = h.sort_values(["building_id", "timestamp"])
h[cols] = h.groupby("building_id")[cols].transform(
    lambda s: s.interpolate(limit=3, limit_area="inside"))
h["imputed"] = h.was_missing & h.electricity_kwh.notna()
say("filled by interpolation:", int(h.imputed.sum()))
say("still blank after filling:", int(h.electricity_kwh.isna().sum()))
h.to_csv("clean_hourly.csv", index=False)

# ---------- 2. MANUAL LOGS ----------
m = pd.read_excel("data/manual_logs.xlsx")
m["logged_at"] = pd.to_datetime(m["logged_at"])
NAME = {
    "b01": "B01", "academic block a": "B01", "acad block a": "B01", "block a": "B01",
    "b02": "B02", "academic block b": "B02", "acad block b": "B02", "block b": "B02",
    "b03": "B03", "central library": "B03", "library": "B03", "central lib": "B03",
    "b04": "B04", "engineering labs & innovation centre": "B04", "innovation centre": "B04",
    "eng labs": "B04", "labs": "B04",
    "b05": "B05", "boys hostel": "B05", "boys hostle": "B05", "bh": "B05",
    "b06": "B06", "girls hostel": "B06", "girls hostle": "B06", "gh": "B06",
    "b07": "B07", "food court": "B07", "fc": "B07", "canteen": "B07",
    "b08": "B08", "admin block": "B08", "admin": "B08",
}
m["building_id"] = m.building.str.lower().str.strip().map(NAME)
say("\n== MANUAL LOGS ==")
say("unmapped building names:", m[m.building_id.isna()].building.unique().tolist())

m["value_std"], m["unit_std"] = m["value"], m["unit"]
kl = m.unit == "kL"
m.loc[kl, "value_std"] = m.loc[kl, "value"] * 1000
m.loc[kl, "unit_std"] = "L"
m.loc[m.unit == "units", "unit_std"] = "kWh"
say("units after cleaning:", m.unit_std.value_counts(dropna=False).to_dict())

common = m.note.value_counts().head(6).index
inc = m[m.note.notna() & ~m.note.isin(common)]
say("\nINCIDENT NOTES (ground truth):")
say(inc[["logged_at", "building_id", "stream", "value_std", "unit_std", "note"]].to_string())
m.to_csv("clean_manual_logs.csv", index=False)

with open("summary3.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
say("\nSaved clean_hourly.csv, clean_manual_logs.csv, summary3.txt")