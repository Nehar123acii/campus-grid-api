import pandas as pd

out = []
def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s); print(s)

h = pd.read_excel("data/hourly_streams.xlsx")
h["timestamp"] = pd.to_datetime(h["timestamp"])
h["received_at"] = pd.to_datetime(h["received_at"])
say("== HOURLY ==")
say("range:", h.timestamp.min(), "to", h.timestamp.max())
say("buildings:", sorted(h.building_id.unique()))
say("duplicate (timestamp,building):", h.duplicated(["timestamp", "building_id"]).sum())
say("missing values:\n", h.isna().sum().to_string())
say("delay (hours):\n", ((h.received_at - h.timestamp).dt.total_seconds() / 3600).describe().to_string())
say("rows per building:\n", h.building_id.value_counts().sort_index().to_string())
say("describe:\n", h.describe().round(2).to_string())

b = pd.read_excel("data/buildings.xlsx")
say("\n== BUILDINGS ==\n", b[["building_id", "building_name", "building_type", "has_solar"]].to_string())

c = pd.read_excel("data/calendar.xlsx")
say("\n== CALENDAR ==")
say(c.day_type.value_counts().to_string())
say("events:\n", c[c.event.notna()][["date", "event"]].to_string())

m = pd.read_excel("data/manual_logs.xlsx")
say("\n== MANUAL LOGS ==")
for col in ["building", "stream", "unit", "logger_role"]:
    say(col, "->", m[col].value_counts(dropna=False).to_dict())
say("notes:", m.note.value_counts().head(10).to_dict())

wtr = pd.read_excel("data/waste_events_train.xlsx")
wte = pd.read_excel("data/waste_events_test.xlsx")
say("\n== WASTE ==")
say("categories:", wtr.category.value_counts().to_dict())
say("train range:", wtr.timestamp.min(), "to", wtr.timestamp.max())
say("test range:", wte.timestamp.min(), "to", wte.timestamp.max())
say("missing in train:\n", wtr.isna().sum().to_string())

with open("summary2.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))