import pandas as pd, numpy as np

ELEC_RS_PER_KWH = 8.0    # ASSUMED tariff, edit to your campus rate
WATER_RS_PER_L = 0.03    # ASSUMED, edit to your campus rate

ev = pd.read_csv("anomalies.csv", parse_dates=["start", "end", "date"])
if "excess_total" not in ev.columns:
    print("WARNING: anomalies.csv has no excess_total column, so costs will be blank.")
    print("Add the excess_total line to anomaly2.py, rerun it, then rerun this.\n")
    ev["excess_total"] = np.nan

b = pd.read_excel("data/buildings.xlsx")
NAME = dict(zip(b.building_id, b.building_name))

m = pd.read_csv("clean_manual_logs.csv", parse_dates=["logged_at"])
common = m.note.value_counts().head(6).index
notes = m[m.note.notna() & ~m.note.isin(common)]

# (cause prefix, urgency weight, [(action, why), ...] in priority order)
PLAYBOOK = [
    ("Persistent multi-day excess, probable leak", 1.5, [
        ("Inspect water lines and pipework in {name}; isolate the section if flow does not drop overnight",
         "Excess ran across {days} days (+{pct}% over forecast){night_txt}, which normal use does not explain"),
        ("Compare the water meter with a manual reading",
         "A faulty meter shows the same pattern, so this separates a meter error from a real leak"),
        ("Log the repair so savings can be measured",
         "Consumption before and after the fix gives a verified saving"),
    ]),
    ("Extreme spike", 0.8, [
        ("Check meter or analyzer calibration in {name} before acting on this reading",
         "The reading was +{pct}% over forecast, far beyond any plausible demand change, so a sensor fault is likelier than real use"),
        ("Exclude these hours from the training baseline",
         "Faulty readings would teach the model a wrong normal"),
    ]),
    ("Excess on non-working day", 1.2, [
        ("Send a shutdown checklist (AC, lights, lab equipment) to security or the warden of {name}",
         "Usage ran +{pct}% over forecast on a {daytype} for {hours} h, when normal demand is low"),
        ("Fit timers or auto cut-off on AC and lighting circuits",
         "Prevents the same waste on future holidays"),
    ]),
    ("Persistent multi-day excess, check meter", 1.2, [
        ("Inspect equipment and the meter in {name}",
         "Excess persisted over {days} days (+{pct}%), which points to equipment or metering and not a one-off"),
    ]),
    ("Overnight excess", 1.1, [
        ("Do an evening walk-through of {name} for equipment left running",
         "{night:.0%} of the flagged hours were overnight (+{pct}%), when the building should be near idle"),
    ]),
    ("Campus-wide shift", 0.6, [
        ("Check shared causes: weather, supply or tank refill schedule, scheduled maintenance",
         "{nb} buildings deviated at once, so a single-building fault is unlikely"),
        ("Review the forecast for a bias before acting on individual buildings",
         "A shift hitting many buildings can come from the model and not from real use"),
    ]),
    ("Event-driven surge", 0.5, [
        ("Pre-plan capacity for similar events in {name} (staffing, loads, bin pickups)",
         "The surge (+{pct}%) coincided with: {event}"),
        ("Add this event type to the forecast calendar",
         "The model then expects the surge next time and stops flagging it"),
    ]),
    ("Exam-period", 0.3, [
        ("No fault action; plan staffing and supply for exam weeks",
         "Higher demand (+{pct}%) fits the exam period"),
    ]),
    ("Unexplained", 0.7, [
        ("Send staff for a manual check of {name} and record the cause",
         "+{pct}% over forecast for {hours} h with no event, holiday or repeat pattern to explain it"),
    ]),
]

DOUBT = ("flicker", "calibrat", "stuck", "not sure", "looked odd")
METER_FAULT = ("Possible meter fault", 0.9, [
    ("Verify the {mname} meter in {name} against a manual reading before acting",
     "A manual log within 2 days says: \"{note}\", so the excess may be a measurement error"),
    ("Exclude these hours from the training baseline until the meter is confirmed",
     "Unconfirmed readings would teach the model a wrong normal"),
])

rows = []
for i, r in ev.iterrows():
    nb = notes[(notes.building_id == r.building_id) &
               (notes.logged_at >= r.start - pd.Timedelta(days=2)) &
               (notes.logged_at <= r.end + pd.Timedelta(days=2))]
    corro = " | ".join(nb.note.tolist())
    doubt = nb[nb.note.str.lower().apply(lambda s: any(w in s for w in DOUBT))]

    cause = r.cause
    prefix, w, acts = next((p for p in PLAYBOOK if cause.startswith(p[0])), PLAYBOOK[-1])
    is_leak = cause.startswith("Persistent multi-day excess, probable leak")

    if len(doubt) and not is_leak:
        prefix, w, acts = METER_FAULT
        cause = "Possible meter fault (a manual log questions this reading)"
    elif is_leak and r.excess_pct < 25:
        cause = "Possible leak or sustained overuse (low severity)"

    event = r.calendar_event if isinstance(r.calendar_event, str) else "no scheduled event"
    ctx = dict(name=NAME[r.building_id], pct=int(r.excess_pct), hours=int(r.hours),
               days=int(r.days_nearby), night=r.night_share, event=event,
               daytype=r.day_type, nb=int(r.n_bldgs),
               mname="electricity" if r.metric == "electricity_kwh" else "water",
               note=doubt.note.iloc[0] if len(doubt) else "",
               night_txt=(f", with {r.night_share:.0%} of flagged hours overnight"
                          if r.night_share >= 0.25 else ""))

    rate = ELEC_RS_PER_KWH if r.metric == "electricity_kwh" else WATER_RS_PER_L
    cost = r.excess_total * rate if pd.notna(r.excess_total) else np.nan
    priority = (min(r.excess_pct, 300) / 100) * r.hours * (1 + 0.25 * min(r.days_nearby, 5)) * w
    severity = "high" if r.excess_pct >= 60 else "medium" if r.excess_pct >= 25 else "low"
    if len(doubt):
        conf = "low (a manual log questions this meter)"
    elif corro:
        conf = "supported by a nearby manual log entry"
    else:
        conf = "meter only"

    for k, (act, why) in enumerate(acts, 1):
        rows.append({
            "event_id": i, "building_id": r.building_id, "building_name": ctx["name"],
            "metric": r.metric, "start": r.start, "end": r.end, "hours": r.hours,
            "excess_pct": r.excess_pct, "severity": severity, "cause": cause,
            "priority_score": round(priority, 1),
            "action_rank": k, "action": act.format(**ctx), "why": why.format(**ctx),
            "est_excess_cost_rs": round(cost) if pd.notna(cost) else None,
            "manual_log_corroboration": corro, "confidence": conf, "held_out": r.held_out,
        })

out = pd.DataFrame(rows)
order = out.drop_duplicates("event_id").sort_values("priority_score", ascending=False)
out["priority_rank"] = out.event_id.map({e: n for n, e in enumerate(order.event_id, 1)})
out = out.sort_values(["priority_rank", "action_rank"])
out.to_csv("recommendations.csv", index=False)

print("=== TOP 6 HELD-OUT EVENTS (Oct 7 onward), first action only ===")
top = out[(out.held_out) & (out.action_rank == 1)].head(6)
for _, r in top.iterrows():
    print(f"\n#{r.priority_rank} {r.building_name} | {r.metric} | {r.start:%b %d %H:%M} | {r.hours:.0f} h | +{r.excess_pct:.0f}% | severity: {r.severity}")
    print("  cause :", r.cause)
    print("  action:", r.action)
    print("  why   :", r.why)
    print("  confidence:", r.confidence, "| est. excess cost (assumed tariff): Rs", r.est_excess_cost_rs)

print("\n=== B06 LEAK, top event ===")
lk = out[(out.building_id == "B06") & (out.metric == "water_litres") & (out.action_rank == 1)].head(1)
for _, r in lk.iterrows():
    print(f"#{r.priority_rank} {r.start:%b %d} | {r.cause}\n  {r.action}\n  {r.why}")
    print(f"  manual log: {r.manual_log_corroboration or 'none'}\n  est. excess cost: Rs {r.est_excess_cost_rs}")
print("\nSaved recommendations.csv:", out.event_id.nunique(), "events,", len(out), "actions")