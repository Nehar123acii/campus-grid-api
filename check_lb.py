import pandas as pd
d = pd.read_csv("leaderboard_daily.csv")
x = d[d.building_id.isin(["B01", "B02"])].pivot(index="date", columns="building_id", values="saving")
print(x.round(4))
print("identical every day:", bool((x.B01 - x.B02).abs().max() < 1e-9))