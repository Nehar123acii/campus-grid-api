import pandas as pd, glob, os

print("Running from:", os.getcwd())
files = sorted(glob.glob("data/*.xlsx"))
print("Excel files found:", len(files))

for f in files:
    sheets = pd.read_excel(f, sheet_name=None)   # reads every sheet
    for name, d in sheets.items():
        print("=" * 60)
        print(f, "| sheet:", name, "| rows:", d.shape[0], "| columns:", d.shape[1])
        print(d.dtypes)
        print(d.head(3).to_string())
        print()