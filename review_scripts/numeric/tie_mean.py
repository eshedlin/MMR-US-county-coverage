import csv, sys
import pandas as pd
k = pd.read_csv("/home/claude/rerun/county_classes.csv")
TR = ["SY2020_21", "SY2021_22", "SY2022_23"]; PRE = ["SY2018_19", "SY2019_20"]
b = k[PRE].mean(axis=1); t = k[TR].mean(axis=1)
dt = t - b
m = (dt + 1).abs() < 1e-6
print(k.loc[m, ["State", "County"] + PRE + TR].assign(dt=dt[m].map(repr)).to_string())
# my pure python
for i in k.index[m]:
    tv = [k.at[i, y] for y in TR if pd.notna(k.at[i, y])]; bv = [k.at[i, y] for y in PRE if pd.notna(k.at[i, y])]
    print(repr(sum(tv)/len(tv) - sum(bv)/len(bv)))
# float-tie count for main thresholds, last-base and trough-base, all thresholds
l = pd.Series([r["SY2023_24"] if r.State == "CT" else r["SY2024_25"] for _, r in k.iterrows()])
for thr in [1, 1.5, 2]:
    for nm, d in [("last", l - b), ("trough_min", k[TR].min(axis=1) - b), ("trough_mean", dt)]:
        near = ((d.abs() - thr).abs() < 1e-6).sum(); wrong = (((d.abs() - thr).abs() < 1e-6) & (d.abs() < thr)).sum()
        print(thr, nm, "near-ties", near, "evaluated below thr", wrong)
