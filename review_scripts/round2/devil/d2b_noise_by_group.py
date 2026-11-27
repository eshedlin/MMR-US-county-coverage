"""Noise sigma_e from the 3 most recent years (22-23, 23-24, 24-25), by baseline-years group and band; within-state, winsorized.
Usage: python3 -I d2b_noise_by_group.py /home/claude/rerun/county_classes.csv"""
import sys, numpy as np, pandas as pd
k = pd.read_csv(sys.argv[1]); k = k[k.cls.notna()].copy()
k["nb"] = k[["SY2018_19", "SY2019_20"]].notna().sum(axis=1)
p = k.dropna(subset=["SY2022_23", "SY2023_24", "SY2024_25"]).copy()
p["band"] = pd.cut(p.base, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"])
dm = lambda s, g: s - s.groupby(g).transform("mean"); w = lambda s: s.clip(s.quantile(.02), s.quantile(.98))
for nb, g0 in p.groupby("nb"):
    for b, g in g0.groupby("band", observed=True):
        d1 = w(dm(g.SY2023_24 - g.SY2022_23, g.State)); d2 = w(dm(g.SY2024_25 - g.SY2023_24, g.State))
        c = np.cov(d1, d2)[0, 1]; v = (d1.var() + d2.var()) / 2
        print(f"nb={nb} band {b:6s} n={len(g):4d} Var(d)={v:6.2f} cov={c:6.2f} sigma_e={np.sqrt(max(-c,0)):.2f} sd_u={np.sqrt(max(v+2*c,0)):.2f}")
print("states nb=1:", p[p.nb==1].State.value_counts().to_dict())
for st, g in p[p.nb==1].groupby("State"):
    d1 = dm(g.SY2023_24 - g.SY2022_23, g.State); d2 = dm(g.SY2024_25 - g.SY2023_24, g.State)
    print(f"  {st}: n={len(g)} sigma_e={np.sqrt(max(-np.cov(d1,d2)[0,1],0)):.2f}  median base {g.base.median():.1f}")
