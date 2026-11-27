"""Devil's advocate item 1: is the pre-COVID one-year change a valid noise estimate?
Usage: python3 -I d1_noise.py /home/claude/rerun/county_classes.csv"""
import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 200)
k0 = pd.read_csv(sys.argv[1])
k = k0[k0.cls.notna()].copy()
Y = ["SY2017_18", "SY2018_19", "SY2019_20", "SY2020_21", "SY2021_22", "SY2022_23", "SY2023_24", "SY2024_25"]
def rsd(v):
    v = np.asarray(v); v = v[~np.isnan(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v)))
def bandof(v): return pd.cut(v, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"])

p = k.dropna(subset=["SY2018_19", "SY2019_20"]).copy()
p["d"] = p.SY2019_20 - p.SY2018_19
p["dw"] = p.d - p.groupby("State").d.transform("median")       # within-state (removes state-wide shifts/artefacts)
p["band"] = bandof(p[["SY2018_19", "SY2019_20"]].mean(axis=1))
print("n pairs", len(p), "states", p.State.nunique())
print("state median one-year change 18-19->19-20:", p.groupby("State").d.median().round(2).sort_values().to_dict())

def row(df, col):
    g = df.groupby("band", observed=True)[col]
    out = {"all_SD": df[col].std(), "all_rSD": rsd(df[col])}
    for b, s in g: out[f"{b}_SD"] = s.std(); out[f"{b}_rSD"] = rsd(s)
    out["n"] = len(df)
    return out
rows = {}
rows["raw d, all (round-1)"] = row(p, "d")
q = p[~p.State.isin(["CA", "CO"])]
rows["raw d, excl CA CO"] = row(q, "d")
rows["within-state d, all"] = row(p, "dw")
rows["within-state d, excl CA CO"] = row(q, "dw")
# earlier pair 17-18 -> 18-19 (CA/CO jump not in it)
p2 = k.dropna(subset=["SY2017_18", "SY2018_19"]).copy()
p2["d"] = p2.SY2018_19 - p2.SY2017_18; p2["dw"] = p2.d - p2.groupby("State").d.transform("median")
p2["band"] = bandof(p2[["SY2017_18", "SY2018_19"]].mean(axis=1))
rows["raw d 17-18->18-19"] = row(p2, "d")
rows["within-state d 17-18->18-19"] = row(p2, "dw")
print("\nSD of one-year change (SD and robust MAD-SD), by band of the 2-yr mean:")
print(pd.DataFrame(rows).T.round(2).to_string())

# ---- separate pure measurement noise from real year-to-year change using 3 pre years
# model: x_t = T_t + e_t, e iid (var s2e), T random walk with increments u (var s2u), u indep of e
# Var(d1)=2 s2e + s2u ; Var(lag2)=2 s2e + 2 s2u ; Cov(d1,d2) = -s2e
p3 = k.dropna(subset=["SY2017_18", "SY2018_19", "SY2019_20"]).copy()
print(f"\n3 consecutive pre years: n={len(p3)}, states {sorted(p3.State.unique())}")
def decomp(df, label, within=True, robust=False):
    d1 = df.SY2018_19 - df.SY2017_18; d2 = df.SY2019_20 - df.SY2018_19; l2 = df.SY2019_20 - df.SY2017_18
    if within:
        d1 = d1 - d1.groupby(df.State).transform("mean"); d2 = d2 - d2.groupby(df.State).transform("mean"); l2 = l2 - l2.groupby(df.State).transform("mean")
    if robust:  # winsorize at 1st/99th pct to tame artefacts
        w = lambda s: s.clip(s.quantile(.02), s.quantile(.98))
        d1, d2, l2 = w(d1), w(d2), w(l2)
    c = np.cov(d1, d2)[0, 1]; v1 = (d1.var() + d2.var()) / 2; vl = l2.var()
    s2e_cov = -c; s2u_cov = v1 - 2 * s2e_cov
    s2e_lag = (2 * v1 - vl) / 2; s2u_lag = vl - v1
    print(f"  {label:38s} n={len(df):4d} Var(d1)={v1:6.2f} Var(lag2)={vl:6.2f} Cov(d1,d2)={c:6.2f} corr={c/v1:5.2f} | "
          f"sigma_e(cov)={np.sqrt(max(s2e_cov,0)):.2f} sd_u(cov)={np.sqrt(max(s2u_cov,0)):.2f} | sigma_e(lag)={np.sqrt(max(s2e_lag,0)):.2f} sd_u(lag)={np.sqrt(max(s2u_lag,0)):.2f}")
    return np.sqrt(max(s2e_cov, 0))
p3["band"] = bandof(p3[["SY2017_18", "SY2018_19", "SY2019_20"]].mean(axis=1))
for lab, df in [("all", p3), ("excl CA CO", p3[~p3.State.isin(["CA", "CO"])])]:
    for within in [False, True]:
        for rob in [False, True]:
            decomp(df, f"{lab} within={within} wins={rob}", within, rob)
print(" by band (excl CA CO, within-state, winsorized):")
for b, g in p3[~p3.State.isin(["CA", "CO"])].groupby("band", observed=True):
    decomp(g, f"band {b}", True, True)
print(" by band (excl CA CO, within-state, not winsorized):")
for b, g in p3[~p3.State.isin(["CA", "CO"])].groupby("band", observed=True):
    decomp(g, f"band {b}", True, False)

# ---- noise from post-COVID 'plateau' years too: 3 consecutive recent years 22-23, 23-24, 24-25
p4 = k.dropna(subset=["SY2022_23", "SY2023_24", "SY2024_25"]).copy()
p4 = p4.rename(columns={"SY2022_23": "a", "SY2023_24": "b", "SY2024_25": "c"})
p4["band"] = bandof(p4[["a", "b", "c"]].mean(axis=1))
def decomp2(df, label):
    d1 = df.b - df.a; d2 = df.c - df.b
    d1 = d1 - d1.groupby(df.State).transform("mean"); d2 = d2 - d2.groupby(df.State).transform("mean")
    w = lambda s: s.clip(s.quantile(.02), s.quantile(.98)); d1, d2 = w(d1), w(d2)
    c = np.cov(d1, d2)[0, 1]; v1 = (d1.var() + d2.var()) / 2
    print(f"  {label:38s} n={len(df):4d} Var(d)={v1:6.2f} Cov={c:6.2f} corr={c/v1:5.2f} sigma_e(cov)={np.sqrt(max(-c,0)):.2f} sd_u={np.sqrt(max(v1-2*max(-c,0),0)):.2f}")
print("\nrecent years 22-23/23-24/24-25 (all states but CT, within-state, winsorized):")
decomp2(p4, "all")
for b, g in p4.groupby("band", observed=True): decomp2(g, f"band {b}")
