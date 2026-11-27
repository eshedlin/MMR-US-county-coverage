"""Devil's advocate item 4: robustness of the independent-banding argument (no recovery in low-baseline counties).
Usage: python3 -I d4_banding.py /home/claude/rerun/county_classes.csv"""
import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
rng = np.random.default_rng(3)
k = pd.read_csv(sys.argv[1]); k = k[k.cls.notna()].copy()
q = k.dropna(subset=["SY2018_19", "SY2019_20"]).copy()
cut = lambda v: pd.cut(v, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"])
lg = lambda v: np.log(v.clip(0.5, 99.5) / (100 - v.clip(0.5, 99.5)))

def summ(df, bandcol, basecol, label):
    df = df.copy(); df["c"] = df["last"] - df[basecol]
    df["cw"] = df.c - df.groupby("State").c.transform("median")
    df["lgc"] = lg(df["last"]) - lg(df[basecol])
    df["unvax_ratio"] = (100 - df["last"]).clip(lower=0.1) / (100 - df[basecol]).clip(lower=0.1)
    g = df.groupby(bandcol, observed=True)
    t = pd.DataFrame({"n": g.size(), "median chg": g.c.median(), "mean chg": g.c.mean(), "within-state median": g.cw.median(),
                      "%up>=1": g.c.apply(lambda s: (s.round(6) >= 1).mean() * 100), "median logit chg": g.lgc.median(),
                      "median unvax ratio": g.unvax_ratio.median(), "top state (share)": g.State.agg(lambda s: f"{s.value_counts().index[0]} {s.value_counts(normalize=True).iloc[0]*100:.0f}%")})
    print(f"\n-- {label}"); print(t.round(2).to_string())
    return df

q["b18"] = cut(q.SY2018_19); q["b19"] = cut(q.SY2019_20); q["bmean"] = cut(q[["SY2018_19", "SY2019_20"]].mean(axis=1))
summ(q, "b19", "SY2019_20", "SAME-year banding: band 2019-20, change last-2019-20 (RTM-contaminated)")
summ(q, "b18", "SY2019_20", "ROUND-1 independent banding: band 2018-19, change last-2019-20 (all 16 states)")
x = q[~q.State.isin(["CA", "CO"])]
summ(x, "b18", "SY2019_20", "independent banding, excl CA CO")
summ(q, "b19", "SY2018_19", "REVERSED independent banding: band 2019-20, change last-2018-19 (all)")
summ(x, "b19", "SY2018_19", "REVERSED independent banding, excl CA CO")
summ(x[x.State != "CT"], "b18", "SY2019_20", "independent banding, excl CA CO CT")

# slope versions, within-state, with bootstrap by state; attenuation-corrected by reliability of the banding year
def slope_test(df, bvar, basecol, label):
    d = df.copy(); d["c"] = d["last"] - d[basecol]
    d["xb"] = d[bvar] - d.groupby("State")[bvar].transform("mean"); d["yc"] = d.c - d.groupby("State").c.transform("mean")
    sl = np.polyfit(d.xb, d.yc, 1)[0]
    # robust: Theil-Sen-ish via median regression IRLS
    X = np.column_stack([np.ones(len(d)), d.xb.to_numpy()]); w = np.ones(len(d))
    for _ in range(100):
        W = np.sqrt(w); yv = d.yc.to_numpy(); b = np.linalg.lstsq(X * W[:, None], yv * W, rcond=None)[0]; w = 1 / np.maximum(np.abs(yv - X @ b), 1e-3)
    gs = {s: g.index for s, g in d.groupby("State")}; S = list(gs); bt = []
    for _ in range(1000):
        idx = np.concatenate([gs[s] for s in rng.choice(S, len(S))]); bt.append(np.polyfit(d.xb.loc[idx], d.yc.loc[idx], 1)[0])
    # reliability of a single year within state: corr(x18, x19) within state
    a = df.SY2018_19 - df.groupby("State").SY2018_19.transform("mean"); bb = df.SY2019_20 - df.groupby("State").SY2019_20.transform("mean")
    rel = np.corrcoef(a, bb)[0, 1]
    print(f"  {label:52s} n={len(d)} OLS slope {sl:+.3f} [state-boot {np.percentile(bt,2.5):+.3f},{np.percentile(bt,97.5):+.3f}] "
          f"LAD slope {b[1]:+.3f}; within-state reliability {rel:.2f} -> disattenuated OLS {sl/rel:+.3f}")
print("\n== within-state slope of change on the INDEPENDENT year (negative = low counties decline more = no catch-up)")
slope_test(q, "SY2018_19", "SY2019_20", "x=18-19, chg=last-19-20, all")
slope_test(x, "SY2018_19", "SY2019_20", "x=18-19, chg=last-19-20, excl CA CO")
slope_test(q, "SY2019_20", "SY2018_19", "x=19-20, chg=last-18-19, all")
slope_test(x, "SY2019_20", "SY2018_19", "x=19-20, chg=last-18-19, excl CA CO")
xx = x[x[["SY2018_19", "SY2019_20"]].max(axis=1) < 99]
slope_test(xx, "SY2018_19", "SY2019_20", "x=18-19, excl CA CO, excl any pre >=99 (ceiling)")
# logit-scale within-state slope
d = x.copy(); d["lc"] = lg(d["last"]) - lg(d.SY2019_20); d["lb"] = lg(d.SY2018_19)
d["xb"] = d.lb - d.groupby("State").lb.transform("mean"); d["yc"] = d.lc - d.groupby("State").lc.transform("mean")
print(f"  logit scale, x=logit 18-19, chg=logit last - logit 19-20, excl CA CO: slope {np.polyfit(d.xb, d.yc, 1)[0]:+.3f} (negative = low counties lose LESS on odds scale)")
# noise by band of the independent year
print("\n== noise check: spread of (last-19-20) by 18-19 band, excl CA CO")
for b, g in x.groupby("b18", observed=True):
    c = g["last"] - g.SY2019_20
    print(f"  {b}: n={len(g)} IQR {c.quantile(.25):.1f} to {c.quantile(.75):.1f}; skew {c.skew():.2f}; share at last>=99: {(g['last']>=99).mean()*100:.0f}%")
