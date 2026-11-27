"""Devil's advocate item 2: real between- and within-state heterogeneity in post-COVID change beyond noise?
Usage: python3 -I d2_hetero.py /home/claude/rerun/county_classes.csv"""
import sys
import numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 200)
k0 = pd.read_csv(sys.argv[1])
k = k0[k0.cls.notna()].copy()
PRE = ["SY2018_19", "SY2019_20"]
k["nb"] = k[PRE].notna().sum(axis=1)
k["chg"] = k["last"] - k.base
bandof = lambda v: pd.cut(v, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"])
k["band"] = bandof(k.base)
rng = np.random.default_rng(11)
wins = lambda s: s.clip(s.quantile(.02), s.quantile(.98))
dm = lambda s, g: s - s.groupby(g).transform("mean")

# noise sigma_e by band: (a) round-1 robust (MAD of 18-19->19-20 / sqrt2), (b) variance-based from 3-yr covariance
# (pre, excl CA CO, within-state, winsorized, from d1_noise.py), (c) same, not winsorized
SIG = {"r1_robust": {"<90": 4.19, "90-95": 1.73, "95+": 1.15},
       "cov_wins": {"<90": 7.95, "90-95": 2.20, "95+": 1.29},
       "cov_raw": {"<90": 8.60, "90-95": 2.40, "95+": 1.39},
       "cov_recent_wins": {"<90": 4.93, "90-95": 2.09, "95+": 1.24}}

# ---------- A. within-state variance of chg vs expected noise variance
print("== A. within-state variance of change (last-base) vs expected pure-noise variance sigma_e^2 (1 + 1/nb)")
k["cw"] = dm(k.chg, k.State)
res = []
for b, g in k.groupby("band", observed=True):
    r = {"band": b, "n": len(g), "obs_var_within": g.cw.var(), "obs_var_within_wins": wins(g.cw).var(),
         "obs_rSD^2": (1.4826 * np.median(np.abs(g.cw - g.cw.median()))) ** 2}
    for nm, s in SIG.items():
        r[f"noise_{nm}"] = (s[b] ** 2 * (1 + 1 / g.nb)).mean()
    res.append(r)
print(pd.DataFrame(res).round(2).to_string(index=False))

# ---------- B. clean test free of any noise model: two change measures with disjoint error terms
# A = x(2022-23) - x(2018-19); B = x(2024-25) - x(2019-20)   (counties with both pre years; CT excluded: no 2024-25)
print("\n== B. covariance of two change measures that share no observation (noise-free test of county-specific real change)")
q = k.dropna(subset=["SY2018_19", "SY2019_20", "SY2022_23", "SY2024_25"]).copy()
def covtest(df, a, b, label):
    A = dm(df[a], df.State); B = dm(df[b], df.State)
    c = np.cov(A, B)[0, 1]; rP = np.corrcoef(A, B)[0, 1]; rS = stats.spearmanr(A, B)[0]
    Aw, Bw = wins(A), wins(B); cw = np.cov(Aw, Bw)[0, 1]
    # state-cluster bootstrap CI of Spearman
    gs = {s: g.index for s, g in df.groupby("State")}; S = list(gs); bt = []
    for _ in range(1000):
        idx = np.concatenate([gs[s] for s in rng.choice(S, len(S))])
        bt.append(stats.spearmanr(A.loc[idx], B.loc[idx])[0])
    print(f"  {label:34s} n={len(df):4d} cov={c:6.2f} (wins {cw:5.2f}) -> SD real county-specific change ~{np.sqrt(max(cw,0)):.2f}; "
          f"Pearson {rP:.2f} Spearman {rS:.2f} [cluster-boot 95% {np.percentile(bt,2.5):.2f},{np.percentile(bt,97.5):.2f}]")
covtest(q, "dA", "dB", "all") if False else None
q["dA"] = q.SY2022_23 - q.SY2018_19; q["dB"] = q.SY2024_25 - q.SY2019_20
covtest(q, "dA", "dB", "all 15 states")
covtest(q[~q.State.isin(["CA", "CO"])], "dA", "dB", "excl CA CO")
for b, g in q.groupby(bandof(q[PRE].mean(axis=1)), observed=True):
    covtest(g, "dA", "dB", f"band {b}")
# placebo version of the same test in a window with no COVID: A = x18-19 - x17-18 ; B = x19-20 - ... not possible with disjoint obs; skip

# ---------- C. persistence: does (2023-24 - base) predict (2024-25 - base) beyond the shared base error?
print("\n== C. persistence: slope of (x24-25 - base) on (x23-24 - base), within state")
print("   pure noise + common shift predicts slope = (s2/nb)/(s2/nb + s2) = 1/3 (nb=2), 1/2 (nb=1)")
r = k[(k.State != "CT")].dropna(subset=["SY2023_24", "SY2024_25"]).copy()
for nb, g in r.groupby("nb"):
    X = dm(g.SY2023_24 - g.base, g.State); Yv = dm(g.SY2024_25 - g.base, g.State)
    sl = np.polyfit(wins(X), wins(Yv), 1)[0]; slr = np.polyfit(X, Yv, 1)[0]
    print(f"  nb={nb} n={len(g)} slope raw {slr:.2f}, winsorized {sl:.2f}; expected under noise {1/3 if nb==2 else 1/2:.2f}")
    for b, gg in g.groupby("band", observed=True):
        X = dm(gg.SY2023_24 - gg.base, gg.State); Yv = dm(gg.SY2024_25 - gg.base, gg.State)
        print(f"     band {b}: n={len(gg)} slope winsorized {np.polyfit(wins(X), wins(Yv), 1)[0]:.2f}")
# single-baseline states: noise-corrected covariance of A=x23-24 - x19-20 and B=x24-25 - x19-20 minus Var(e19)
print("  single-baseline counties: Cov(A,B) - sigma_e^2 (should be ~0 if no real county-specific change)")
s1 = r[r.nb == 1]
for b, g in s1.groupby("band", observed=True):
    A = wins(dm(g.SY2023_24 - g.SY2019_20, g.State)); B = wins(dm(g.SY2024_25 - g.SY2019_20, g.State))
    c = np.cov(A, B)[0, 1]
    print(f"     band {b}: n={len(g)} Cov {c:.2f}; minus sigma_e^2 (recent cov est {SIG['cov_recent_wins'][b]**2:.2f}) = {c - SIG['cov_recent_wins'][b]**2:.2f}"
          f"; minus round-1 sigma_e^2 ({SIG['r1_robust'][b]**2:.2f}) = {c - SIG['r1_robust'][b]**2:.2f}")

# ---------- D. between-state heterogeneity (random-effects, DerSimonian-Laird on state means of chg)
print("\n== D. between-state heterogeneity in mean change")
def dl(df, col):
    g = df.groupby("State")[col]; m = g.mean(); v = g.var() / g.size(); w = 1 / v
    mu = (w * m).sum() / w.sum(); Q = (w * (m - mu) ** 2).sum(); dfree = len(m) - 1
    tau2 = max(0, (Q - dfree) / (w.sum() - (w ** 2).sum() / w.sum()))
    return m, mu, Q, dfree, tau2, 1 - stats.chi2.cdf(Q, dfree)
for lab, df in [("all 22", k), ("excl CA CT", k[~k.State.isin(["CA", "CT"])]), ("excl CA CO CT", k[~k.State.isin(["CA", "CO", "CT"])]),
                ("both-year states only, excl CA CO CT", k[(k.nb == 2) & ~k.State.isin(["CA", "CO", "CT"])])]:
    m, mu, Q, dfree, tau2, pv = dl(df, "chg")
    sm = df.groupby("State").chg.median()
    print(f"  {lab:38s} states={len(m)} Q={Q:6.1f} df={dfree} p={pv:.1e} tau(SD of true state means)={np.sqrt(tau2):.2f}; "
          f"I2={max(0,(Q-dfree)/Q)*100:.0f}%; state medians range {sm.min():.1f} to {sm.max():.1f}, IQR {sm.quantile(.25):.1f} to {sm.quantile(.75):.1f}")
print("  state median change:", k.groupby("State").chg.median().round(1).sort_values().to_dict())
# share of total variance of chg that is between-state
tot = k.chg.var(); within = k.cw.var()
print(f"  total var {tot:.1f}; within-state var {within:.1f}; between share {(1-within/tot)*100:.0f}%")
