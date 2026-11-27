"""Devil's advocate item 5: is the pair-permutation p = 1/231 for CA+CT sound given CA's 2019-20 jump?
Effect = OLS coefficient of treated dummy in chg ~ (base-95) + treated, as round 1.
Usage: python3 -I d5_pairperm.py /home/claude/rerun/county_classes.csv"""
import sys, itertools
import numpy as np, pandas as pd
k = pd.read_csv(sys.argv[1]); k = k[k.cls.notna()].copy()

def eff(df, base, last, sts):
    y = (df[last] - df[base]).to_numpy(); b = df[base].to_numpy(); t = df.State.isin(sts).astype(float).to_numpy()
    ok = ~np.isnan(y) & ~np.isnan(b)
    X = np.column_stack([np.ones(ok.sum()), b[ok] - 95, t[ok]])
    return np.linalg.lstsq(X, y[ok], rcond=None)[0][2]

k["base19"] = k.SY2019_20
k["last_c"] = np.where(k.State == "CT", k.SY2023_24, k["last"])      # as main
k["last23"] = k.SY2023_24                                            # common last year for everyone
def pairtest(df, base, last, label, target=("CA", "CT")):
    states = sorted(df.State.unique())
    pe = pd.Series({pr: eff(df, base, last, set(pr)) for pr in itertools.combinations(states, 2)})
    obs = pe[tuple(sorted(target))]
    rank = (pe >= obs).sum()
    top = pe.sort_values(ascending=False).head(4).round(2)
    print(f"  {label:58s} pairs={len(pe)} obs={obs:5.2f} rank={rank} p={rank/len(pe):.4f} | top: {dict(top)}")
    return pe
def singletest(df, base, last, st, label):
    states = sorted(df.State.unique())
    se = pd.Series({s: eff(df, base, last, {s}) for s in states})
    r = (se >= se[st]).sum()
    print(f"  {label:58s} states={len(se)} {st}={se[st]:5.2f} rank={r} p={r/len(se):.3f} | top: {dict(se.sort_values(ascending=False).head(4).round(2))}")
    return se

print("== pair permutation (all 22 states)")
pairtest(k, "base", "last", "main: mixed base, last (CT 23-24)")
pairtest(k, "base19", "last", "base = 2019-20 only for every county")
pairtest(k, "base", "last23", "last = 2023-24 for every county")
pairtest(k, "base19", "last23", "base 2019-20 only + last 2023-24 for all")
print("\n== single-state permutation")
singletest(k, "base", "last", "CA", "CA, main")
singletest(k, "base19", "last", "CA", "CA, base 2019-20 only")
noCA = k[k.State != "CA"]
singletest(noCA, "base", "last", "CT", "CT among 21 (CA dropped), main")
singletest(noCA, "base19", "last", "CT", "CT among 21, base 2019-20")
singletest(noCA, "base", "last23", "CT", "CT among 21, last 2023-24 for all")
singletest(noCA, "base19", "last23", "CT", "CT among 21, base 19-20 + last 23-24")
# how much of CA+CT pair effect is CA?
print("\n== decomposition: CA's pre-period jump as a placebo 'effect'")
pre = k.dropna(subset=["SY2018_19", "SY2019_20"])
pj = pd.Series({s: (g.SY2019_20 - g.SY2018_19).median() for s, g in pre.groupby("State")}).sort_values()
print("  pre-period (18-19 -> 19-20) state median change, i.e. a 'treatment effect' before any treatment:", pj.round(2).to_dict())
# placebo pair test in the pre-period: effect on 19-20 vs 18-19 with 18-19 as base
pre2 = pre.assign(b=pre.SY2018_19, l=pre.SY2019_20)
pairtest(pre2, "b", "l", "PRE-PERIOD placebo (18-19 -> 19-20), 16 states")
