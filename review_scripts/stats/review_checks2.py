"""Supplementary checks: ties in-memory, shifted null simulation, Conley-Taber-style interval, nb-specific slope.
Usage: python3 -I review_checks2.py /home/claude/data /home/claude/rerun"""
import sys, pathlib, itertools
import numpy as np, pandas as pd
data, out = map(pathlib.Path, sys.argv[1:3])
rng = np.random.default_rng(7)
PRE = ["SY2018_19", "SY2019_20"]; TR = ["SY2020_21", "SY2021_22", "SY2022_23"]; REPEAL = {"CT", "CA"}

# ---- ties, in memory exactly as analysis.py
d = pd.read_csv(data / "mmr_data_us_counties_v2.csv", encoding="utf-8-sig", dtype=str)
YRS = [c for c in d.columns if c.startswith("SY")]
raw = d[YRS]; star = raw.apply(lambda c: c.str.contains(r"\*", na=False))
val = (raw.apply(lambda c: pd.to_numeric(c.str.replace("*", "", regex=False), errors="coerce")) * 100).mask(star)
kc = pd.read_csv(out / "county_classes.csv"); kc = kc[kc.cls.notna()]
panel = pd.concat([d[["State", "County"]], val], axis=1)
panel = panel.merge(kc[["State", "County"]], on=["State", "County"])
b = panel[PRE].mean(axis=1); t = panel[TR].min(axis=1)
l = pd.Series(np.where(panel.State == "CT", panel.SY2023_24, panel.SY2024_25), index=panel.index)
print("n", len(panel))
for thr in [1, 1.5, 2]:
    for nm, v in [("last-base", l - b), ("trough-base", t - b)]:
        a = (v >= thr) | (v <= -thr); r = (v.round(6) >= thr) | (v.round(6) <= -thr)
        print(f"thr {thr} {nm}: exact-tie cases {int(((v.round(6).abs()) == thr).sum())}, flips raw vs round6 {int((a != r).sum())}")
        ex = v[(a != r)]
        if len(ex): print("   e.g.", [repr(x) for x in ex.head(3)])
# integer-arithmetic check: raw strings -> exact decimals
from decimal import Decimal
def dec(s):
    return None if (pd.isna(s) or "*" in s) else Decimal(s) * 100
flips = 0
for i, row in d.merge(kc[["State", "County"]], on=["State", "County"]).iterrows():
    pre = [dec(row[c]) for c in PRE]; pre = [x for x in pre if x is not None]
    lv = dec(row["SY2023_24"] if row.State == "CT" else row["SY2024_25"])
    if not pre or lv is None: continue
    bb = sum(pre) / len(pre); ch = lv - bb
    fl = float(ch)
    if (round(fl, 6) >= 2) != (ch >= 2) or (round(fl, 6) <= -2) != (ch <= -2): flips += 1
print("round(6) vs exact Decimal disagreements at thr 2 (last-base):", flips)

# ---- shifted null: no county-specific true change; common shift delta; noise from pre-COVID pool
k = kc.copy(); k["nb"] = k[PRE].notna().sum(axis=1); k["nt"] = k[TR].notna().sum(axis=1)
p = pd.read_csv(out / "county_classes.csv").dropna(subset=PRE)
dd = (p.SY2019_20 - p.SY2018_19); bandp = pd.cut(p[PRE].mean(axis=1), [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"])
pool = {bn: (dd[bandp == bn] / np.sqrt(2)).to_numpy() for bn in ["<90", "90-95", "95+"]}
r1 = p.SY2018_19.corr(p.SY2019_20)
def sim(delta, reps=100, trim=False):
    out_ = []
    for _ in range(reps):
        rel = np.where(k.nb == 2, 2*r1/(1+r1), r1)
        m = k.groupby("State").base.transform("mean").to_numpy()
        T = m + np.sqrt(rel) * (k.base.to_numpy() - m)
        bandT = pd.cut(T, [-1e9, 90, 95, 1e9], right=False, labels=["<90", "90-95", "95+"]).astype(str)
        e = np.zeros((len(k), 6))
        for bn in pool:
            mm = bandT == bn
            e[mm] = rng.choice(pool[bn], size=(mm.sum(), 6)) * rng.choice([-1, 1], size=(mm.sum(), 6))
        bs = T + np.where(k.nb == 2, e[:, :2].mean(1), e[:, 0])
        ls = np.minimum(T + delta + e[:, 5], 100); bs = np.minimum(bs, 100)
        ts = np.minimum(T + delta + np.where(k.nt == 3, e[:, 2:5].min(1), np.where(k.nt == 2, e[:, 2:4].min(1), e[:, 2])), 100)
        ch = ls - bs
        up = ch >= 1; dn = ch <= -1; rec = ~up & ~dn & (ts - bs <= -1)
        out_.append([dn.mean()*100, up.mean()*100, rec.mean()*100, (~up & ~dn & ~rec).mean()*100,
                     (bs[up] < 95).mean()*100, np.median(ch), up[bs < 90].mean()*100, up[bs >= 95].mean()*100])
    return pd.Series(np.mean(out_, 0), index=["down", "up", "dip+rec", "stable", "%up below95", "median chg", "up|<90", "up|95+"]).round(1)
print("\nshifted-null simulation (every county shifts by delta, plus noise+RTM):")
print(pd.DataFrame({f"delta {dl}": sim(dl) for dl in [0, -2, -2.9, -3.5]}).T.to_string())
ch = k["last"] - k.base
print("observed: down %.1f up %.1f dip+rec %.1f stable %.1f, %%up below95 %.1f, median %.2f, up|<90 %.1f, up|95+ %.1f" % (
    (ch.round(6) <= -1).mean()*100, (ch.round(6) >= 1).mean()*100, (k.cls == "ירידה והתאוששות").mean()*100, (k.cls == "יציב").mean()*100,
    (k.base[ch.round(6) >= 1] < 95).mean()*100, ch.median(), (ch.round(6) >= 1)[k.base < 90].mean()*100, (ch.round(6) >= 1)[k.base >= 95].mean()*100))

# ---- Conley-Taber style interval from placebo pair effects among controls
k["chg"] = ch; k["rep"] = k.State.isin(REPEAL).astype(float)
def eff(df, sts):
    X = np.column_stack([np.ones(len(df)), df.base - 95, df.State.isin(sts).astype(float)])
    return np.linalg.lstsq(X, df.chg.to_numpy(), rcond=None)[0][2]
ctrl = k[k.rep == 0]
pp = np.array([eff(ctrl, set(pr)) for pr in itertools.combinations(sorted(ctrl.State.unique()), 2)])
obs = eff(k, REPEAL)
print(f"\nplacebo pair effects among 20 controls (190 pairs): SD {pp.std():.2f}, 2.5/97.5% {np.percentile(pp,2.5):.2f}/{np.percentile(pp,97.5):.2f}")
print(f"Conley-Taber-style 95% interval for repeal effect: {obs-np.percentile(pp,97.5):.2f} to {obs-np.percentile(pp,2.5):.2f} (point {obs:.2f})")
# nb-specific baseline slopes
X = np.column_stack([np.ones(len(k)), k.base - 95, (k.base - 95)*(k.nb == 1), (k.nb == 1).astype(float), k.rep])
print("repeal with nb-specific slope/intercept:", round(np.linalg.lstsq(X, k.chg.to_numpy(), rcond=None)[0][-1], 2))
# two-step: within-state slope, then state means of adjusted change
q = k.copy(); q["bd"] = q.base - q.groupby("State").base.transform("mean"); q["cd"] = q.chg - q.groupby("State").chg.transform("mean")
bw = np.polyfit(q.bd, q.cd, 1)[0]
q["adj"] = q.chg - bw * (q.base - 95)
sm = q.groupby("State").agg(adj=("adj", "mean"), rep=("rep", "first"))
print(f"within-state slope {bw:.3f}; state-mean adjusted change: repeal {sm[sm.rep==1].adj.mean():.2f} vs others {sm[sm.rep==0].adj.mean():.2f}; diff {sm[sm.rep==1].adj.mean()-sm[sm.rep==0].adj.mean():.2f}")
print("rank of CA, CT among 22 states on adjusted change:", sm.adj.rank(ascending=False)[["CA", "CT"]].to_dict())
