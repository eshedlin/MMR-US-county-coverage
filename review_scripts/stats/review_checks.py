"""Independent statistical review checks. Usage: python3 -I review_checks.py /home/claude/rerun"""
import sys, pathlib, itertools
import numpy as np, pandas as pd

pd.set_option("display.width", 200)
out = pathlib.Path(sys.argv[1])
k0 = pd.read_csv(out / "county_classes.csv")
PRE = ["SY2018_19", "SY2019_20"]; TR = ["SY2020_21", "SY2021_22", "SY2022_23"]
REPEAL = {"CT", "CA"}
rng = np.random.default_rng(1)
k = k0[k0.cls.notna()].copy()
k["nb"] = k[PRE].notna().sum(axis=1); k["nt"] = k[TR].notna().sum(axis=1)
k["chg"] = k["last"] - k["base"]
k["band"] = pd.cut(k.base, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"])
print("classified", len(k), "| baseline years:", k.nb.value_counts().to_dict(),
      "| single-year-baseline states:", sorted(k[k.nb == 1].State.value_counts()[lambda s: s > 5].index))

# ============ 1. reliability / RTM =============
print("\n==== 1. RELIABILITY / RTM")
p = k0.dropna(subset=PRE)
x1, x2 = p.SY2018_19, p.SY2019_20
d = x2 - x1
print(f"pairs n={len(p)}; Pearson r(18-19,19-20)={x1.corr(x2):.3f}; Spearman={x1.corr(x2, method='spearman'):.3f}")
sl = np.polyfit(x1, d, 1)[0]
print(f"slope d~x1 {sl:.3f}; corr(d,x1) {np.corrcoef(x1, d)[0,1]:.3f}; slope d~mean(x1,x2) {np.polyfit((x1+x2)/2, d, 1)[0]:.3f} (should be ~0 if pure noise, var equal)")
print(f"sd x1 {x1.std():.2f} sd x2 {x2.std():.2f}")
# trimmed: drop |d| > 10
m = d.abs() <= 10
print(f"excluding |d|>10 (n dropped {(~m).sum()}): r={x1[m].corr(x2[m]):.3f}, slope {np.polyfit(x1[m], d[m], 1)[0]:.3f}")
for lo, hi in [(0, 90), (90, 95), (95, 101)]:
    mm = (((x1+x2)/2) >= lo) & (((x1+x2)/2) < hi)
    print(f"  band {lo}-{hi}: n={mm.sum()} r={x1[mm].corr(x2[mm]):.3f}")
# lag structure: 3 pre years
p3 = k0.dropna(subset=["SY2017_18"] + PRE)
c = p3[["SY2017_18"] + PRE].corr()
print(f"3 pre-years n={len(p3)}: r(17,18)={c.iloc[0,1]:.3f} r(18,19)={c.iloc[1,2]:.3f} r(17,19 lag2)={c.iloc[0,2]:.3f}")
# long-lag correlation of single years with last year
for col in ["SY2017_18", "SY2018_19", "SY2019_20"]:
    q = k.dropna(subset=[col])
    print(f"  r({col}, last) n={len(q)}: {q[col].corr(q['last']):.3f}")
r1 = x1.corr(x2)
rel2 = 2*r1/(1+r1)
print(f"Spearman-Brown reliability of 2-yr mean = {rel2:.3f}; implied RTM slope for chg~base, 2-yr base: {-(1-rel2):.3f}; 1-yr base: {-(1-r1):.3f}")
# observed post slopes
non = k[~k.State.isin(REPEAL)]
for nb in [1, 2]:
    q = non[non.nb == nb]
    print(f"observed post slope chg~base, non-repeal, nb={nb}: {np.polyfit(q.base, q.chg, 1)[0]:.3f} (n={len(q)})")
# within-state slope (demeaned by state), removes between-state composition
q = non.copy(); q["bd"] = q.base - q.groupby("State").base.transform("mean"); q["cd"] = q.chg - q.groupby("State").chg.transform("mean")
print(f"within-state post slope: {np.polyfit(q.bd, q.cd, 1)[0]:.3f}")

# empirical null: pre-COVID pseudo-period. 2-yr base (17-18,18-19) -> 19-20, and 1-yr base 18-19 -> 19-20
def cls4(b, t, l, thr):
    up = l - b >= thr; dn = l - b <= -thr; rec = ~up & ~dn & (t - b <= -thr)
    return np.select([up, dn, rec], ["up", "down", "dip+rec"], "stable")
print("empirical no-COVID null (shares of |chg|>=1 in a pre-COVID pseudo-window, no trough):")
for name, b, l in [("1yr base 18-19 -> 19-20", x1, x2),
                   ("2yr base 17/18-18/19 -> 19-20", p3[["SY2017_18", "SY2018_19"]].mean(axis=1), p3.SY2019_20)]:
    ch = l - b
    bd = pd.cut(b, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"])
    t = pd.DataFrame({"up>=1": ch >= 1, "down<=-1": ch <= -1, "up>=2": ch >= 2}).groupby(bd, observed=True).mean().mul(100).round(0)
    t["n"] = bd.value_counts()
    print(" ", name, f"overall up>=1 {100*(ch>=1).mean():.0f}% down<=-1 {100*(ch<=-1).mean():.0f}%"); print(t.to_string())
print("observed post-COVID class by band (main):")
print((pd.crosstab(k.band, k.cls, normalize="index")*100).round(0).to_string())

# ============ 2. noise SD =============
print("\n==== 2. NOISE SD")
def rsd(v): return 1.4826*np.median(np.abs(v - np.median(v)))
sd_d, rsd_d = d.std(), rsd(d)
print(f"one-year change d: SD {sd_d:.2f}, robust SD (MAD) {rsd_d:.2f}, median|d| {d.abs().median():.2f} (normal: median|d|=0.674 SD -> SD {d.abs().median()/0.674:.2f}); kurtosis {d.kurt():.1f}")
se = sd_d/np.sqrt(2); rse = rsd_d/np.sqrt(2)
print(f"implied single-year error sigma_e: SD-based {se:.2f}, robust {rse:.2f}")
print(f"null SD(last-base): 2-yr base = sigma_e*sqrt(1.5) = {se*np.sqrt(1.5):.2f} (robust {rse*np.sqrt(1.5):.2f}); 1-yr base = sigma_e*sqrt(2) = {sd_d:.2f} (robust {rsd_d:.2f})")
bandp = pd.cut((x1+x2)/2, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"])
tab = pd.DataFrame({"n": d.groupby(bandp, observed=True).size(), "sd_d": d.groupby(bandp, observed=True).std(),
                    "rsd_d": d.groupby(bandp, observed=True).apply(rsd)})
tab["null_sd_2yr(robust)"] = tab.rsd_d/np.sqrt(2)*np.sqrt(1.5)
tab["null_sd_1yr(robust)"] = tab.rsd_d
tab["thr_1.645sd_2yr"] = 1.645*tab["null_sd_2yr(robust)"]
print(tab.round(2).to_string())
SIG = (tab.rsd_d/np.sqrt(2)).to_dict()  # robust single-year error by band

# class shares under alternative thresholds
def classify(df, thr_series):
    t = df[TR].min(axis=1)
    b, l = df.base, df["last"]
    thr = thr_series
    up = (l - b).round(6) >= thr; dn = (l - b).round(6) <= -thr
    rec = ~up & ~dn & ((t - b).round(6) <= -thr)
    return pd.Series(np.select([up, dn, rec], ["up", "down", "dip+rec"], "stable"), index=df.index)
def shares(c): return (c.value_counts(normalize=True)*100).reindex(["stable", "dip+rec", "down", "up"]).fillna(0).round(1)
sig_e = k.band.map(SIG).astype(float)
null_sd = sig_e*np.sqrt(np.where(k.nb == 2, 1.5, 2.0))
rows = {}
for nm, thr in [("fixed 1", 1.0), ("fixed 2", 2.0), ("fixed 3", 3.0),
                ("band 1.0*nullSD", null_sd), ("band 1.645*nullSD", 1.645*null_sd), ("band 1.96*nullSD", 1.96*null_sd)]:
    thr_s = pd.Series(thr, index=k.index) if np.isscalar(thr) else thr
    rows[nm] = shares(classify(k, thr_s))
print(pd.DataFrame(rows).T.to_string())
print("band-specific null SD used (2yr base):", {b: round(SIG[b]*np.sqrt(1.5), 2) for b in SIG}, "| 1yr base:", {b: round(SIG[b]*np.sqrt(2), 2) for b in SIG})

# pure-null simulation: no true change; observed = T + e, e resampled from d/sqrt2 within band; uses each county's nb and nt
print("pure-noise null simulation (no true change at all), class shares, using county's own #base and #trough years:")
dpool = {b: (d[bandp == b]/np.sqrt(2)).to_numpy() for b in ["<90", "90-95", "95+"]}
def sim_null(thr, reps=200, shift=0.0):
    res = []
    for _ in range(reps):
        cls = []
        for b in ["<90", "90-95", "95+"]:
            sub = k[k.band == b]; n = len(sub); pool = dpool[b]
            e = rng.choice(pool, size=(n, 6)) * rng.choice([-1, 1], size=(n, 6))  # symmetrize
            eb = np.where(sub.nb.to_numpy() == 2, (e[:, 0]+e[:, 1])/2, e[:, 0])
            nt = sub.nt.to_numpy()
            et = np.where(nt == 3, e[:, 2:5].min(1), np.where(nt == 2, e[:, 2:4].min(1), e[:, 2]))
            el = e[:, 5] + shift
            up = el - eb >= thr; dn = el - eb <= -thr; rec = ~up & ~dn & (et + shift*0 - eb <= -thr)
            cls.append(np.select([up, dn, rec], ["up", "down", "dip+rec"], "stable"))
        res.append(shares(pd.Series(np.concatenate(cls))))
    return pd.DataFrame(res).mean().round(1)
print(pd.DataFrame({f"null thr {t}": sim_null(t) for t in [1, 2]}).T.to_string())

# ============ 3. repeal regression =============
print("\n==== 3. REPEAL REGRESSION")
cc = k.copy(); cc["rep"] = cc.State.isin(REPEAL).astype(float); cc["b95"] = cc.base - 95
def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return b, y - X @ b
y = cc.chg.to_numpy()
X = np.column_stack([np.ones(len(cc)), cc.b95, cc.rep]); b, r = ols(X, y)
print(f"linear: repeal {b[2]:.2f}, slope {b[1]:.3f}")
X2 = np.column_stack([X, cc.b95**2, cc.b95**3]); b2, _ = ols(X2, y)
print(f"cubic baseline: repeal {b2[2]:.2f}")
# band dummies + interactions-free spline (knots 90,95,98)
sp = np.column_stack([np.ones(len(cc)), cc.base] + [np.clip(cc.base - kn, 0, None) for kn in [90, 95, 98]] + [cc.rep])
bs, _ = ols(sp, y); print(f"linear spline knots 90/95/98: repeal {bs[-1]:.2f}")
# headroom: change relative to max possible gain; logit scale
eps = 0.5
lg = lambda v: np.log((v.clip(eps, 100-eps))/(100 - v.clip(eps, 100-eps)))
yl = (lg(cc["last"]) - lg(cc.base)).to_numpy()
bl, _ = ols(np.column_stack([np.ones(len(cc)), lg(cc.base), cc.rep]), yl)
print(f"logit-scale change ~ logit base + repeal: {bl[2]:.3f} (log-odds)")
print(f"repeal counties at base>=99: {((cc.rep==1)&(cc.base>=99)).sum()}, any last==100: {(cc['last']>=99.999).sum()}")
# residual nonlinearity by base decile for controls
cc["res"] = r
print("mean residual by base decile (controls):", cc[cc.rep == 0].groupby(pd.qcut(cc[cc.rep == 0].base, 10), observed=True).res.mean().round(2).tolist())
# heteroskedasticity
print("residual SD by band:", cc.groupby("band", observed=True).res.std().round(2).to_dict())
# HC3 SE
H = X @ np.linalg.inv(X.T @ X) @ X.T; h = np.diag(H)
XtXi = np.linalg.inv(X.T @ X)
V3 = XtXi @ (X.T * (r/(1-h))**2) @ X @ XtXi
print(f"HC3 SE repeal (ignores state clustering, NOT valid): {np.sqrt(V3[2,2]):.2f}")
# CR1 as in robustness.py
G = cc.State.unique(); meat = np.zeros((3, 3))
for g in G:
    mm = (cc.State == g).to_numpy(); s = X[mm].T @ r[mm]; meat += np.outer(s, s)
n = len(y); V = XtXi @ meat @ XtXi * (len(G)/(len(G)-1)) * ((n-1)/(n-3))
print(f"CR1 SE repeal (as robustness.py): {np.sqrt(V[2,2]):.2f} -> CI {b[2]-1.96*np.sqrt(V[2,2]):.2f} to {b[2]+1.96*np.sqrt(V[2,2]):.2f}; G={len(G)}, treated clusters=2")
# Cook's distance / influence
s2 = (r**2).sum()/(n-3); cook = r**2/(3*s2) * h/(1-h)**2
cc["cook"] = cook
print("top-10 Cook's D:", cc.nlargest(10, "cook")[["State", "County", "base", "last", "chg", "cook"]].round(2).to_string(index=False))
keep = cook < 4/n
bk, _ = ols(X[keep], y[keep]); print(f"drop Cook>4/n ({(~keep).sum()} counties): repeal {bk[2]:.2f}")
# median (LAD) regression via IRLS
def lad(X, y, it=200):
    w = np.ones(len(y))
    for _ in range(it):
        W = np.sqrt(w); b = np.linalg.lstsq(X*W[:, None], y*W, rcond=None)[0]
        w = 1/np.maximum(np.abs(y - X @ b), 1e-4)
    return b
print(f"LAD (median) regression: repeal {lad(X, y)[2]:.2f}")
# CT jackknife
ctidx = cc.index[cc.State == "CT"]
jk = []
for i in ctidx:
    mm = (cc.index != i); bj, _ = ols(X[mm], y[mm]); jk.append(bj[2])
print(f"drop each CT county: repeal effect range {min(jk):.2f} to {max(jk):.2f}")
for st in ["CT", "CA"]:
    mm = (~cc.State.isin(REPEAL - {st})).to_numpy()
    Xi = np.column_stack([np.ones(mm.sum()), cc.b95[mm], (cc.State[mm] == st).astype(float)])
    print(f"  {st} alone: {ols(Xi, y[mm])[0][2]:.2f}")
# CT gap issue: CT last=2023-24; others' 2023-24 vs 2024-25
o = k[~k.State.isin(REPEAL)].dropna(subset=["SY2023_24"])
print(f"controls: median (2024-25 - 2023-24) = {(o.SY2024_25 - o.SY2023_24).median():.2f}; using 2023-24 as last for everyone:")
alt = k.assign(last2=k.SY2023_24).dropna(subset=["last2"]); alt["chg2"] = alt.last2 - alt.base
Xa = np.column_stack([np.ones(len(alt)), alt.base - 95, alt.State.isin(REPEAL).astype(float)])
print(f"  repeal effect with common last=2023-24: {ols(Xa, alt.chg2.to_numpy())[0][2]:.2f} (n={len(alt)})")

# placebo: effect per state vs all OTHER states (all 22), and its relation to state n
def eff(df, sts):
    Xi = np.column_stack([np.ones(len(df)), df.base - 95, df.State.isin(sts).astype(float)])
    return ols(Xi, df.chg.to_numpy())[0][2]
ctrl = cc[~cc.State.isin(REPEAL)]
pl = pd.DataFrame({"eff": {s: eff(ctrl, {s}) for s in ctrl.State.unique()}, "n": ctrl.State.value_counts()})
print(f"\nplacebo (vs 19 other controls): SD {pl.eff.std():.2f}; corr(|eff|, log n) {np.corrcoef(pl.eff.abs(), np.log(pl.n))[0,1]:.2f}")
print("placebo small states (n<=20):", pl[pl.n <= 20].round(2).to_dict("index"))
# within-state sampling SE of a state effect: sd(res within state)/sqrt(n)
cc["res2"] = cc.chg - cc.groupby("State").chg.transform("mean")
wsd = cc.groupby("State").res2.std()
print("within-state SD of change, CT %.2f CA %.2f, median control %.2f" % (wsd["CT"], wsd["CA"], wsd.drop(list(REPEAL)).median()))
print("sampling SE of a state mean: CT %.2f (n=8), CA %.2f (n=57); between-state SD of placebo effects %.2f" %
      (wsd["CT"]/np.sqrt(8), wsd["CA"]/np.sqrt(57), pl.eff.std()))
# exact single-state rank p
for st in ["CT", "CA"]:
    e = eff(cc[cc.State.isin({st}) | ~cc.State.isin(REPEAL)], {st})
    print(f"  {st} effect {e:.2f}: rank p (one-sided, incl. itself) = {((pl.eff >= e).sum()+1)/(len(pl)+1):.3f}")
# pair permutation over all 22 states: pooled effect of each pair vs remaining 20
states = sorted(cc.State.unique())
pe = {pr: eff(cc, set(pr)) for pr in itertools.combinations(states, 2)}
pe = pd.Series(pe); obs = pe[("CA", "CT")]
print(f"pair permutation: {len(pe)} pairs; observed CA+CT {obs:.2f}; pairs >= obs: {(pe >= obs).sum()} -> p = {(pe >= obs).mean():.4f}")
print("  top pairs:", pe.sort_values(ascending=False).head(5).round(2).to_dict())

# state-level analysis
st = cc.groupby("State").agg(base=("base", "mean"), chg=("chg", "mean"), n=("chg", "size"), rep=("rep", "first"))
st_med = cc.groupby("State").agg(base=("base", "median"), chg=("chg", "median"), rep=("rep", "first"))
for nm, S in [("state means", st), ("state medians", st_med)]:
    Xs = np.column_stack([np.ones(len(S)), S.base - 95, S.rep]); bs_, rs_ = ols(Xs, S.chg.to_numpy())
    # classical SE with 19 df
    s2s = (rs_**2).sum()/(len(S)-3); Vs = s2s*np.linalg.inv(Xs.T @ Xs)
    print(f"{nm} (22 obs): repeal {bs_[2]:.2f}, classical SE {np.sqrt(Vs[2,2]):.2f}, slope {bs_[1]:.3f}")
    # permutation within state-level
    perm = []
    for pr in itertools.combinations(range(len(S)), 2):
        rr = np.zeros(len(S)); rr[list(pr)] = 1
        perm.append(ols(np.column_stack([Xs[:, :2], rr]), S.chg.to_numpy())[0][2])
    perm = np.array(perm); print(f"   state-level pair permutation p = {(perm >= bs_[2]).mean():.4f}")
print(st.round(2).sort_values("chg").to_string())

# ============ 4. trough = min =============
print("\n==== 4. TROUGH MIN BIAS")
print("expected min of 3 iid N(0,1) = -0.846; of 2 = -0.564")
for b in ["<90", "90-95", "95+"]:
    print(f"  band {b}: sigma_e (robust) {SIG[b]:.2f} -> E[min of 3 noise] = {-0.846*SIG[b]:.2f} points")
sim_t = []
for _ in range(2000):
    e = rng.normal(size=(1, 5))
print("P(trough-base <= -1) under pure noise, by band (2-yr base, 3 trough yrs, normal):")
for b in ["<90", "90-95", "95+"]:
    e = rng.normal(0, SIG[b], size=(200000, 5))
    tb = e[:, 2:5].min(1) - e[:, :2].mean(1)
    print(f"  {b}: thr1 {np.mean(tb <= -1)*100:.0f}%  thr2 {np.mean(tb <= -2)*100:.0f}%; mean(trough-base) {tb.mean():.2f}")
obs_dip = (k.trough - k.base).round(6)
print("observed share with trough-base<=-1 by band:", (obs_dip <= -1).groupby(k.band, observed=True).mean().mul(100).round(0).to_dict())
print("by #trough years, share 'dip+rec':", (k.cls == "ירידה והתאוששות").groupby(k.nt).mean().mul(100).round(1).to_dict(), k.nt.value_counts().to_dict())

# ============ 5. bootstrap =============
print("\n==== 5. STATE-CLUSTER BOOTSTRAP")
lab = {"ירידה שלא התאוששה": "down", "עלייה": "up", "ירידה והתאוששות": "dip+rec", "יציב": "stable"}
k["c"] = k.cls.map(lab)
gs = {s: g.c.to_numpy() for s, g in k.groupby("State")}
S = list(gs); B = 5000; bt = []; btw = []
for _ in range(B):
    draw = rng.choice(S, len(S))
    allc = np.concatenate([gs[s] for s in draw])
    bt.append([(allc == c).mean()*100 for c in ["down", "up", "dip+rec", "stable"]])
    btw.append(np.mean([(gs[s] == "down").mean()*100 for s in draw]))
bt = np.array(bt)
for i, c in enumerate(["down", "up", "dip+rec", "stable"]):
    print(f"  {c}: point {(k.c == c).mean()*100:.1f}, 95% percentile CI {np.percentile(bt[:, i], 2.5):.1f} to {np.percentile(bt[:, i], 97.5):.1f}")
print(f"  state-weighted mean 'down' share: {np.mean([(gs[s]=='down').mean()*100 for s in S]):.1f}, CI {np.percentile(btw, 2.5):.1f} to {np.percentile(btw, 97.5):.1f}")
print(f"  TX share of counties {len(gs['TX'])/len(k)*100:.1f}%")
# thr2 version
k["c2"] = classify(k, pd.Series(2.0, index=k.index))
gs2 = {s: g.c2.to_numpy() for s, g in k.groupby("State")}
bt2 = [np.mean(np.concatenate([gs2[s] for s in rng.choice(S, len(S))]) == "down")*100 for _ in range(B)]
print(f"  thr2 'down': point {(k.c2=='down').mean()*100:.1f}, CI {np.percentile(bt2, 2.5):.1f} to {np.percentile(bt2, 97.5):.1f}")
# continuous alternative
bm = [np.median(np.concatenate([k.chg[k.State == s].to_numpy() for s in rng.choice(S, len(S))])) for _ in range(2000)]
print(f"  median change: {k.chg.median():.2f}, cluster-bootstrap CI {np.percentile(bm, 2.5):.2f} to {np.percentile(bm, 97.5):.2f}")
print(f"  share chg<0: {(k.chg<0).mean()*100:.1f}")

# ============ 6. ties =============
print("\n==== 6. FLOATING-POINT TIES")
for thr in [1, 1.5, 2]:
    for nm, v in [("last-base", k["last"] - k.base), ("trough-base", k.trough - k.base)]:
        raw_ge = (v >= thr) | (v <= -thr); rnd_ge = (v.round(6) >= thr) | (v.round(6) <= -thr)
        near = (v.abs() - thr).abs() < 1e-6
        print(f"  thr {thr} {nm}: |v| within 1e-6 of thr: {near.sum()}, classification flips with round(6): {(raw_ge != rnd_ge).sum()}")
vv = pd.concat([k["last"] - k.base, k.trough - k.base])
fr = (vv*1e6 - np.round(vv*1e6)).abs()
print(f"  max |difference*1e6 - round| (sub-1e-6 content): {fr.max():.3g}; values with genuine digits beyond 6 decimals: {(fr > 1e-3).sum()} of {len(vv)}")
gap = (vv.abs() - 1).abs(); gap2 = (vv.abs() - 2).abs(); gap15 = (vv.abs() - 1.5).abs()
g = pd.concat([gap, gap15, gap2]); print(f"  smallest nonzero distance to any threshold (after round 6): {g.round(6)[g.round(6) > 0].min():.2e}")
