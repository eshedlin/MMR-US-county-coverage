"""Devil's advocate item 3: error rates of fixed vs band-specific thresholds under a model WITH real heterogeneity.
Endpoint classification only (last - base): 'up' / 'down' / 'not classified as changed'.
Usage: python3 -I d3_error_rates.py /home/claude/rerun/county_classes.csv"""
import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
rng = np.random.default_rng(2026)
k = pd.read_csv(sys.argv[1]); k = k[k.cls.notna()].copy()
PRE = ["SY2018_19", "SY2019_20"]
k["nb"] = k[PRE].notna().sum(axis=1).to_numpy()
k["chg"] = (k["last"] - k.base).round(6)
k["band"] = pd.cut(k.base, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"]).astype(str)
B = k.band.to_numpy(); NB = k.nb.to_numpy()

# empirical heavy-tailed noise pools: pre-period one-year change, within-state, /sqrt2, by band (as round 1 but within-state)
p = k.dropna(subset=PRE).copy(); p["d"] = p.SY2019_20 - p.SY2018_19
p["d"] = p.d - p.groupby("State").d.transform("median")
pb = pd.cut(p[PRE].mean(axis=1), [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"]).astype(str)
POOL = {b: (p.d[pb == b] / np.sqrt(2)).to_numpy() for b in ["<90", "90-95", "95+"]}
rsd = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
SIG = {b: rsd(POOL[b]) for b in POOL}  # robust sigma_e (round-1 style)
print("robust sigma_e by band (within-state pool):", {b: round(v, 2) for b, v in SIG.items()})
nullsd = np.array([SIG[b] for b in B]) * np.sqrt(np.where(NB == 2, 1.5, 2.0))

def draw_noise(post_scale):
    eb = np.zeros(len(k)); el = np.zeros(len(k))
    for b in POOL:
        m = B == b; n = m.sum()
        e = rng.choice(POOL[b], (n, 3)) * rng.choice([-1, 1], (n, 3))
        eb[m] = np.where(NB[m] == 2, e[:, :2].mean(1), e[:, 0]); el[m] = e[:, 2] * post_scale
    return el - eb

state_med = k.groupby("State").chg.transform("median").to_numpy()
SU = {"<90": 4.2, "90-95": 2.7, "95+": 1.8}  # county-specific real SD from noise-free covariance test (d2_hetero.py B)
def true_change(scen):
    if scen == "R1 world: common shift -2.9, no heterogeneity":
        return np.full(len(k), -2.9)
    if scen == "states differ, no within-state":
        return state_med.copy()
    if scen == "states + within-state (SD from cov test)":
        return state_med + rng.normal(0, 1, len(k)) * np.array([SU[b] for b in B])
    if scen == "states + within-state SD 3.5 all bands":
        return state_med + rng.normal(0, 3.5, len(k))

RULES = {"fixed 1": np.full(len(k), 1.0), "fixed 2": np.full(len(k), 2.0), "fixed 3": np.full(len(k), 3.0),
         "band 1.0*nullSD": 1.0 * nullsd, "band 1.645*nullSD": 1.645 * nullsd}
print("thresholds of band 1.645*nullSD (nb=2 / nb=1):",
      {b: (round(1.645 * SIG[b] * np.sqrt(1.5), 2), round(1.645 * SIG[b] * np.sqrt(2), 2)) for b in SIG})
obs = {r: {"obs %down": np.mean(k.chg <= -t) * 100, "obs %up": np.mean(k.chg >= t) * 100,
           "obs %unchanged": np.mean((k.chg > -t) & (k.chg < t)) * 100} for r, t in RULES.items()}
print("\nobserved endpoint shares by rule:"); print(pd.DataFrame(obs).T.round(1).to_string())

rows = []
for scen in ["R1 world: common shift -2.9, no heterogeneity", "states differ, no within-state",
             "states + within-state (SD from cov test)", "states + within-state SD 3.5 all bands"]:
    for post_scale in [1.0, 1.5]:
        acc = {r: [] for r in RULES}
        for _ in range(300):
            D = true_change(scen); Y = D + draw_noise(post_scale)
            for r, t in RULES.items():
                dn, up = Y <= -t, Y >= t
                real = np.abs(D) >= 2; null = np.abs(D) < 1
                acc[r].append([
                    dn.mean() * 100, (D <= -2).mean() * 100,                      # classified down vs truly down >=2
                    (dn | up)[null].mean() * 100 if null.any() else np.nan,       # false change among |D|<1
                    (~dn & ~up)[real].mean() * 100,                               # missed: |D|>=2 called unchanged
                    ((up & (D <= -2)) | (dn & (D >= 2))).mean() * 100,           # sign errors among all
                    (dn & (D > -1)).sum() / max(dn.sum(), 1) * 100,               # of those called down, % truly not down >=1
                    (~dn & ~up).mean() * 100])
        for r in RULES:
            a = np.nanmean(acc[r], 0)
            rows.append(dict(scenario=scen, post_noise_x=post_scale, rule=r, called_down=a[0], true_down2=a[1],
                             false_change_if_D_lt1=a[2], missed_if_absD_ge2=a[3], sign_err=a[4],
                             FDR_down=a[5], called_unchanged=a[6]))
res = pd.DataFrame(rows).round(1)
print("\nsimulated error rates (% ; 300 reps each):")
print(res.to_string(index=False))
