"""County kindergarten MMR coverage before and after COVID-19 (v2, after two review rounds).

Data: Johns Hopkins CSSE, https://github.com/CSSEGISandData/MMR_data
Citation: Dong, E., Saiyed, S., Nearchou, A., Okura, Y., & Gardner, L. M. (2025).
  Trends in county-level MMR vaccination coverage in children in the United States.
  JAMA. https://doi.org/10.1001/jama.2025.8952

Usage: python get_data.py data && python analysis.py data out

Decisions (see decisions.md): D1 continuous change is the primary outcome;
D2 baseline = 2019-20 for all states; D3 CA 2017-18 and 2018-19 removed
(different measure), CO kept with a 3-year-mean sensitivity; D4 Connecticut is
a descriptive case, no causal estimate and no confidence interval from 2 clusters.
"""
import sys, pathlib
import numpy as np
import pandas as pd

data_dir, out_dir = map(pathlib.Path, sys.argv[1:3])
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261127)
B = 5000
R = lambda x: np.round(x, 6)  # threshold ties: raw values have <=7 decimals in percent

# ---------------------------------------------------------------- 1. load and clean
d = pd.read_csv(data_dir / "mmr_data_us_counties_v2.csv", encoding="utf-8-sig", dtype=str)
s = pd.read_csv(data_dir / "mmr_data_sources_v2.csv")
for c in ["Spatial Unit", "Age Group", "Vaccination Type Provided"]:
    s[c] = s[c].astype(str).str.strip()
YRS = [c for c in d.columns if c.startswith("SY")]

measure_ok = s["Vaccination Type Provided"].str.lower().isin(["2 mmr", "2+ mmr", "up to date (2 mmr)", "2 measles"])
comp = s[measure_ok & (s["Age Group"] == "Kindergarten") & s["Spatial Unit"].str.startswith("County")]
COMP = sorted(comp.Abbreviation)

raw = d[YRS]
star = raw.apply(lambda c: c.str.contains(r"\*", na=False))          # non-comparable method
val = raw.apply(lambda c: pd.to_numeric(c.str.replace("*", "", regex=False), errors="coerce")) * 100
val = val.mask(star)
d["FIPS5"] = d.FIPS.str.zfill(5)
p = pd.concat([d[["State", "FIPS5", "County"]], val], axis=1)
p = p[~((p.State == "CT") & p[YRS[:-1]].isna().all(axis=1))]          # CT planning regions (2024-25 only)
p.loc[p.State == "CA", ["SY2017_18", "SY2018_19"]] = np.nan            # D3: not 2+ MMR in CA
p = p[p.State.isin(COMP)].reset_index(drop=True)

BASE = "SY2019_20"
p["last"] = np.where(p.State == "CT", p.SY2023_24, p.SY2024_25)
p["chg"] = R(p["last"] - p[BASE])
a = p.dropna(subset=["chg"]).copy()                                   # analysed counties

flow = {
    "comparable states": len(COMP),
    "counties in comparable states": len(p),
    "counties with baseline and last value": len(a),
    "states with analysed counties": a.State.nunique(),
    "states with no analysed county": sorted(set(COMP) - set(a.State)),
    "starred values removed (whole file)": int(star.values.sum()),
}

# ---------------------------------------------------------------- 2. primary: continuous change
def headline(df):
    return pd.Series({
        "median change": df.chg.median(),
        "% below baseline": (df.chg < 0).mean() * 100,
        "% down >=1": (df.chg <= -1).mean() * 100,
        "% down >=2": (df.chg <= -2).mean() * 100,
        "median of state medians": df.groupby("State").chg.median().median(),
    })

def cluster_boot(df, f, b=B):
    groups = {st: g for st, g in df.groupby("State")}
    names = list(groups)
    out = []
    for _ in range(b):
        pick = rng.choice(names, size=len(names), replace=True)
        out.append(f(pd.concat([groups[n] for n in pick], ignore_index=True)))
    return pd.DataFrame(out)

H = headline(a)
boot = cluster_boot(a, headline)
prim = pd.DataFrame({"estimate": H, "ci_low": boot.quantile(.025), "ci_high": boot.quantile(.975)}).round(2)
prim.to_csv(out_dir / "primary.csv", encoding="utf-8-sig")

# year path relative to 2019-20 (no CT in 2024-25)
path = pd.Series({y: R(a[y] - a[BASE]).median() for y in YRS[3:]}).round(2)
path.to_csv(out_dir / "year_path.csv", header=["median change vs 2019-20"], encoding="utf-8-sig")

# by state
st = a.groupby("State").chg.agg(n="size", median="median",
                                pct_below=lambda x: (x < 0).mean() * 100).round(2).sort_values("median")
st.to_csv(out_dir / "by_state.csv", encoding="utf-8-sig")

# heterogeneity: between states, and within states from two changes sharing no observation
het = {"SD of state medians": st["median"].std(),
       "state median range": (st["median"].min(), st["median"].max())}
w = p.dropna(subset=["SY2018_19", "SY2019_20", "SY2022_23", "SY2024_25"]).copy()
w = w[w.State != "CT"]
w["A"] = w.SY2022_23 - w.SY2018_19
w["Bc"] = w.SY2024_25 - w.SY2019_20
def within_corr(df):
    dm = df[["A", "Bc"]] - df.groupby("State")[["A", "Bc"]].transform("mean")
    return dm.A.corr(dm.Bc, method="spearman")
wc = within_corr(w)
wcb = cluster_boot(w, lambda df: pd.Series({"r": within_corr(df)}), b=2000)
het["within-state corr of independent changes"] = (round(wc, 2), round(wcb.r.quantile(.025), 2),
                                                   round(wcb.r.quantile(.975), 2), len(w), w.State.nunique())

# ---------------------------------------------------------------- 3. sensitivity and influence
sens = {"primary": H}
co3 = a.copy()
m = co3.State == "CO"
co3.loc[m, "chg"] = R(co3.loc[m, "last"] - p.loc[co3.index[m], ["SY2017_18", "SY2018_19", BASE]].mean(axis=1))
sens["CO baseline = mean 2017-18 to 2019-20"] = headline(co3.dropna(subset=["chg"]))
sens["drop CO"] = headline(a[a.State != "CO"])
sens["drop VT, WI (capped at 95)"] = headline(a[~a.State.isin(["VT", "WI"])])
sens["drop TN (baseline weighted by dataset authors), OK (optional survey)"] = headline(a[~a.State.isin(["TN", "OK"])])
sens["drop Wasco, OR (shares series with Sherman)"] = headline(a[~((a.State == "OR") & (a.County == "Wasco"))])
sens["last = 2023-24 for all states"] = headline(a.assign(chg=R(a.SY2023_24 - a[BASE])).dropna(subset=["chg"]))
sens["state-weighted (mean of state values)"] = pd.DataFrame(
    {x: headline(g) for x, g in a.groupby("State")}).T.mean()
sens = pd.DataFrame(sens).T.round(2)
sens.to_csv(out_dir / "sensitivity.csv", encoding="utf-8-sig")

loo = pd.DataFrame({x: headline(a[a.State != x]) for x in sorted(a.State.unique())}).T
loo_rng = pd.DataFrame({"min": loo.min(), "state at min": loo.idxmin(),
                        "max": loo.max(), "state at max": loo.idxmax()})
loo.round(2).to_csv(out_dir / "loo_states.csv", encoding="utf-8-sig")

# ---------------------------------------------------------------- 4. secondary: descriptive classes
TROUGH = ["SY2020_21", "SY2021_22", "SY2022_23"]
ORDER = ["יציב", "ירידה והתאוששות", "ירידה שלא התאוששה", "עלייה"]
def classes(df, thr):
    t = R(df[TROUGH].min(axis=1) - df[BASE])
    ok = df.chg.notna() & t.notna()
    c = pd.Series(pd.NA, index=df.index, dtype="object")
    up, dn = ok & (df.chg >= thr), ok & (df.chg <= -thr)
    rec = ok & ~up & ~dn & (t <= -thr)
    c[up], c[dn], c[rec] = "עלייה", "ירידה שלא התאוששה", "ירידה והתאוששות"
    c[ok & ~up & ~dn & ~rec] = "יציב"
    return c
cls = pd.DataFrame({f"סף {t}": (classes(a, t).value_counts(normalize=True) * 100).reindex(ORDER)
                    for t in [1, 1.5, 2, 3]}).round(1)
cls.loc["n"] = [int(classes(a, t).notna().sum()) for t in [1, 1.5, 2, 3]]
cls.to_csv(out_dir / "classes_secondary.csv", encoding="utf-8-sig")
a.assign(cls_thr1=classes(a, 1), cls_thr2=classes(a, 2)).to_csv(out_dir / "county_change.csv",
                                                               index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 5. Connecticut, descriptive (D4)
ct = a[a.State == "CT"]
others = a[a.State != "CT"]
# fair window for CT comparison: everyone ends 2023-24
f = a.assign(chg=R(a.SY2023_24 - a[BASE])).dropna(subset=["chg"])
def adj(df, s_):
    X = np.column_stack([np.ones(len(df)), df[BASE] - 95, df.State.eq(s_).astype(float)])
    return np.linalg.lstsq(X, df.chg.to_numpy(), rcond=None)[0][2]
noCA = f[f.State != "CA"]
rank = pd.Series({x: adj(noCA, x) for x in noCA.State.unique()}).sort_values(ascending=False)
ct_out = {
    "CT counties": len(ct),
    "CT median change (to 2023-24)": round(ct.chg.median(), 2),
    "CT counties below baseline": int((ct.chg < 0).sum()),
    "others median change, same window (to 2023-24)": round(f[f.State != "CT"].chg.median(), 2),
    "CT baseline-adjusted contrast, window to 2023-24, CA excluded": round(rank["CT"], 2),
    "CT rank among states (CA excluded)": f"{list(rank.index).index('CT') + 1} of {len(rank)}",
    "minimum attainable one-sided rank p": round(1 / len(rank), 3),
    "next states": rank.drop("CT").head(3).round(2).to_dict(),
    "leave-one-CT-county-out contrast range": (
        round(min(adj(noCA.drop(i), "CT") for i in noCA.index[noCA.State == "CT"]), 2),
        round(max(adj(noCA.drop(i), "CT") for i in noCA.index[noCA.State == "CT"]), 2)),
    "CT year path": {y: round(R(ct[y] - ct[BASE]).median(), 2) for y in YRS[3:7]},
}
pd.Series(rank.round(2)).to_csv(out_dir / "ct_rank.csv", header=["baseline-adjusted contrast"], encoding="utf-8-sig")

# Maine (secondary; full-series measure, outside comparable set)
me = pd.concat([d[["State", "County"]], val], axis=1)
me = me[me.State == "ME"].copy()
me["chg"] = R(me.SY2024_25 - me[BASE])
me_out = {"ME counties": int(me.chg.notna().sum()), "ME median change": round(me.chg.median(), 2),
          "ME counties above baseline": int((me.chg > 0).sum())}

# ---------------------------------------------------------------- 6. print
pd.set_option("display.width", 200)
for k_, v in flow.items(): print(f"{k_}: {v}")
print("\n== primary (state-clustered bootstrap 95% CI)\n", prim)
print("\n== year path (median change vs 2019-20)\n", path.to_string())
print("\n== heterogeneity\n", het)
print("\n== by state\n", st.to_string())
print("\n== sensitivity\n", sens.to_string())
print("\n== leave-one-state-out range\n", loo_rng.round(2).to_string())
print("\n== secondary classes (descriptive, threshold-dependent)\n", cls.to_string())
print("\n== Connecticut (descriptive)"); [print(f"  {k_}: {v}") for k_, v in ct_out.items()]
print("\n== Maine (secondary)", me_out)
