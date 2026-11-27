import sys, numpy as np, pandas as pd
k = pd.read_csv(sys.argv[1] + "/county_classes.csv"); k=k[k.cls.notna()].copy()
Y=[c for c in k.columns if c.startswith("SY")]
both=k.dropna(subset=["SY2018_19","SY2019_20"])
print("== state medians by year (counties with both pre years)")
print(both.groupby("State")[Y].median().round(1).to_string())
print("== median 1-yr change 18-19 -> 19-20 by state:", (both.SY2019_20-both.SY2018_19).groupby(both.State).median().round(1).sort_values().to_dict())
def eff(df,base,st):
    y=(df["last"]-base).values; X=np.column_stack([np.ones(len(df)),base-95,df.State.eq(st).astype(float)])
    ok=~np.isnan(y)&~np.isnan(base.values); return np.linalg.lstsq(X[ok],y[ok],rcond=None)[0][2]
non=k[~k.State.isin(["CA","CT"])]
for bname in ["base","SY2019_20"]:
    out={}
    for st in ["CA","CT"]:
        z=pd.concat([non,k[k.State==st]]); out[st]=round(eff(z,z[bname],st),2)
    pl={st:eff(non,non[bname],st) for st in non.State.unique()}
    pl=pd.Series(pl).dropna().sort_values()
    print(bname,"effects",out,"placebo max %.2f (%s), n placebo>=CA: %d, >=CT: %d"%(pl.max(),pl.idxmax(),(pl>=out["CA"]).sum(),(pl>=out["CT"]).sum()))
# CA counties: last vs 2019-20 only
ca=k[k.State=="CA"]; print("CA median last-2019-20: %.2f; others: %.2f"%((ca["last"]-ca.SY2019_20).median(),(non["last"]-non.SY2019_20).median()))
ct=k[k.State=="CT"]; print("CT median last-2019-20: %.2f"%((ct["last"]-ct.SY2019_20).median()))
# CA class under 2019-20-only baseline
b=ca.SY2019_20; c=np.select([ca["last"]-b>=1, ca["last"]-b<=-1],["up","down"],"mid"); print("CA classes with 2019-20 baseline:",pd.Series(c).value_counts().to_dict(), "| main:",ca.cls.value_counts().to_dict())
# Overall class shares with baseline = 2019-20 only (common single-year baseline for all states)
b=k.SY2019_20; d=k["last"]-b; t=k[["SY2020_21","SY2021_22","SY2022_23"]].min(axis=1)
for thr in [1,2]:
    ok=b.notna(); up=d>=thr; dn=d<=-thr; rec=~up&~dn&(t-b<=-thr)
    print("baseline=2019-20, thr",thr,"n",ok.sum(),"down %.1f up %.1f rec %.1f stable %.1f"%tuple(x[ok].mean()*100 for x in [dn,up,rec,~up&~dn&~rec]))
