import sys, numpy as np, pandas as pd
k = pd.read_csv(sys.argv[1] + "/county_classes.csv"); k=k[k.cls.notna()].copy()
p=k.dropna(subset=["SY2018_19","SY2019_20"]); d=p.SY2019_20-p.SY2018_19
for lab,m in [("all",p.State.notna()),("excl CA,CO",~p.State.isin(["CA","CO"]))]:
    x=d[m]; print(lab,"n",m.sum(),"median|d| %.2f  |d|>=2 %.0f%%  SD %.2f  MAD-based SD %.2f"%(x.abs().median(),(x.abs()>=2).mean()*100,x.std(),1.4826*(x-x.median()).abs().median()))
    band=pd.cut(p[m][["SY2018_19","SY2019_20"]].mean(axis=1),[0,90,95,101],right=False)
    print("   robust SD by band:",x.groupby(band,observed=True).apply(lambda v:round(1.4826*(v-v.median()).abs().median(),2)).to_dict())
# implied SD of (last - 2yr mean base) under pure noise if single-year noise sd s: sqrt(s^2/2 + s^2)?  s from d: s=sd(d)/sqrt2
s=1.4826*(d[~p.State.isin(["CA","CO"])]-d.median()).abs().median()/np.sqrt(2)
print("per-year noise SD ~ %.2f; SD of last - 2yr mean ~ %.2f; of last - 1yr base ~ %.2f"%(s,s*np.sqrt(1.5),s*np.sqrt(2)))
# Floor/ceiling: share with base>=98
print("base>=98: %.1f%%; max possible up for those <2"%((k.base>=98).mean()*100))
print("classes by state baseline-years type:", pd.crosstab(k[["SY2018_19","SY2019_20"]].notna().sum(axis=1),k.cls,normalize="index").round(3).mul(100).to_string())
