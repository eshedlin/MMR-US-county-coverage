import sys, numpy as np, pandas as pd
k = pd.read_csv(sys.argv[1] + "/county_classes.csv")
PRE=["SY2018_19","SY2019_20"]; TR=["SY2020_21","SY2021_22","SY2022_23"]
k=k[k.cls.notna()].copy()
k["npre"]=k[PRE].notna().sum(axis=1); k["ntr"]=k[TR].notna().sum(axis=1)
k["chg"]=k["last"]-k["base"]
print("n",len(k))
print("== baseline years used, by state"); print(k.groupby("State").npre.agg(["min","max","size"]).T.to_string())
print("share classified counties with single-year baseline: %.1f%%"%((k.npre==1).mean()*100))
print("trough years count:", k.ntr.value_counts().to_dict(), k[k.ntr<3].State.value_counts().to_dict())
# capped states
for st in ["VT","WI"]:
    x=k[k.State==st]; v=x[PRE+TR+["SY2023_24","SY2024_25"]]
    print(st,"cells at exactly 95:",int((v.round(4)==95).sum().sum()),"of",int(v.notna().sum().sum()), "class:",x.cls.value_counts().to_dict())
# state-weighted shares
DN="ירידה שלא התאוששה"; UP="עלייה"
st=pd.crosstab(k.State,k.cls,normalize="index")*100
print("== county-weighted vs state-weighted share down-not-recovered: %.1f vs mean of states %.1f, median %.1f"%((k.cls==DN).mean()*100, st[DN].mean(), st[DN].median()))
print("excluding CA,CT: county %.1f, state-mean %.1f"%((k[~k.State.isin(["CA","CT"])].cls==DN).mean()*100, st.drop(["CA","CT"])[DN].mean()))
print("TX share of sample %.1f%%"%((k.State=="TX").mean()*100))
# down-not-recovered without any COVID dip
dn=k[k.cls==DN]
print("== down-not-recovered with no trough dip >=1: %d of %d"%((dn.trough-dn.base>-1).sum(),len(dn)))
print("   ... whose trough >= base (never below baseline in 2020-23): %d"%((dn.trough>=dn.base).sum()))
# null benchmark: pre-COVID one-year change distribution vs post change
p=k.dropna(subset=PRE); d1=p.SY2019_20-p.SY2018_19
for thr in [1,2,3]:
    print("thr %d: pre 1-yr change <=-thr %.0f%%, >=+thr %.0f%% (n=%d) | post last-base <=-thr %.0f%%, >=+thr %.0f%%"%(thr,(d1<=-thr).mean()*100,(d1>=thr).mean()*100,len(d1),(k.chg<=-thr).mean()*100,(k.chg>=thr).mean()*100))
# note: post change uses 2-yr mean baseline for these counties; compare same counties single-year to single-year: 2024-25 vs 2019-20
q=p.copy(); dl=q["last"]-q.SY2019_20
print("same counties, last - 2019-20 (single vs single): <=-2 %.0f%%, >=2 %.0f%%; median %.2f"%((dl<=-2).mean()*100,(dl>=2).mean()*100,dl.median()))
print("pre 1yr median change %.2f"%d1.median())
# which states in noise set
print("states in noise check:",sorted(p.State.unique()), "missing:",sorted(set(k.State)-set(p.State)))
# RTM: band by independent year. Use counties with both pre years: band by 2018-19, change = last - 2019-20
q["band_ind"]=pd.cut(q.SY2018_19,[0,90,95,101],right=False,labels=["<90","90-95","95+"])
q["band_same"]=pd.cut(q.SY2019_20,[0,90,95,101],right=False,labels=["<90","90-95","95+"])
print("== RTM check: median(last-2019-20) by band defined on same year vs independent year (2018-19)")
print(pd.DataFrame({"same":q.groupby("band_same",observed=True).apply(lambda g:(g["last"]-g.SY2019_20).median()),
 "indep":q.groupby("band_ind",observed=True).apply(lambda g:(g["last"]-g.SY2019_20).median()),
 "n_ind":q.band_ind.value_counts()}).round(2))
# pre slope with 2-yr mean baseline: what's the expected RTM for a 2yr mean? split-half reliability
r=np.corrcoef(p.SY2018_19,p.SY2019_20)[0,1]; print("test-retest r 2018-19 vs 2019-20: %.3f; implied reliability of 2-yr mean (Spearman-Brown) %.3f"%(r,2*r/(1+r)))
# repeal effect with alternate 'last' = mean(2023-24, 2024-25) and with band by 2018-19
def eff(df,y,b,treat):
    X=np.column_stack([np.ones(len(df)),b-95,treat.astype(float)]); return np.linalg.lstsq(X,y,rcond=None)[0][2]
kk=k.copy(); kk["last2"]=np.where(kk.State=="CT",kk.SY2023_24,kk[["SY2023_24","SY2024_25"]].mean(axis=1))
for st_ in ["CT","CA"]:
    m=~kk.State.isin({"CT","CA"}-{st_})
    z=kk[m]; print(st_,"effect last=1yr %.2f, last=2yr-mean %.2f"%(eff(z,(z["last"]-z.base).values,z.base.values,z.State.eq(st_).values), eff(z,(z.last2-z.base).values,z.base.values,z.State.eq(st_).values)))
# CT vs neighbors and median counties
for s_ in ["CT","MA","VT","CA"]:
    x=k[k.State==s_]; print(s_,"n",len(x),"median base %.1f last %.1f chg %.1f"%(x.base.median(),x["last"].median(),x.chg.median()))
# CT trajectory by year
print(k[k.State=="CT"][["County"]+PRE+TR+["SY2023_24"]].round(1).to_string(index=False))
print("CA state median by year:",k[k.State=="CA"][PRE+TR+["SY2023_24","SY2024_25"]].median().round(1).to_dict())
print("others median by year:",k[~k.State.isin(["CA","CT"])][PRE+TR+["SY2023_24","SY2024_25"]].median().round(1).to_dict())
