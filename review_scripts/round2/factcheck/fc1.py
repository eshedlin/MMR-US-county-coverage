import sys, numpy as np, pandas as pd
k = pd.read_csv(sys.argv[1]); k = k[k.cls.notna()].copy()
PRE=["SY2018_19","SY2019_20"]; TR=["SY2020_21","SY2021_22","SY2022_23"]
k["nb"]=k[PRE].notna().sum(1); k["ch"]=(k["last"]-k.base).round(6)
print("n",len(k)); one=k[k.nb==1]; print("nb1",len(one), one.State.value_counts().to_dict(), "only2019-20:", one.SY2019_20.notna().sum())
two=k[k.nb==2]; print("both",len(two),two.State.nunique(), (two.cls.value_counts(normalize=True)*100).round(1).to_dict())
print("obs shares",(k.cls.value_counts(normalize=True)*100).round(1).to_dict())
# declined
print("ch<0 %.1f  ch<=-1 %.1f ch<=-2 %.1f"%((k.ch<0).mean()*100,(k.ch<=-1).mean()*100,(k.ch<=-2).mean()*100))
sm=k.groupby("State").ch.median().sort_values(); print("state median change:\n",sm.round(2).to_dict()); print("states median<0:",(sm<0).sum(),"of",len(sm))
print("state medians range", sm.min(), sm.max())
print("median",k.ch.median())
# claim 3 independent banding
def ib(df,lab):
    d=df.dropna(subset=["SY2018_19","SY2019_20"]).copy()
    d["band"]=pd.cut(d.SY2018_19,[0,90,95,101],right=False,labels=["<90","90-95","95+"])
    d["c2"]=d["last"]-d.SY2019_20
    g=d.groupby("band",observed=True).c2.agg(["median","count"]); print(lab, g.round(2).to_dict())
    return d
d=ib(k,"all")
ib(k[~k.State.isin(["CA","CO"])],"exCA,CO")
ib(k[~k.State.isin(["CA","CO","CT"])],"exCA,CO,CT")
print("nb2 states:",sorted(two.State.unique()))
# by state independent banding: <90 minus 95+ difference within states
dd=d[~d.State.isin(["CA","CO"])]
rows=[]
for s,g in dd.groupby("State"):
    m=g.groupby("band",observed=True).c2.median(); n=g.band.value_counts()
    rows.append([s,len(g)]+[round(m.get(b,np.nan),2) for b in ["<90","90-95","95+"]]+[n.get(b,0) for b in ["<90","90-95","95+"]])
print(pd.DataFrame(rows,columns=["st","n","<90","90-95","95+","n<90","n90","n95"]).to_string())
# regression-style: within-state comparison, change from 2019-20 on 2018-19 band dummies with state FE
X=pd.get_dummies(dd.band).astype(float).iloc[:,:2]; S=pd.get_dummies(dd.State).astype(float)
A=np.column_stack([X.values,S.values]); b=np.linalg.lstsq(A,dd.c2.values,rcond=None)[0]; print("within-state FE (exCA,CO): <90 vs 95+ %.2f, 90-95 vs 95+ %.2f"%(b[0],b[1]))
# also mean
print("means by band exCA,CO", dd.groupby("band",observed=True).c2.mean().round(2).to_dict())
# 'up' counties in <90: what fraction of up among <90 under indep design?
