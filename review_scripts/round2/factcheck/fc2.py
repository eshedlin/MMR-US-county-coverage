import sys, numpy as np, pandas as pd
rng=np.random.default_rng(11)
PRE=["SY2018_19","SY2019_20"]; TR=["SY2020_21","SY2021_22","SY2022_23"]
p0=pd.read_csv(sys.argv[1]); k=p0[p0.cls.notna()].copy()
k["nb"]=k[PRE].notna().sum(axis=1); k["nt"]=k[TR].notna().sum(axis=1)
k["ch"]=(k["last"]-k.base).round(6)
p=p0.dropna(subset=PRE).copy(); p["dd"]=p.SY2019_20-p.SY2018_19
p["band"]=pd.cut(p[PRE].mean(axis=1),[0,90,95,101],right=False,labels=["<90","90-95","95+"])
print("pool n",len(p), "states", p.State.nunique())
print("state median dd (18-19->19-20):", p.groupby("State").dd.median().round(2).sort_values().to_dict())
def pools(df, demean=False):
    x=df.dd-(df.groupby("State").dd.transform("median") if demean else 0)
    return {b:(x[df.band==b]/np.sqrt(2)).to_numpy() for b in ["<90","90-95","95+"]}
P_orig=pools(p); P_dm=pools(p,True); P_ex=pools(p[~p.State.isin(["CA","CO"])]); P_exdm=pools(p[~p.State.isin(["CA","CO"])],True)
for nm,P in [("orig",P_orig),("demeaned",P_dm),("exCA,CO",P_ex),("exCACO demeaned",P_exdm)]:
    print(nm,"per-year noise SD by band:",{b:round(v.std(),2) for b,v in P.items()}, "robust(1.4826*MAD):",{b:round(1.4826*np.median(np.abs(v-np.median(v))),2) for b,v in P.items()})
r1=p.SY2018_19.corr(p.SY2019_20); print("r1",round(r1,3))
def sim(delta,P,reps=200,thr=1,ret_states=False):
    out=[]; smeds=[]
    dl=np.broadcast_to(np.asarray(delta,float),(len(k),))
    for _ in range(reps):
        rel=np.where(k.nb==2,2*r1/(1+r1),r1)
        m=k.groupby("State").base.transform("mean").to_numpy()
        T=m+np.sqrt(rel)*(k.base.to_numpy()-m)
        bT=pd.cut(T,[-1e9,90,95,1e9],right=False,labels=["<90","90-95","95+"]).astype(str)
        e=np.zeros((len(k),6))
        for bn in P:
            mm=bT==bn; e[mm]=rng.choice(P[bn],size=(mm.sum(),6))*rng.choice([-1,1],size=(mm.sum(),6))
        bs=np.minimum(T+np.where(k.nb==2,e[:,:2].mean(1),e[:,0]),100)
        ls=np.minimum(T+dl+e[:,5],100)
        ts=np.minimum(T+dl+np.where(k.nt==3,e[:,2:5].min(1),np.where(k.nt==2,e[:,2:4].min(1),e[:,2])),100)
        ch=ls-bs; up=ch>=thr; dn=ch<=-thr; rec=~up&~dn&(ts-bs<=-thr)
        out.append([dn.mean()*100,up.mean()*100,rec.mean()*100,(~up&~dn&~rec).mean()*100])
        smeds.append(pd.Series(ch).groupby(k.State.values).median())
    o=np.array(out); sm=pd.concat(smeds,axis=1)
    return o.mean(0).round(1), o.std(0).round(2), sm
def obs(thr=1, sub=None):
    kk=k if sub is None else k[sub]; ch=kk.ch
    up=ch>=thr; dn=ch<=-thr; rec=~up&~dn&((kk.trough-kk.base).round(6)<=-thr)
    return np.round([dn.mean()*100,up.mean()*100,rec.mean()*100,(~up&~dn&~rec).mean()*100],1)
osm=k.groupby("State").ch.median()
print("\nobserved thr1 [down,up,dip+rec,stable]",obs(1))
print("observed thr2",obs(2))
stdelta=k.State.map(osm).to_numpy()
for nm,dl,P in [("common -2.9, orig pool",-2.9,P_orig),("common -2.9, demeaned pool",-2.9,P_dm),("common -2.9, exCACO demeaned",-2.9,P_exdm),
                ("state-specific delta, orig pool",stdelta,P_orig),("state-specific delta, demeaned pool",stdelta,P_dm)]:
    mn,sd,sm=sim(dl,P)
    print(f"{nm}: mean {mn} rep-SD {sd}")
    if np.isscalar(dl):
        print("   SD of 22 state medians: null mean %.2f (95%% %.2f-%.2f) vs observed %.2f; null range of state medians typical %.2f..%.2f"%(
            sm.std().mean(), sm.std().quantile(.025), sm.std().quantile(.975), osm.std(), sm.min().mean(), sm.max().mean()))
# pure noise at thr 2 and thr 1
for thr in [1,2]:
    for nm,P in [("orig",P_orig),("demeaned",P_dm)]:
        mn,sd,_=sim(0,P,thr=thr); print(f"pure noise thr {thr} pool {nm}: down,up,dip+rec,stable {mn}")
# excluding CA, CO, CT from observed vs common-null restricted
sub=~k.State.isin(["CA","CO","CT"]).to_numpy()
print("observed excl CA,CO,CT thr1",obs(1,sub))
