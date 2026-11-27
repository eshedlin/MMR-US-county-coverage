import sys, numpy as np, pandas as pd
k=pd.read_csv(sys.argv[1]); k=k[k.cls.notna()].copy(); k["chg"]=k["last"]-k.base
both=k.dropna(subset=["SY2018_19","SY2019_20"])
sm=both.groupby("State")[["SY2018_19","SY2019_20"]].median(); sm["d"]=sm.SY2019_20-sm.SY2018_19
print(sm.round(2).sort_values("d").to_string())
print("median of county diffs:",(both.SY2019_20-both.SY2018_19).groupby(both.State).median().round(2).sort_values().tail(4).to_dict())
def eff(df,trt,y="chg",base="base"):
    X=np.column_stack([np.ones(len(df)),df[base]-95,df.State.isin(trt).astype(float)])
    return np.linalg.lstsq(X,df[y].to_numpy(),rcond=None)[0][2]
for comp in [["MA"],["VT"],["MA","VT"]]:
    s=k[k.State.isin(["CT"]+comp)]
    s2=s.dropna(subset=["SY2023_24"]).assign(c2=lambda x:x.SY2023_24-x.base)
    print("CT vs",comp,"adj (pooled slope within these):%.2f"%eff(s,["CT"]),
          "| raw mean diff %.2f median diff %.2f"%(s[s.State=="CT"].chg.mean()-s[s.State!="CT"].chg.mean(), s[s.State=="CT"].chg.median()-s[s.State!="CT"].chg.median()),
          "| last=2023-24 for all adj %.2f (n=%d)"%(eff(s2,["CT"],"c2"),len(s2)))
# with slope from all 22 states (CT effect relative to each placebo)
non=k[~k.State.isin(["CA","CT"])]
X=lambda df,st: np.column_stack([np.ones(len(df)),df.base-95,df.State.eq(st).astype(float)])
def e_vs_rest(st):
    z=pd.concat([non[non.State!=st] if st!="CT" else non, k[k.State==st]]) if st=="CT" else non
    return np.linalg.lstsq(X(z,st),z.chg.to_numpy(),rcond=None)[0][2]
ct=e_vs_rest("CT"); ma=e_vs_rest("MA"); vt=e_vs_rest("VT")
print("vs-all-controls effects CT %.2f MA %.2f VT %.2f -> CT-MA %.2f CT-VT %.2f"%(ct,ma,vt,ct-ma,ct-vt))
for st in ["CT","MA","VT"]:
    g=k[k.State==st]; print(st,"n",len(g),"median base %.1f, years with data:"%g.base.median(), g[[c for c in k.columns if c.startswith("SY")]].notna().sum().to_dict())
print("CT counties by-year median:",k[k.State=="CT"][[c for c in k.columns if c.startswith("SY")]].median().round(1).to_dict())
for st in ["MA","VT"]: print(st,"by-year median:",k[k.State==st][[c for c in k.columns if c.startswith("SY")]].median().round(1).to_dict())
