import sys, numpy as np, pandas as pd
k=pd.read_csv(sys.argv[1]); k=k[k.cls.notna()].copy()
PRE=["SY2018_19","SY2019_20"]; Y=["SY2020_21","SY2021_22","SY2022_23","SY2023_24","SY2024_25"]
print("2018-19-only baseline:",k[k.SY2019_20.isna()][["State","County"]].values.tolist())
X=lambda df,y: np.linalg.lstsq(np.column_stack([np.ones(len(df)),df.base-95,df.State.eq("CT")]),y,rcond=None)[0][2]
z=k[k.State!="CA"].dropna(subset=["SY2023_24"]); print("CT alone, last=2023-24 for all: %.2f n=%d"%(X(z,(z.SY2023_24-z.base).to_numpy()),len(z)))
z=k[k.State!="CA"]; print("CT alone main: %.2f"%X(z,(z["last"]-z.base).to_numpy()))
# year profile: median (year - base) by year, all, ex CA/CO/CT, and within 796
for lab,s in [("all",k),("exCA,CO,CT",k[~k.State.isin(["CA","CO","CT"])]),("796 exCA,CO,CT",k[(k[PRE].notna().all(axis=1))&~k.State.isin(["CA","CO","CT"])])]:
    print(lab,{y[2:]:round((s[y]-s.base).median(),2) for y in Y}, "n",len(s))
# share whose last exceeds their own min COVID year by >=1, and median last-trough
for lab,s in [("all",k),("exCA,CO,CT",k[~k.State.isin(["CA","CO","CT"])])]:
    print(lab,"median last-trough %.2f; last>=trough+1 %.1f%%"%((s["last"]-s.trough).median(),((s["last"]-s.trough)>=1).mean()*100))
# state-level: states whose 2024-25 median above their 2021-22 median
s=k[~k.State.isin(["CT"])]
g=s.groupby("State")[["base"]+Y].median(); g["rec_21_to_25"]=g.SY2024_25-g.SY2021_22; g["trough_state"]=g[["SY2020_21","SY2021_22","SY2022_23"]].min(axis=1)
g["last_minus_statetrough"]=g.SY2024_25-g.trough_state
print(g[["base","SY2020_21","SY2021_22","SY2022_23","SY2024_25","last_minus_statetrough"]].round(1).sort_values("last_minus_statetrough").to_string())
