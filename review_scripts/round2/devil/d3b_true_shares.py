"""True share of |D|<1, D<=-2, D>=2 under the heterogeneity scenarios of d3_error_rates.py.
Usage: python3 -I d3b_true_shares.py /home/claude/rerun/county_classes.csv"""
import sys, numpy as np, pandas as pd
rng = np.random.default_rng(5)
k = pd.read_csv(sys.argv[1]); k = k[k.cls.notna()].copy(); k["chg"] = k["last"] - k.base
B = pd.cut(k.base, [0, 90, 95, 101], right=False, labels=["<90", "90-95", "95+"]).astype(str).to_numpy()
sm = k.groupby("State").chg.transform("median").to_numpy()
SU = {"<90": 4.2, "90-95": 2.7, "95+": 1.8}
for nm, D in [("states only", sm), ("states+within cov-test SD", sm + rng.normal(0, 1, (200, len(k))) * np.array([SU[b] for b in B])),
              ("states+within SD 3.5", sm + rng.normal(0, 3.5, (200, len(k))))]:
    print(f"{nm:28s} |D|<1 {np.mean(np.abs(D)<1)*100:.1f}%  D<=-2 {np.mean(D<=-2)*100:.1f}%  D<=-1 {np.mean(D<=-1)*100:.1f}%  D>=1 {np.mean(D>=1)*100:.1f}%  D>=2 {np.mean(D>=2)*100:.1f}%")
