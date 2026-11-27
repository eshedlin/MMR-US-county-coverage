"""Figures for the README. Usage: python make_figures.py out   (reads out/county_change.csv)."""
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = pathlib.Path(sys.argv[1])
a = pd.read_csv(out / "county_change.csv")
YRS = ["SY2019_20", "SY2020_21", "SY2021_22", "SY2022_23", "SY2023_24", "SY2024_25"]
LAB = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
BLUE, ORANGE, GRAY, TXT, TXT2, GRID = "#257de0", "#f47230", "#c8c8c8", "#191919", "#666666", "#ececec"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "font.family": "DejaVu Sans"})

def med_path(df):
    return [np.round(df[y] - df["SY2019_20"], 6).median() for y in YRS]

# ---- Figure 1: median change by year
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
x = np.arange(len(YRS))
for st, g in a.groupby("State"):
    if st in ("CT",):
        continue
    ax.plot(x, med_path(g), color=GRAY, lw=1, zorder=1)
allp = med_path(a[a.State != "CT"])
ct = med_path(a[a.State == "CT"])[:5]
ax.plot(x, allp, color=BLUE, lw=2.5, marker="o", ms=8, zorder=3, label="All counties, 21 states (median)")
ax.plot(x[:5], ct, color=ORANGE, lw=2.5, marker="o", ms=8, zorder=3,
        label="Connecticut, 8 counties (median; county series ends 2023-24)")
ax.axhline(0, color=TXT2, lw=1, zorder=2)
ax.annotate(f"{allp[-1]:+.1f}", (x[-1], allp[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", color=TXT, fontsize=12)
ax.annotate(f"{ct[-1]:+.1f}", (x[4], ct[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", color=TXT, fontsize=12)
ax.plot([], [], color=GRAY, lw=1, label="Each state's median county")
ax.set_xticks(x, LAB)
ax.set_ylabel("Change vs 2019-20 (percentage points)")
ax.set_title("Kindergarten 2-dose MMR coverage: change since 2019-20, median county",
             loc="left", color=TXT, fontsize=14)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=11)
fig.text(0.01, 0.005, "Data: Johns Hopkins CSSE MMR_data (Dong et al., 2025). Unweighted counties, not children. "
         "California 2017-19 excluded (different measure).", fontsize=9, color=TXT2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out / "fig_year_path.png")

# ---- Figure 2: state medians, dot plot
st = a.groupby("State").chg.agg(["median", "size"]).sort_values("median")
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
y = np.arange(len(st))
for i, (name, r) in enumerate(st.iterrows()):
    c = ORANGE if name == "CT" else BLUE
    ax.plot([0, r["median"]], [i, i], color=GRAY, lw=1, zorder=1)
    ax.scatter(r["median"], i, s=70, color=c, zorder=3, edgecolor="white", linewidth=2)
    ax.text(r["median"] + (0.25 if r["median"] >= 0 else -0.25), i, f"{r['median']:+.1f}  (n={int(r['size'])})",
            va="center", ha="left" if r["median"] >= 0 else "right", fontsize=10, color=TXT)
ax.axvline(0, color=TXT2, lw=1)
ax.set_yticks(y, st.index)
ax.set_xlim(-13, 5)
ax.set_xlabel("Median county change, 2019-20 to 2024-25 (CT to 2023-24), percentage points")
ax.set_title("The decline differs widely between states", loc="left", color=TXT, fontsize=14)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(out / "fig_by_state.png")
# ---- Figure 3: ranked table with sparklines (shared y-scale so amplitudes are comparable)
SLATE = "#5f7a8a"
rows = [("All counties", a)] + [(s_, g) for s_, g in a.groupby("State")]
tab = []
for name, g in rows:
    lvl = [g[y].median() for y in YRS]
    path_ = [np.round(g[y] - g["SY2019_20"], 6).median() if g[y].notna().any() else np.nan for y in YRS]
    last_i = 4 if name == "CT" else 5
    tab.append(dict(name=name, n=len(g), start=lvl[0], end=lvl[last_i], chg=g.chg.median(),
                    path=path_[:last_i + 1], ct=name == "CT"))
head, body = tab[0], sorted(tab[1:], key=lambda r: -r["chg"])
tab = [head] + body
lo = min(np.nanmin(r["path"]) for r in tab) - 0.5
hi = max(np.nanmax(r["path"]) for r in tab) + 0.5

nr = len(tab)
fig = plt.figure(figsize=(9.5, 0.62 * nr + 1.9), dpi=150)
top = 1 - 0.95 / fig.get_figheight()
rowh = (top - 0.75 / fig.get_figheight()) / nr
cols = dict(name=0.02, n=0.25, start=0.37, spark=(0.44, 0.30), end=0.83, chg=0.97)
fig.text(0.02, 1 - 0.3 / fig.get_figheight(),
         "Ranking counties' kindergarten MMR coverage by change since 2019-20, by state",
         fontsize=14, color=TXT, weight="bold", va="center")
hy = top + rowh * 0.35
for k_, lab, ha in [("name", "State", "left"), ("n", "Counties", "right"), ("start", "2019-20", "right"),
                    ("end", "2024-25", "right"), ("chg", "Change", "right")]:
    fig.text(cols[k_], hy, lab, fontsize=11, color=TXT2, ha=ha, va="center")
fig.text(cols["spark"][0] + cols["spark"][1] / 2, hy, "Median change by year (pp)", fontsize=11,
         color=TXT2, ha="center", va="center")
for i, r in enumerate(tab):
    yc = top - rowh * (i + 0.5)
    bold = "bold" if i == 0 else "normal"
    fig.add_artist(plt.Line2D([0.02, 0.98], [top - rowh * (i + 1)] * 2, color=GRID, lw=0.8 if i else 1.5))
    fig.text(cols["name"], yc, r["name"] + (" *" if r["ct"] else ""), fontsize=11, color=TXT, va="center", weight=bold)
    fig.text(cols["n"], yc, f"{r['n']}", fontsize=10.5, color=TXT2, va="center", ha="right")
    fig.text(cols["start"], yc, f"{r['start']:.1f}", fontsize=11, color=TXT, va="center", ha="right")
    fig.text(cols["end"], yc, f"{r['end']:.1f}", fontsize=11, color=TXT, va="center", ha="right")
    fig.text(cols["chg"], yc, (f"{r['chg']:+.1f}" if round(r['chg'], 1) != 0 else "0.0"), fontsize=11, color=TXT, va="center", ha="right", weight=bold)
    ax = fig.add_axes([cols["spark"][0], yc - rowh * 0.46, cols["spark"][1], rowh * 0.92])
    c = BLUE if i == 0 else (ORANGE if r["ct"] else SLATE)
    xs = np.arange(len(r["path"]))
    ax.axhline(0, color=TXT2, lw=0.6)
    ax.plot(xs, r["path"], color=c, lw=2, marker="o", ms=2.5)
    ax.set_xlim(-0.3, 5.3); ax.set_ylim(lo, hi); ax.axis("off")
foot = ("2019-20 and 2024-25: median county coverage (%), 2 doses MMR, kindergarten. Change: median of the county changes,\n"
        "so it can differ from the difference of the two medians. Lines share one scale "
        f"({lo + 0.5:.0f} to {hi - 0.5:+.0f} pp); flat line = 2019-20; gaps = year not reported.\n"
        "* Connecticut's county series ends 2023-24, so its last value is 2023-24.  Data: JHU CSSE MMR_data (Dong et al., 2025).")
fig.text(0.02, 0.4 / fig.get_figheight(), foot, fontsize=8.5, color=TXT2, va="center", linespacing=1.5)
fig.savefig(out / "fig_ranking_table.png")
print("saved", out / "fig_year_path.png", out / "fig_by_state.png", out / "fig_ranking_table.png")
