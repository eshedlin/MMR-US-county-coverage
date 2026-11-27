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

# ---- Figure 1: median change by year, band = middle half of counties
BAND = "#cfe1f7"
x = np.arange(len(YRS))
rest = a  # all analysed counties; CT has no 2024-25 value
chg_y = pd.DataFrame({y: np.round(rest[y] - rest["SY2019_20"], 6) for y in YRS})
med, q1, q3 = chg_y.median(), chg_y.quantile(.25), chg_y.quantile(.75)
ct = med_path(a[a.State == "CT"])[:5]
below = (rest.chg < 0).mean() * 100

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
ax.fill_between(x, q1, q3, color=BAND, lw=0, zorder=1)
ax.axhline(0, color=TXT2, lw=1, zorder=2)
ax.plot(x, med, color=BLUE, lw=2.5, marker="o", ms=8, mec="white", mew=2, zorder=4)
ax.plot(x[:5], ct, color=ORANGE, lw=2.5, marker="o", ms=8, mec="white", mew=2, zorder=4)
# direct labels
ax.text(x[-1] + 0.12, med.iloc[-1], f"Median county\n{med.iloc[-1]:+.1f} pp", color=BLUE, va="center", fontsize=11, weight="bold")
ax.text(x[4] + 0.12, ct[-1], f"Connecticut\n{ct[-1]:+.1f} pp", color=ORANGE, va="center", fontsize=11, weight="bold")
ax.text(x[-1] + 0.12, (q1.iloc[-1] + q3.iloc[-1]) / 2 - 1.7, "Middle half\nof counties", color=TXT2, va="center", fontsize=10)
ax.annotate("CT preschool exemptions\nexpire (Sept 2022)", xy=(x[3], ct[3]), xytext=(x[2] - 0.2, 2.3),
            fontsize=9.5, color=TXT2, ha="center", arrowprops=dict(arrowstyle="-", color=TXT2, lw=0.8))
ax.set_xticks(x, LAB)
ax.set_xlim(-0.2, len(YRS) - 0.2 + 1.0)
ax.set_ylim(-7.5, 3.5)
ax.set_ylabel("Change vs 2019-20 (percentage points)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s_ in ["top", "right", "left"]:
    ax.spines[s_].set_visible(False)
ax.tick_params(length=0)
fig.text(0.015, 0.965, "Kindergarten MMR coverage kept falling after COVID; Connecticut's rose",
         fontsize=14, weight="bold", color=TXT, va="top")
fig.text(0.015, 0.915, f"2-dose MMR, change since 2019-20. {len(rest):,} counties in 22 states; "
         f"by 2024-25, {below:.0f}% were below their 2019-20 level.", fontsize=10.5, color=TXT2, va="top")
fig.text(0.015, 0.015, "Data: JHU CSSE MMR_data (Dong et al., 2025). Unweighted counties, not children. "
         "Connecticut: 8 counties, series ends 2023-24.", fontsize=8.5, color=TXT2)
fig.tight_layout(rect=(0, 0.035, 1, 0.88))
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
print("saved", out / "fig_year_path.png", out / "fig_by_state.png")
