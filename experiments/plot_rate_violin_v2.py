"""Per-clade DTL rates, branchwise (free per-branch) vs shared (clade-grouped), on a
PROPER LOG RATE axis (actual rates, not log2 units), with loss highlighted to show how
the clade-grouped model collapses real within-clade loss-rate variation.

Reads rate_violin_data.json (Df/Lf/Tf, Ds/Ls/Ts are log2 rates)."""
import json, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

src = sys.argv[1] if len(sys.argv) > 1 else "rate_violin_data.json"
out = sys.argv[2] if len(sys.argv) > 2 else "rate_violin_v2.png"
rows = json.load(open(src))
L10 = math.log10(2.0)  # log2 rate -> log10 rate

# order clades by number of branches (most populated first)
clades = sorted({r["clade"] for r in rows}, key=lambda c: -sum(1 for r in rows if r["clade"] == c))
ns = [sum(1 for r in rows if r["clade"] == c) for c in clades]
RATES = [("Duplication", "Df", "Ds", "#4C72B0", 0.94),
         ("Loss",        "Lf", "Ls", "#C44E52", 0.38),   # the heterogeneous one -> red
         ("Transfer",    "Tf", "Ts", "#55A868", 0.94)]

fig, axes = plt.subplots(3, 1, figsize=(max(12, len(clades) * 0.6), 11.5), sharex=True)
pos = np.arange(len(clades))
# common log10-rate tick locations -> labelled as actual rates
yt = [-3, -2, -1, 0, 1]
ytlab = ["0.001", "0.01", "0.1", "1", "10"]

for ax, (name, ff, fs, col, r2) in zip(axes, RATES):
    spreads = []
    for i, c in enumerate(clades):
        vals = [r[ff] * L10 for r in rows if r["clade"] == c]          # log10 rate
        sh = float(np.median([r[fs] * L10 for r in rows if r["clade"] == c]))
        if len(vals) >= 3:
            vp = ax.violinplot([vals], positions=[i], widths=0.85, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.5); b.set_edgecolor(col); b.set_linewidth(0.5)
            spreads.append(max(vals) - min(vals))
        ax.scatter([i] * len(vals), vals, color="#333333", s=8, alpha=0.45, zorder=4)
        ax.scatter([i], [sh], color="k", marker="D", s=34, zorder=6,
                   facecolor="white", edgecolor="k", linewidth=1.1)
    ax.set_yscale("linear")
    ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FixedFormatter(ytlab))
    ax.set_ylabel(f"{name}\nrate (events/branch)", fontsize=11)
    note = (f"$R^2$(clade) = {r2:.2f}   (least clade-structured)"
            if name == "Loss" else f"$R^2$(clade) = {r2:.2f}   (clade-constant)")
    ax.text(0.004, 0.90, note, transform=ax.transAxes, fontsize=11.5,
            weight=("bold" if name == "Loss" else "normal"),
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.grid(axis="y", which="major", alpha=0.3, ls=":")
axes[1].set_facecolor("#fcf3f3")  # tint the Loss panel to draw the eye

# legend (proxy handles)
from matplotlib.lines import Line2D
axes[0].legend(handles=[
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#4C72B0", markersize=12,
           alpha=0.6, label="branchwise (free per-branch): spread of branch rates within a clade"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="k",
           markersize=9, label="shared (clade-grouped): one rate per clade")],
    loc="upper right", fontsize=9.5, framealpha=0.95)
axes[-1].set_xticks(pos)
axes[-1].set_xticklabels([f"{c}  (n={n})" for c, n in zip(clades, ns)], rotation=60, ha="right", fontsize=8)
fig.suptitle("Per-clade duplication, loss and transfer rates: clade-grouped vs.\\ fully branch-wise\n"
             "Duplication and transfer are clade-constant ($R^2\\approx0.94$); loss departs from "
             "clade-constancy, mainly along the backbone ($R^2\\approx0.38$)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  ({len(clades)} clades)")
