"""Violin plot: per-clade DTL rates, BRANCHWISE (free per-branch, violins) vs SHARED
(clade-grouped, diamonds), one panel per rate (D, L, T). Reads rate_violin_data.json."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "rate_violin_data.json"
out = sys.argv[2] if len(sys.argv) > 2 else "rate_violin.png"
rows = json.load(open(src))

# order clades by number of branches (most populated first)
clades = sorted({r["clade"] for r in rows}, key=lambda c: -sum(1 for r in rows if r["clade"] == c))
ns = [sum(1 for r in rows if r["clade"] == c) for c in clades]
RATES = [("D", "Df", "Ds", "Duplication"), ("L", "Lf", "Ls", "Loss"), ("T", "Tf", "Ts", "Transfer")]
R2 = {"D": 0.94, "L": 0.38, "T": 0.94}
FLOOR = -33.0  # theta floor (rate ~ 0); clip display so it doesn't crush the scale

fig, axes = plt.subplots(3, 1, figsize=(max(11, len(clades) * 0.55), 11), sharex=True)
pos = np.arange(len(clades))
for ax, (key, ff, fs, name) in zip(axes, RATES):
    shareds = []
    for i, c in enumerate(clades):
        vals = [r[ff] for r in rows if r["clade"] == c]
        sh = [r[fs] for r in rows if r["clade"] == c]
        shareds.append(float(np.median(sh)))
        vclip = [max(v, FLOOR + 1) for v in vals]
        if len(vals) >= 3:
            vp = ax.violinplot([vclip], positions=[i], widths=0.85, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor("#4C72B0"); b.set_alpha(0.55); b.set_edgecolor("#2c4a78")
        ax.scatter([i] * len(vals), vclip, color="#33415c", s=9, alpha=0.5, zorder=4)
    ax.scatter(pos, shareds, color="#C44E52", marker="D", s=34, zorder=6,
               edgecolor="k", linewidth=0.4, label="shared (clade-grouped)")
    ax.set_ylabel(f"{name}\nlog₂ rate", fontsize=11)
    ax.text(0.004, 0.90, f"R²(clade) = {R2[key]:.2f}", transform=ax.transAxes,
            fontsize=12, weight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.grid(axis="y", alpha=0.25)
axes[0].legend(loc="upper right", fontsize=10, framealpha=0.9)
axes[-1].set_xticks(pos)
axes[-1].set_xticklabels([f"{c}  (n={n})" for c, n in zip(clades, ns)], rotation=60, ha="right", fontsize=8)
fig.suptitle("Per-clade DTL rates  —  branchwise (free per-branch, violins)  vs  shared "
             "(clade-grouped, ◆)\nbig tree, Euryarchaeota root", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  ({len(clades)} clades, {len(rows)} branches)")
