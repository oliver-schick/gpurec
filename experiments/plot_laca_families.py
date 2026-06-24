"""LACA genome-size bar chart, three models, isolating the origination effect from the
rate-granularity effect. Order: uniform clade -> non-uniform clade -> branchwise.
Reads laca_compare.py's 3-model JSON (uniform_clade / nonuniform_clade / branchwise)."""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else "laca_families.png"
F = d["F"]
models = [
    ("uniform clade\nclade rates, uniform origination",      d["uniform_clade"]["content"],    "#b0b8c1"),
    ("non-uniform clade\nclade rates, fitted origination",   d["nonuniform_clade"]["content"],  "#6b8cae"),
    ("branchwise\nper-branch rates, fitted origination",     d["branchwise"]["content"],        "#2c4a78"),
]
fig, ax = plt.subplots(figsize=(7.4, 5.4))
xs = range(len(models))
bars = ax.bar(xs, [m[1] for m in models], color=[m[2] for m in models],
              width=0.62, edgecolor="k", linewidth=0.7)
for b, m in zip(bars, models):
    ax.text(b.get_x() + b.get_width() / 2, m[1] + max(mm[1] for mm in models) * 0.02,
            f"{m[1]}", ha="center", fontsize=15, weight="bold")
ax.set_xticks(list(xs)); ax.set_xticklabels([m[0] for m in models], fontsize=10)
ax.set_ylabel("gene families at the root  (LACA genome size)", fontsize=11.5)
ax.set_title("Reconstructed last archaeal common ancestor genome size\n"
             f"under three models  ({F:,} gene families analysed)", fontsize=12)
ax.set_ylim(0, max(m[1] for m in models) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
# annotate the two contrasts
y = max(m[1] for m in models) * 1.10
ax.annotate("", xy=(1, y), xytext=(0, y), arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text(0.5, y * 1.005, "origination", ha="center", fontsize=9, color="0.35", style="italic")
ax.annotate("", xy=(2, y), xytext=(1, y), arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text(1.5, y * 1.005, "rate granularity", ha="center", fontsize=9, color="0.35", style="italic")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  ({[m[1] for m in models]})")
