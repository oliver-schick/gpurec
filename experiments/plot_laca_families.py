"""LACA genome-size bar chart, three models, isolating origination from rate-granularity.
Order: uniform clade -> non-uniform clade -> branchwise. The last two bars (both fitted
origination) are split into the families SHARED between them vs each model's own.
Reads laca_compare.py's 3-model JSON."""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

d = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else "laca_families.png"
F = d["F"]
uni = d["uniform_clade"]["content"]
nu = d["nonuniform_clade"]["content"]
bw = d["branchwise"]["content"]
bwset = set(d["root_families_branchwise"]); nuset = set(d["root_families_nonuniform"])
shared = len(bwset & nuset); nu_only = nu - shared; bw_only = bw - shared

DARK, LIGHT, GREY = "#2c4a78", "#9bb4d4", "#b0b8c1"
fig, ax = plt.subplots(figsize=(7.4, 5.4))
x = [0, 1, 2]
# bar 1: uniform clade (solid grey)
ax.bar(0, uni, width=0.62, color=GREY, edgecolor="k", linewidth=0.7)
# bars 2,3: shared (dark) + model-specific (light)
ax.bar([1, 2], [shared, shared], width=0.62, color=DARK, edgecolor="k", linewidth=0.7)
ax.bar([1, 2], [nu_only, bw_only], bottom=[shared, shared], width=0.62,
       color=LIGHT, edgecolor="k", linewidth=0.7)
for xi, tot in zip(x, [uni, nu, bw]):
    ax.text(xi, tot + max(uni, nu, bw) * 0.02, f"{tot}", ha="center", fontsize=15, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["uniform clade\nclade rates, uniform origination",
                    "non-uniform clade\nclade rates, fitted origination",
                    "branchwise\nper-branch rates, fitted origination"], fontsize=10)
ax.set_ylabel("gene families at the root  (LACA genome size)", fontsize=11.5)
ax.set_title("Reconstructed last archaeal common ancestor genome size\n"
             f"under three models  ({F:,} gene families analysed)", fontsize=12)
ax.set_ylim(0, max(uni, nu, bw) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.legend(handles=[Patch(facecolor=DARK, edgecolor="k", label=f"shared ({shared})"),
                   Patch(facecolor=LIGHT, edgecolor="k", label="model-specific")],
          loc="upper left", bbox_to_anchor=(0.02, 0.66), fontsize=9.5, framealpha=0.95,
          title="last two bars", title_fontsize=8.5)
# annotate the two contrasts
y = max(uni, nu, bw) * 1.10
ax.annotate("", xy=(1, y), xytext=(0, y), arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text(0.5, y * 1.005, "origination", ha="center", fontsize=9, color="0.35", style="italic")
ax.annotate("", xy=(2, y), xytext=(1, y), arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text(1.5, y * 1.005, "rate granularity", ha="center", fontsize=9, color="0.35", style="italic")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  (uni={uni} nu={nu}[{shared}+{nu_only}] bw={bw}[{shared}+{bw_only}])")
