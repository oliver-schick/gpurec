"""Ancestral genome size at each clade ancestor, from per_node_copies' .per_clade.json.

Horizontal bar chart of the reconstructed genome size (number of gene families inferred
present, summed posterior presence) at the ancestor of each named clade, under the
branch-wise (FULLbasin) model on the Euryarchaeota-rooted tree. The root ancestor
(LACA) is highlighted -- it is the families-at-root = origination-at-root quantity that
Figure 2 reports under three models.

  python experiments/plot_clade_genome_sizes.py pnc_Eury.per_clade.json out.png
"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else "clade_genome_sizes.png"

# (label, genome_size); rename LACA(root) for display
rows = [(k, v["genome_size"]) for k, v in d.items()]
rows.sort(key=lambda kv: kv[1])                              # ascending -> biggest on top
labels = [("LACA (root)" if k == "LACA(root)" else k) for k, _ in rows]
sizes = [v for _, v in rows]
is_root = [k == "LACA(root)" for k, _ in rows]

DARK, ROOT = "#5a7aa8", "#b5341f"
colors = [ROOT if r else DARK for r in is_root]
fig, ax = plt.subplots(figsize=(7.6, max(4.0, 0.32 * len(rows) + 1.2)))
y = range(len(rows))
ax.barh(list(y), sizes, color=colors, edgecolor="k", linewidth=0.5)
for yi, s in zip(y, sizes):
    ax.text(s + max(sizes) * 0.01, yi, f"{s:.0f}", va="center", fontsize=8.5,
            weight="bold" if is_root[yi] else "normal",
            color=ROOT if is_root[yi] else "0.2")
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("reconstructed ancestral genome size  (gene families present)", fontsize=11)
ax.set_title("Ancestral genome size at each clade ancestor\n"
             "branch-wise DTL+origination model, Euryarchaeota root", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.25)
ax.margins(x=0.12, y=0.01)
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
laca = d.get("LACA(root)", {}).get("genome_size")
print(f"wrote {out}  ({len(rows)} ancestors; LACA(root)={laca})")
