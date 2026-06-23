"""LACA bar chart: gene families at the root, big tree (Eury). Numbers from
undine/laca_Eury.json (thresholded content PP>=0.5; the expected counts are NOT shown).
  total families analysed = 7059
  at root: branchwise (free per-branch) = 903 ; clade-grouped (AleRax rates) = 772
  shared = 722 (Jaccard 0.76)
"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# defaults from laca_Eury.json (override by passing the json path as argv[1])
F, bw, ax_, shared, jac = 7059, 903, 772, 722, 0.758
if len(sys.argv) > 1:
    d = json.load(open(sys.argv[1]))
    F, bw, ax_, shared = d["F"], d["content_full"], d["content_alerax"], d["inter"]
    jac = d["jaccard"]
out = sys.argv[2] if len(sys.argv) > 2 else "laca_families.png"

fig, ax = plt.subplots(figsize=(5.4, 5.2))
x = [0, 1]
labels = ["branchwise\n(free per-branch)", "clade-grouped\n(AleRax rates)"]
ax.bar(x, [shared, shared], width=0.62, color="#2c4a78", label=f"shared ({shared})")
ax.bar(x, [bw - shared, ax_ - shared], bottom=[shared, shared], width=0.62,
       color="#9bb4d4", label="model-specific")
for xi, tot in zip(x, [bw, ax_]):
    ax.text(xi, tot + 22, f"{tot}", ha="center", fontsize=14, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5)
ax.set_ylabel("gene families at the root  (LACA genome)", fontsize=11.5)
ax.set_title(f"Last archaeal common ancestor — genes at the root\n"
             f"{F:,} gene families analysed  ·  {shared} shared (Jaccard {jac:.2f})",
             fontsize=11.5)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax.set_ylim(0, max(bw, ax_) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
