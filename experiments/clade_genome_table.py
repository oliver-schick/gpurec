"""Per-clade-ancestor genome size under TWO rate sets, side by side.

Reads two per_node_copies .per_clade.json files (e.g. AleRax paper rates and the
branch-wise FULLbasin rates) and prints/plots, per named clade ancestor:
  copies      = Σ_f gene copies  (S+SL+leaf, the ALE branch_counts["copies"])
  pres>0.5    = # families with posterior presence > 0.5

  python experiments/clade_genome_table.py paper:pnc_Eury_AX.per_clade.json \
         branchwise:pnc_Eury_BW.per_clade.json  [out.png]
"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

specs = [a for a in sys.argv[1:] if ":" in a and a.endswith(".json")]
out = next((a for a in sys.argv[1:] if a.endswith(".png")), None)
labels, data = [], []
for sp in specs:
    lab, path = sp.split(":", 1)
    labels.append(lab); data.append(json.load(open(path)))

clades = sorted(data[0], key=lambda k: -data[0][k]["n_presence_gt0p5"])

# ---- printed table ----
hdr = f"{'clade':20s}"
for lab in labels:
    hdr += f" | {lab+' cop':>10s} {lab+' p>.5':>8s}"
print(hdr); print("-" * len(hdr))
for k in clades:
    row = f"{('LACA(root)' if k=='LACA(root)' else k):20s}"
    for d in data:
        v = d.get(k, {})
        row += f" | {v.get('sum_copies', 0):10.1f} {v.get('n_presence_gt0p5', 0):8d}"
    print(row)

# ---- comparison figure: presence>0.5 (left) and copies (right), both rate sets ----
if out:
    disp = [("LACA (root)" if k == "LACA(root)" else k) for k in clades]
    y = np.arange(len(clades)); h = 0.8 / len(data)
    cols = ["#5a7aa8", "#c07a30", "#5fa05f"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, max(4.5, 0.34 * len(clades) + 1.3)), sharey=True)
    for i, (lab, d) in enumerate(zip(labels, data)):
        pres = [d.get(k, {}).get("n_presence_gt0p5", 0) for k in clades]
        cop = [d.get(k, {}).get("sum_copies", 0) for k in clades]
        off = (i - (len(data) - 1) / 2) * h
        axL.barh(y + off, pres, height=h, color=cols[i % 3], edgecolor="k", linewidth=0.4, label=lab)
        axR.barh(y + off, cop, height=h, color=cols[i % 3], edgecolor="k", linewidth=0.4, label=lab)
    axL.set_yticks(y); axL.set_yticklabels(disp, fontsize=9); axL.invert_yaxis()
    axL.set_xlabel("# families present  (posterior presence > 0.5)", fontsize=10.5)
    axR.set_xlabel("summed gene copies  (S + SL + leaf)", fontsize=10.5)
    axL.set_title("families present", fontsize=11); axR.set_title("gene copies", fontsize=11)
    for ax in (axL, axR):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Reconstructed ancestral genome size per clade ancestor (Euryarchaeota root)\n"
                 "paper (AleRax DTL+O) vs branch-wise rates  ·  100 samples/family, C++ backtrack",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")
