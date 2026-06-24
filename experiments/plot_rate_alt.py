"""Rate-structure figure: per-branch rate departure from its clade's mean rate,
three panels side by side (D / L / T), shared x-range. D & T are clade-constant
(sharp spike at 0); loss is branch-variable (broad)."""
import json, numpy as np, sys
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "rate_violin_data.json"))
out = sys.argv[2] if len(sys.argv) > 2 else "rate_deviation.png"
RATES = [("Duplication", "Df", "#4C72B0"), ("Loss", "Lf", "#C44E52"), ("Transfer", "Tf", "#55A868")]
R2 = {"Duplication": 0.94, "Loss": 0.38, "Transfer": 0.94}
cm = defaultdict(lambda: defaultdict(list))
for r in rows:
    for _, ff, _ in RATES:
        cm[r["clade"]][ff].append(r[ff])
clade_mean = {c: {ff: float(np.mean(v)) for ff, v in d.items()} for c, d in cm.items()}

XR = 2.5
bins = np.linspace(-XR, XR, 51)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
for ax, (name, ff, col) in zip(axes, RATES):
    dev = np.array([r[ff] - clade_mean[r["clade"]][ff] for r in rows])
    ax.hist(dev, bins=bins, color=col, alpha=0.85, density=True, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlim(-XR, XR)
    ax.set_title(f"{name}\n$R^2$(clade) = {R2[name]:.2f}", fontsize=13,
                 weight=("bold" if name == "Loss" else "normal"))
    ax.set_xlabel("per-branch rate vs clade mean  [$\\log_2$ fold]", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, ls=":")
axes[0].set_ylabel("density of branches", fontsize=11)
fig.suptitle("How far each branch departs from its clade's mean rate — "
             "duplication & transfer are clade-constant, loss is branch-variable", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
