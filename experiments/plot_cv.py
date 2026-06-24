"""Cross-validation / overfitting figure: the branch-wise model's per-family
predictive advantage over the clade-grouped model, on the families used to fit the
rates vs. families withheld. Both positive => the extra parameters generalise."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# from cv_eval_Eury (FULLbasin, train_n=3530)
free   = dict(train=-1373618.1, test=-257724.7, ntr=3530, nte=3529)
grp    = dict(train=-1383169.5, test=-259280.5, ntr=3530, nte=3529)
dtr = free["train"]/free["ntr"] - grp["train"]/grp["ntr"]    # +2.71
dte = free["test"]/free["nte"]  - grp["test"]/grp["nte"]     # +0.44
fig, ax = plt.subplots(figsize=(6.4, 5))
bars = ax.bar(["families used\nto fit the rates\n(n=3,530)", "families withheld\nfrom fitting\n(n=3,529)"],
              [dtr, dte], color=["#9aa7b5", "#4C72B0"], width=0.6, edgecolor="k", linewidth=0.6)
ax.axhline(0, color="k", lw=1.2)
for b, v in zip(bars, [dtr, dte]):
    ax.text(b.get_x()+b.get_width()/2, v+0.07, f"+{v:.2f}", ha="center", fontsize=13, weight="bold")
ax.set_ylabel("branch-wise advantage\n($\\Delta$ log-likelihood per gene family)", fontsize=11.5)
ax.set_ylim(0, dtr*1.25)
ax.set_title("Does the per-branch model just fit noise?\n"
             "It predicts better even on families withheld from fitting $\\Rightarrow$ not overfitting",
             fontsize=12)
ax.text(0.5, -0.34, "If the extra ~2,000 parameters were fitting noise, the right-hand bar would be $\\leq$ 0.",
        transform=ax.transAxes, ha="center", fontsize=9.5, style="italic", color="0.3")
fig.tight_layout(); fig.savefig("/tmp/cv_overfit.png", dpi=150, bbox_inches="tight")
print(f"dtrain={dtr:.3f}/fam  dtest={dte:.3f}/fam")
