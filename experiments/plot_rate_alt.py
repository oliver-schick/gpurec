# (committed copy of the alt-rate plots: deviation histogram + branchwise-vs-clade-mean scatter)
import json, numpy as np, sys
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=json.load(open(sys.argv[1] if len(sys.argv)>1 else "rate_violin_data.json"))
RATES=[("Duplication","Df","#4C72B0"),("Loss","Lf","#C44E52"),("Transfer","Tf","#55A868")]
R2={"Duplication":0.94,"Loss":0.38,"Transfer":0.94}
cm=defaultdict(lambda: defaultdict(list))
for r in rows:
    for _,ff,_ in RATES: cm[r["clade"]][ff].append(r[ff])
clade_mean={c:{ff:float(np.mean(v)) for ff,v in d.items()} for c,d in cm.items()}
fig,ax=plt.subplots(figsize=(8.6,5))
for name,ff,col in RATES:
    dev=np.array([r[ff]-clade_mean[r["clade"]][ff] for r in rows])
    ax.hist(dev,bins=np.linspace(-2.5,2.5,71),alpha=.55,color=col,density=True,label=f"{name}  ($R^2$={R2[name]:.2f})")
ax.axvline(0,color="k",lw=1); ax.set_xlim(-2.5,2.5)
ax.set_xlabel("per-branch rate vs its clade's mean rate   [$\\log_2$ fold]"); ax.set_ylabel("density of branches")
ax.set_title("Departure of each branch from its clade mean: D & T clade-constant, loss branch-variable")
ax.legend(fontsize=12); fig.tight_layout(); fig.savefig(sys.argv[2] if len(sys.argv)>2 else "rate_deviation.png",dpi=150,bbox_inches="tight")
