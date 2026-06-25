"""Stage B FULL (both branch- AND family-heterogeneous), partition step.

From the Stage-A rfx posterior gamma[K,F] in rfx_<root>.npz, assign each family to a
rate COMPONENT, then emit per-component:
  comp_<j>.ale_list  -- the .ale paths of that component's families (subset primitive)
  init_<j>.json      -- theta_log2 init = that regime's grid rate broadcast to all S branches
                        (warm start FROM Stage A; the fit refines it BRANCHWISE)
  manifest.json      -- active components + sizes + origin regime + centroid rate

Each component is then fit branchwise (its own theta_k[S,3]) on its families with the shared
p^O pinned -> the family-mixture-of-branchwise = BOTH heterogeneous. Family heterogeneity comes
from the partition (each family in its best component), branch heterogeneity from theta_k[S,3].

MAP mode (option 1): hard argmax over ACTIVE components (deterministic).
The exact .ale<->gamma alignment is asserted (len(ALE)==F) so file basenames map 1:1 to columns.
"""
import sys, json, glob, argparse, math
from pathlib import Path
import numpy as np

DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
ALE = sorted(p for p in glob.glob(str(DD / "3_UFBOOTs" / "ufboot_for_alerax" / "*.ale"))
             if not Path(p).name.startswith("._"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfx-npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-fam", type=int, default=50,
                    help="components with fewer MAP families than this are dropped; their "
                         "families reassign to their best surviving component.")
    ap.add_argument("--seed", type=int, default=0, help="(stochastic mode) sample component ~ gamma")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample each family's component from gamma (SEM E-step) instead of argmax")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.rfx_npz, allow_pickle=True)
    gamma = d["gamma"].astype(np.float64)              # [K,F]
    dlts = d["dlts"].astype(np.float64)                # [K,3]
    K, F = gamma.shape
    S = int(d["log_pO"].shape[0])
    if len(ALE) != F:
        raise SystemExit(f"ALIGNMENT FAIL: {len(ALE)} .ale files != gamma F={F}. "
                         "file<->column mapping unsafe; aborting.")
    # initial assignment
    if args.stochastic:
        rng = np.random.default_rng(args.seed)
        g = gamma / gamma.sum(0, keepdims=True)
        comp0 = np.array([rng.choice(K, p=g[:, f]) for f in range(F)])
    else:
        comp0 = gamma.argmax(0)                          # MAP
    counts = np.bincount(comp0, minlength=K)
    active = np.where(counts >= args.min_fam)[0]
    if len(active) == 0:
        active = np.array([int(counts.argmax())])
    # reassign every family to its best ACTIVE component
    gA = gamma[active, :]                                # [M,F]
    if args.stochastic:
        rng = np.random.default_rng(args.seed + 1)
        gAn = gA / gA.sum(0, keepdims=True)
        comp = np.array([active[rng.choice(len(active), p=gAn[:, f])] for f in range(F)])
    else:
        comp = active[gA.argmax(0)]
    manifest = {"root_rfx": str(args.rfx_npz), "S": S, "F": F, "n_components": int(len(active)),
                "min_fam": args.min_fam, "stochastic": bool(args.stochastic), "components": []}
    for j, k in enumerate(active):
        members = np.where(comp == k)[0]
        if len(members) == 0:
            continue
        (out / f"comp_{j:03d}.ale_list").write_text("\n".join(ALE[i] for i in members) + "\n")
        D, L, T = (float(x) for x in dlts[k])
        row = [math.log2(D), math.log2(L), math.log2(T)]
        json.dump({"theta_log2": [row for _ in range(S)]}, open(out / f"init_{j:03d}.json", "w"))
        manifest["components"].append({"ordinal": j, "regime_k": int(k), "n_fam": int(len(members)),
                                       "D": D, "L": L, "T": T})
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"  partitioned {F} families -> {len(manifest['components'])} active components "
          f"(min_fam={args.min_fam}, {'stochastic' if args.stochastic else 'MAP'})")
    for c in manifest["components"]:
        print(f"    comp {c['ordinal']:3d}: regime {c['regime_k']:2d}  n={c['n_fam']:5d}  "
              f"D={c['D']:.3f} L={c['L']:.3f} T={c['T']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
