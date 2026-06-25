"""Stage B warm start FROM Stage A: emit a per-branch theta init = the random-effect's
rate centroid (the w-weighted log2-mean over the MIX_GRID regimes in rfx_<root>.npz),
broadcast to all S branches. Stage B then REFINES this branchwise (staying in Stage A's
basin) instead of searching the multi-basin rate landscape from scratch. The shared p^O
is pinned separately via --origination-fixed-po-npz, so this writes theta only.
"""
import sys, json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfx-npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = np.load(args.rfx_npz, allow_pickle=True)
    gamma = d["gamma"].astype(np.float64)              # [K,F] regime posterior
    w = gamma.mean(1); w = w / w.sum()                 # [K] mixing weights
    dlts = d["dlts"].astype(np.float64)                # [K,3] (D,L,T) grid
    S = int(d["log_pO"].shape[0])
    logr = (w[:, None] * np.log2(dlts)).sum(0)         # [3] w-weighted log2-mean rate
    eff = np.exp2(logr)
    theta = [logr.tolist() for _ in range(S)]          # broadcast to every branch
    json.dump({"theta_log2": theta}, open(args.out, "w"))
    print(f"  Stage-A centroid init: D={eff[0]:.4f} L={eff[1]:.4f} T={eff[2]:.4f} "
          f"(w-weighted log-mean of {len(w)} regimes) -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
