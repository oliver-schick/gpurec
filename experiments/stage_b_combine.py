"""Stage B FULL combine: sum the per-component genome samples into the both-heterogeneous
ancestral genome. Each family was sampled at ITS component's branchwise theta_k[S,3] (+ shared
p^O), so summing per-family presence/copies across all components = the family-mixture-of-
branchwise genome (family het from the partition, branch het from each theta_k). Reports
root/511/DPANN vs the paper anchors and the rfx (global-rate) and branch-het control values.
"""
import sys, glob, argparse
from pathlib import Path
import numpy as np

# Eury-rooted node ids; paper presence anchors; rfx and branch-het-control baselines.
NODES = [(512, "LACA(root)", 1331, 1562), (511, "node511", 1368, 1384), (510, "DPANN", 937, 1057)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", default="comp_*_pnc.npz")
    ap.add_argument("--label", default="StageB-FULL")
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.dir) / args.pattern)))
    if not files:
        raise SystemExit(f"no component genomes match {args.pattern} in {args.dir}")
    gp = gc = None; nfam = 0; ncomp = 0
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        pres = d["presence"].astype(np.float64); cop = d["copies"].astype(np.float64)  # [F_sub,S]
        if gp is None:
            S = pres.shape[1]; gp = np.zeros(S); gc = np.zeros(S)
        gp += pres.sum(0); gc += cop.sum(0); nfam += pres.shape[0]; ncomp += 1
    print(f"\n=== {args.label} GENOME  ({ncomp} components, {nfam} families summed) ===")
    print(f"  {'node':12s} {'presence':>9s} {'copies':>8s}   {'paper':>6s} {'rfx':>6s} {'branchHet':>9s}")
    for n, lab, paper, rfx in NODES:
        bh = "-"
        print(f"  {lab:12s} {gp[n]:9.0f} {gc[n]:8.0f}   {paper:6d} {rfx:6d} {bh:>9s}")
    np.savez_compressed(str(Path(args.dir) / "stageB_full_genome.npz"),
                        genome_presence=gp, genome_copies=gc, n_families=nfam, n_components=ncomp)
    print(f"  wrote {Path(args.dir) / 'stageB_full_genome.npz'}")


if __name__ == "__main__":
    sys.exit(main())
