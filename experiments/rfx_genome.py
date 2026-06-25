"""gamma-weighted ancestral GENOME read-out under the per-family RANDOM-EFFECTS model.

The rooting EM (em_origination.rooting_rfx) saves, per root, rfx_<root>.npz with:
  gamma[K,F]  -- each family's posterior over the K rate regimes (the random effect)
  log_pO[S]   -- the ONE shared origination profile (root-large, 511 ~ 0)
  dlts[K,3]   -- the MIX_GRID rate regimes (D,L,T)
  rb, labels, label_nodes

The expected genome at an internal node is
    genome[node] = Σ_f Σ_k gamma[k,f] * E[ copies_f[node] | regime k, shared p^O ]
so each family must be sampled UNDER EACH (active) regime's rates with the shared p^O,
then gamma-weighted. A single mean-rate sample would discard the random effect (esp. the
high-transfer families that explain non-Eury patchiness by HGT, not by 511-origination).

This script has three modes (driven by slurm/rfx_genome_saion.sbatch):
  --mode sidecar  : write a per_node_copies sidecar for ONE regime k = uniform (D_k,L_k,T_k)
                    broadcast to all S branches + origination_prob = shared p^O. Exits 2
                    (SKIP) if regime k carries < --weight-thresh of the mixing weight, so the
                    array can cover all 50 regimes and the inactive ones cost nothing.
  --mode combine  : load gamma + every per-regime per_node_copies .npz (presence/copies [F,S])
                    and form the gamma-weighted genome (presence AND copies) at every node;
                    report root / 511 / DPANN / named clades vs the paper anchors.
"""
from __future__ import annotations
import argparse, glob, json, math, re, sys
from pathlib import Path
import numpy as np

# paper 4_Ancestral_reconstruction anchors (Euryarchaeota root) -- presence(=genome) / copies
PAPER = {"LACA(root)": (1331, 1456), "node511": (1368, 1493), "DPANN": (937, 1066)}


def _active_weights(rfx):
    gamma = rfx["gamma"].astype(np.float64)            # [K,F]
    w = gamma.mean(1)                                  # mixing weight per regime
    return gamma, w


def mode_sidecar(args):
    rfx = np.load(args.rfx_npz, allow_pickle=True)
    gamma, w = _active_weights(rfx)
    K = w.shape[0]; k = args.regime_idx
    if k >= K:
        print(f"  regime {k} >= K={K}; nothing to do"); return 0
    if w[k] < args.weight_thresh:
        print(f"  SKIP regime {k}: w={w[k]:.5f} < thresh {args.weight_thresh}")
        return 2                                       # sentinel: skip (no sampling)
    D, L, T = (float(x) for x in rfx["dlts"][k])
    log_pO = rfx["log_pO"].astype(np.float64)
    S = log_pO.shape[0]
    pO = np.exp2(log_pO); pO = pO / pO.sum()
    row = [math.log2(D), math.log2(L), math.log2(T)]
    sidecar = {"theta_log2": [row for _ in range(S)],
               "origination": {"origination_prob": pO.tolist()},
               "_rfx_regime": k, "_dlt": [D, L, T], "_w": float(w[k])}
    Path(args.out_sidecar).write_text(json.dumps(sidecar))
    print(f"  regime {k}: D,L,T={D:.3f},{L:.3f},{T:.3f}  w={w[k]:.4f}  -> {args.out_sidecar}")
    return 0


def mode_combine(args):
    rfx = np.load(args.rfx_npz, allow_pickle=True)
    gamma, w = _active_weights(rfx)                    # [K,F]
    K, F = gamma.shape
    rb = int(rfx["rb"])
    lab2n = {str(a): int(b) for a, b in zip(rfx["labels"], rfx["label_nodes"])}
    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no per-regime copies match {args.pattern}")
    # which regimes were actually sampled (filename ..._regime_<k>.npz) -> renorm gamma over them
    sampled = {}
    for fp in files:
        m = re.search(r"regime_(\d+)\.npz$", fp)
        if not m:
            continue
        sampled[int(m.group(1))] = fp
    ks = sorted(sampled)
    print(f"  combining {len(ks)} sampled regimes: {ks}")
    gA = gamma[ks, :]                                   # [M,F]
    gA = gA / np.clip(gA.sum(0, keepdims=True), 1e-30, None)   # renorm each family over sampled regimes
    S = None
    genome_pres = None; genome_cop = None
    nfam_check = None
    for j, k in enumerate(ks):
        d = np.load(sampled[k], allow_pickle=True)
        pres = d["presence"].astype(np.float64)        # [F,S]
        cop = d["copies"].astype(np.float64)           # [F,S]
        if pres.shape[0] != F:
            raise SystemExit(f"family-count mismatch: gamma F={F} vs {sampled[k]} F={pres.shape[0]}")
        if S is None:
            S = pres.shape[1]; genome_pres = np.zeros(S); genome_cop = np.zeros(S)
            names = list(d["names"]); nfam_check = pres.shape[0]
        g = gA[j][:, None]                             # [F,1]
        genome_pres += (g * pres).sum(0)               # Σ_f gamma'[k,f] * presence_k[f,:]
        genome_cop += (g * cop).sum(0)
    node511 = 511 if S > 511 else None
    dpann = lab2n.get("DPANN")
    rows = [("LACA(root)", rb), ("node511", node511), ("DPANN", dpann)]
    for lab in ("Undinarchaeota", "Asgard", "TackA", "TACK", "Eury", "Korarchaeota"):
        if lab in lab2n:
            rows.append((lab, lab2n[lab]))
    print("\n" + "=" * 72)
    print(f"RANDOM-EFFECTS gamma-weighted GENOME (root=Eury, {nfam_check} families, "
          f"{len(ks)} regimes)")
    print(f"  {'node':16s} {'idx':>5s} {'presence':>10s} {'copies':>9s}   {'paper pres/cop':>16s}")
    for lab, idx in rows:
        if idx is None:
            continue
        pa = PAPER.get(lab)
        pstr = f"{pa[0]}/{pa[1]}" if pa else ""
        print(f"  {lab:16s} {idx:5d} {genome_pres[idx]:10.1f} {genome_cop[idx]:9.1f}   {pstr:>16s}")
    if args.out:
        np.savez_compressed(args.out + ".npz", genome_presence=genome_pres,
                            genome_copies=genome_cop, names=np.array(names, dtype=object),
                            rb=rb, sampled_regimes=np.array(ks))
        with open(args.out + ".tsv", "w") as fh:
            fh.write("node\tname\tpresence\tcopies\n")
            for s in range(S):
                fh.write(f"{s}\t{names[s]}\t{genome_pres[s]:.3f}\t{genome_cop[s]:.3f}\n")
        print(f"\n  wrote {args.out}.npz / {args.out}.tsv")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["sidecar", "combine"])
    ap.add_argument("--rfx-npz", required=True)
    ap.add_argument("--regime-idx", type=int, default=0)
    ap.add_argument("--weight-thresh", type=float, default=0.004)
    ap.add_argument("--out-sidecar")
    ap.add_argument("--pattern", default="pnc_rfx_regime_*.npz")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.mode == "sidecar":
        return mode_sidecar(args)
    return mode_combine(args)


if __name__ == "__main__":
    sys.exit(main())
