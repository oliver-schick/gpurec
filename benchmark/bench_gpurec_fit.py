#!/usr/bin/env python
"""Benchmark driver: gpurec DTL rate fit on Williams .ale, mode-selectable.

The bundled experiments/run_williams_branchwise.py and gpurec_fit.py both
hardcode specieswise=True. This thin driver adds --mode {global,specieswise}
and otherwise reuses their exact helpers + optimize_theta_wave, so the global
run matches AleRax's default (one shared D,L,T) and the specieswise run matches
what we already validated. Staged into <gpurec>/experiments/ (untracked) by the
kit and run with the repo venv; imports run_williams_branchwise for the helpers.

  python experiments/bench_gpurec_fit.py --mode global --root DPANN \
      --data-dir <Williams_et_al_2017> --families 200 --fm-mode e-only --out rates.txt
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402  (same helpers the validated run used)
    _resolve_paths, _load_species_helpers, _load_families, _build_wave_layout,
    _sp_helpers_for_uniform, _parse_fraction_missing, _build_leaf_E, KNOWN_ROOTS,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="gpurec rate fit (global or specieswise) on Williams .ale")
    ap.add_argument("--mode", choices=["global", "specieswise"], default="global")
    # Williams-style input: a data dir with rooted_phylogeny/<root> + ccps/.
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--root", default="DPANN")
    # OR explicit input (any rooted tree + a glob of .ale; e.g. Davin):
    ap.add_argument("--species-tree", default=None, help="rooted species tree (newick)")
    ap.add_argument("--ale-glob", default=None, help="glob for .ale files, e.g. '/…/ALEs/*.ale'")
    ap.add_argument("--fraction-missing", default=None, help="whitespace 'species fraction' file")
    ap.add_argument("--families", type=int, default=0, help="0 = all; else first N")
    ap.add_argument("--min-species", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "adam", "sgd"])
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--init-rate", type=float, default=0.1)
    ap.add_argument("--fm-mode", default="e-only", choices=["both", "e-only", "off"])
    ap.add_argument("--family-batch-size", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import glob as _glob
    if args.species_tree:                       # explicit input (Davin etc.)
        if not args.ale_glob:
            raise SystemExit("--species-tree requires --ale-glob")
        tree_path = Path(args.species_tree)
        ale_paths = sorted(p for p in _glob.glob(args.ale_glob)
                           if not Path(p).name.startswith("._"))
        fm_path = Path(args.fraction_missing) if args.fraction_missing else Path("/nonexistent")
    else:                                       # Williams data-dir/root layout
        if not args.data_dir:
            raise SystemExit("need either --data-dir (+ --root) or --species-tree (+ --ale-glob)")
        data_dir = Path(args.data_dir)
        if args.root not in KNOWN_ROOTS:
            print(f"[warn] root {args.root!r} not in {KNOWN_ROOTS}", flush=True)
        tree_path, ale_paths, fm_path = _resolve_paths(data_dir, args.root, None)
    if not tree_path.exists():
        raise SystemExit(f"species tree not found: {tree_path}")
    if not ale_paths:
        raise SystemExit(f"no .ale files found (glob/data-dir empty)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (run on an A100).")
    device = torch.device("cuda")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    specieswise = (args.mode == "specieswise")

    from gpurec.optimization.wave_optimizer import optimize_theta_wave

    print(f"[1/4] species tree {tree_path.name}", flush=True)
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"]); names = list(sp["names"]); name2idx = sp["species_name_to_index"]

    print(f"[2/4] loading .ale families (min-species={args.min_species}) ...", flush=True)
    families, stats = _load_families(ale_paths, name2idx, min_species=args.min_species,
                                     dtype=dtype, limit=args.families)
    print(f"      kept {stats['kept']} / considered {stats['n_considered']}", flush=True)
    if not families:
        raise SystemExit("no usable families after filtering")

    print(f"[3/4] cross-family wave layout for {len(families)} families ...", flush=True)
    wave_layout, root_clade_ids = _build_wave_layout(families, device, dtype)
    sp_gpu, _aT = _sp_helpers_for_uniform(sp, device, dtype)
    unnorm_row_max = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)

    # fraction-missing: e-only matches AleRax (extinction term only).
    leaf_E = None; leaf_obs_log = None; fm_used = False
    if args.fm_mode != "off" and fm_path.exists():
        fm, _n, _sk = _parse_fraction_missing(fm_path, name2idx, S)
        leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
        leaf_obs_log = leaf_E if args.fm_mode == "both" else None
        fm_used = True

    init_log2 = math.log2(args.init_rate)
    theta_init = init_log2 * (torch.ones(S, 3, dtype=dtype, device=device) if specieswise
                              else torch.ones(3, dtype=dtype, device=device))

    print(f"[4/4] optimizing {args.mode} theta (steps={args.steps}, opt={args.optimizer}, "
          f"fm={args.fm_mode}) ...", flush=True)
    t0 = time.time()
    result = optimize_theta_wave(
        wave_layout=wave_layout, species_helpers=sp_gpu, root_clade_ids=root_clade_ids,
        unnorm_row_max=unnorm_row_max, theta_init=theta_init, steps=args.steps,
        optimizer=args.optimizer, specieswise=specieswise, pibar_mode="uniform",
        families=families, family_batch_size=args.family_batch_size,
        device=device, dtype=dtype, leaf_E=leaf_E, leaf_obs_log=leaf_obs_log, verbose=True,
    )
    elapsed = time.time() - t0

    rates = torch.as_tensor(result["rates"]).detach().cpu()
    nll = float(result["negative_log_likelihood"])
    logL = float(result.get("log_likelihood", -nll))
    print(f"\n  DONE  mode={args.mode}  root={args.root}  families={len(families)}  "
          f"time={elapsed:.1f}s  NLL(log2)={nll:.4f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("# node D L T\n")
        if specieswise:
            r = rates.reshape(S, 3)
            for s in range(S):
                fh.write(f"{names[s]} {float(r[s,0]):.10g} {float(r[s,1]):.10g} {float(r[s,2]):.10g}\n")
        else:
            r = rates.flatten()
            fh.write(f"GLOBAL {float(r[0]):.10g} {float(r[1]):.10g} {float(r[2]):.10g}\n")
    sidecar = out.with_suffix(out.suffix + ".json")
    json.dump({
        "command": " ".join(sys.argv), "mode": args.mode, "root": args.root, "S": S,
        "n_families_kept": len(families), "n_families_considered": stats["n_considered"],
        "fraction_missing_used": fm_used, "fm_mode": args.fm_mode,
        "negative_log_likelihood_log2": nll, "log_likelihood_log2": logL,
        "negative_log_likelihood_ln": nll * math.log(2.0),
        "rates": rates.tolist(), "names": names if specieswise else None,
        "elapsed_s": elapsed, "n_steps": len(result.get("history", [])),
    }, open(sidecar, "w"), indent=2)
    print(f"  rates -> {out}\n  sidecar -> {sidecar}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
