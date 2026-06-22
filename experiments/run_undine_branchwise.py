"""Branch-wise (per-species-branch) DTL rates + rooting for the BIG TREE
(This_study / "Undine C60", ~257 taxa, 15 candidate roots), driven by gpurec's
specieswise wave optimizer.

This is the big-tree analogue of ``run_williams_branchwise.py``. This_study
ships per-family **UFBoot bootstrap tree samples** (``3_UFBOOTs/ufboot_for_alerax/
*.ufboot``) rather than pre-built CCPs. We convert each ufboot to a classic
ALEobserve ``.ale`` CCP once (``ALEobserve <ufboot>`` via the boussau/alesuite
Singularity image -- exactly what AleRax consumes), then load it through gpurec's
validated ``.ale`` pipeline (``io/ale.py``). So the family path is IDENTICAL to
Williams (``_load_species_helpers`` + ``_load_families``); only the tree paths and
the ccp directory differ. Everything downstream (wave layout, fraction-missing,
Brownian prior, origination, the specieswise wave optimize, the output format) is
shared with the Williams driver and imported from it, so the two stay in
lock-step.

Output matches AleRax ``model_parameters.txt`` (``# node D L T [O]``) + a JSON
sidecar with the prior-free ``data_log_likelihood_ln`` used for the rooting test.

Run on an A100 (CUDA required for optimize):

    PYTHONPATH=. python experiments/run_undine_branchwise.py --root Eury \
        --fm-mode e-only --origination optimize --prior brownian --brownian-sigma 1.0

Data-wiring sanity (load a few .ale, check species mapping; CPU only):

    PYTHONPATH=. python experiments/run_undine_branchwise.py --root Eury --preflight
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# Reuse the Williams driver's PURE compute helpers (stable; importing does not
# run anything heavy and does not touch the in-flight Williams runs).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _build_wave_layout,
    _sp_helpers_for_uniform,
    _build_leaf_E,
    _parse_fraction_missing,
    _load_species_helpers,
    _load_families,
)

# ALEobserve writes <ufboot>.ale next to the input, so the CCPs live in the
# ufboot dir by default.
DEFAULT_CCP_SUBDIR = "3_UFBOOTs/ufboot_for_alerax"

DEFAULT_DATA_DIR = (
    "/work/SzollosiU/gergely-szollosi/williams_run/data/"
    "3_Reconciliation/This_study"
)
# The 15 Undine candidate roots (4_species_tree/Undine_C60_<ROOT>root_short_name.nw).
KNOWN_ROOTS = (
    "Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury", "HalobacThermopl",
    "Kor", "MHH", "Micra5", "MicraDia", "TACA", "TackA", "TAC", "UndineClu2",
)


# ── paths ─────────────────────────────────────────────────────────────────────
def _tree_path(data_dir: Path, root: str) -> Path:
    return data_dir / "4_species_tree" / f"Undine_C60_{root}root_short_name.nw"


def _ale_paths(data_dir: Path, ccp_subdir: str):
    """The ALEobserve .ale CCPs (one per family, produced from the ufboots)."""
    return sorted(
        p for p in glob.glob(str(data_dir / ccp_subdir / "*.ale"))
        if not Path(p).name.startswith("._")
    )


def _fraction_missing_path(data_dir: Path) -> Path:
    # This_study may or may not ship a fraction_missing file; tolerate absence.
    return data_dir / "fraction_missing"


# ── family loading via the validated .ale pipeline (ALEobserve output) ─────────
def _load_species_and_families(tree_path, ale_paths, *, min_species, dtype,
                               limit=0):
    """Load species helpers from the Undine tree + per-family CCPs from .ale.

    Identical machinery to the Williams driver (``_load_species_helpers`` +
    ``_load_families`` -> ``build_family_from_ale``). Returns
    (species_helpers, families, stats).
    """
    t0 = time.time()
    species_helpers = _load_species_helpers(str(tree_path))
    sp_name_to_idx = species_helpers["species_name_to_index"]
    families, fstats = _load_families(
        ale_paths, sp_name_to_idx,
        min_species=min_species, dtype=dtype, limit=limit,
    )
    stats = dict(fstats)
    stats["parse_seconds"] = time.time() - t0
    return species_helpers, families, stats


# ── preflight (CPU; load a few .ale, check species mapping) ────────────────────
def _preflight(args, data_dir: Path):
    tree_path = _tree_path(data_dir, args.root)
    if not tree_path.exists():
        raise SystemExit(f"species tree not found: {tree_path}")
    ale_paths = _ale_paths(data_dir, args.ccp_dir)
    if not ale_paths:
        raise SystemExit(f"no .ale files under {data_dir / args.ccp_dir} "
                         f"(run ALEobserve on the ufboots first)")
    print(f"[preflight] root={args.root}")
    print(f"  tree : {tree_path}")
    print(f"  .ale : {len(ale_paths)} families found under {args.ccp_dir}")
    n = min(args.families if args.families > 0 else 50, len(ale_paths))
    print(f"  loading first {n} families (species mapping check) ...", flush=True)
    sp, fams, stats = _load_species_and_families(
        tree_path, ale_paths[:n], min_species=args.min_species, dtype=torch.float64,
    )
    S = int(sp["S"])
    names = list(sp["names"])
    sP = sp["s_P_indexes"]
    internal = sP[sP < S].unique()
    n_leaves = S - int(internal.numel())
    print(f"  species tree: S={S} nodes ({n_leaves} leaves)")
    print(f"  tree leaf labels (first 8): {sorted(names)[:8]}")
    print(f"  families kept={stats['kept']} dropped(<{args.min_species} sp)="
          f"{stats.get('dropped_min_species', 0)}  load={stats['parse_seconds']:.2f}s")
    if stats.get("missing_species_counter"):
        print(f"  [warn] species in .ale absent from tree (top): "
              f"{stats['missing_species_counter'].most_common(8)}")
    else:
        print(f"  [OK] all sampled .ale species present in the tree.")
    fm = _fraction_missing_path(data_dir)
    print(f"  fraction_missing: {'FOUND ' + str(fm) if fm.exists() else 'not present'}")
    print("PREFLIGHT ok. Re-run without --preflight on an A100 to optimize.")
    return 0


# ── full run ───────────────────────────────────────────────────────────────────
def _run(args, data_dir: Path):
    tree_path = _tree_path(data_dir, args.root)
    if args.root not in KNOWN_ROOTS:
        print(f"[warn] root {args.root!r} not in known Undine roots {KNOWN_ROOTS}")
    if not tree_path.exists():
        raise SystemExit(f"species tree not found: {tree_path}")
    ale_paths = _ale_paths(data_dir, args.ccp_dir)
    if not ale_paths:
        raise SystemExit(f"no .ale files under {data_dir / args.ccp_dir} "
                         f"(run ALEobserve on the ufboots first)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for optimization (run on the A100). "
                         "Use --preflight for the CPU data-wiring check.")
    # Multi-GPU DDP: under torchrun (RANK/WORLD_SIZE set) init NCCL + pin the device;
    # otherwise world=1 and this is a no-op (byte-identical single-GPU path).
    from gpurec import distributed as ddp
    rank, world, local, device = ddp.maybe_init_distributed(device_pref="cuda")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if world > 1:
        print(f"[DDP] rank {rank}/{world} (local {local}) on {device}", flush=True)

    from gpurec.optimization.wave_optimizer import optimize_theta_wave
    from gpurec.core.tree_prior import species_parent_index

    print(f"[1/5] Loading {len(ale_paths)} .ale families on tree "
          f"{tree_path.name} (min-species={args.min_species}) ...", flush=True)
    species_helpers, families, stats = _load_species_and_families(
        tree_path, ale_paths, min_species=args.min_species, dtype=dtype,
        limit=args.families,
    )
    S = int(species_helpers["S"])
    names = list(species_helpers["names"])
    sp_name_to_idx = species_helpers["species_name_to_index"]
    print(f"      S={S} species nodes;  families kept={stats['kept']} "
          f"dropped(<{args.min_species} sp)={stats.get('dropped_min_species', 0)}  "
          f"load={stats['parse_seconds']:.1f}s", flush=True)
    if not families:
        raise SystemExit("no usable families after filtering; aborting.")

    # DDP: each rank fits on its family SHARD. The loss is Sum_families, E is
    # family-independent (recomputed per rank), and the E-adjoint solve is linear
    # with a shared operator, so the optimizer's all_reduce(SUM) of nll+grad makes
    # sum-of-shards == whole (validated: Part C NCCL 9e-13). World=1 -> no-op.
    fam_use = ddp.shard_families(families, rank, world) if world > 1 else families
    if world > 1:
        print(f"[DDP] rank {rank}: {len(fam_use)}/{len(families)} families this shard",
              flush=True)

    print(f"[2/5] Building cross-family wave layout for {len(fam_use)} "
          f"families ...", flush=True)
    t0 = time.time()
    wave_layout, root_clade_ids = _build_wave_layout(fam_use, device, dtype)
    sp_helpers_gpu, _ = _sp_helpers_for_uniform(species_helpers, device, dtype)
    unnorm_row_max = torch.log2(
        species_helpers["Recipients_mat"]
    ).max(dim=-1).values.to(device=device, dtype=dtype)
    print(f"      wave layout built in {time.time() - t0:.1f}s", flush=True)

    # 3. fraction-missing (decoupled E vs Pi leaf boundary), same as Williams.
    fm_path = _fraction_missing_path(data_dir)
    leaf_E = None
    leaf_obs_log = None
    fm_info = {"enabled": False, "fm_mode": args.fm_mode}
    if args.fm_mode == "off":
        print("[3/5] fraction-missing DISABLED (fm-mode=off)", flush=True)
    elif not fm_path.exists():
        print(f"[3/5] [warn] fraction_missing not found ({fm_path}); proceeding "
              f"WITHOUT it (fm-mode={args.fm_mode}).", flush=True)
        fm_info = {"enabled": False, "fm_mode": args.fm_mode,
                   "note": "fraction_missing file not found"}
    else:
        fm, n_set, fm_skipped = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
        leaf_E_cpu, _ = _build_leaf_E(species_helpers, fm, S, dtype)
        leaf_E = leaf_E_cpu.to(device=device, dtype=dtype)
        n_missing = int(torch.isfinite(leaf_E).sum().item())
        leaf_obs_log = leaf_E if args.fm_mode == "both" else None
        fm_info = {"enabled": True, "fm_mode": args.fm_mode,
                   "applied_to_E": True, "applied_to_Pi": args.fm_mode == "both",
                   "rows_mapped": n_set, "leaves_with_fraction": n_missing,
                   "skipped_rows": fm_skipped}
        print(f"[3/5] fraction-missing ENABLED ({args.fm_mode}): {n_set} rows, "
              f"{n_missing} leaves with fraction>0", flush=True)

    # 4. Brownian prior + origination, same conventions as Williams.
    parent_index = species_parent_index(species_helpers).to(device)

    # CLADE-GROUPED DTL + structured origination (constrained channel-1 control;
    # the paper's per-clade branch-wise DTL_br + {root, 2 children, DPANN} O).
    group_index = None
    omega_group_index = None
    _alerax_theta_init = None       # AleRax-seeded theta [S,3] (--init-from-alerax)
    _alerax_omega_init = None       # AleRax-seeded omega [S]
    if args.clade_groups:
        # AleRax's EXACT grouping + (optional) init at AleRax's fitted rates. Tests
        # whether gpurec's optimizer stays in AleRax's (Eury) basin or descends to SGA.
        from clade_groups import (group_index_for_species_helpers,
                                  omega_group_index_for_species_helpers,
                                  branch_params_from_alerax)
        cg_dir = Path(args.clade_groups)
        cg_tree = cg_dir / "species_trees" / "starting_species_tree.newick"
        cg_mp = cg_dir / "model_parameters" / "model_parameters.txt"
        if not cg_mp.exists():
            raise SystemExit(f"--clade-groups: missing {cg_mp}")
        cg_tree = str(cg_tree) if cg_tree.exists() else str(tree_path)
        group_index, _cat_rates, _cat_orig, _cgd = group_index_for_species_helpers(
            species_helpers, cg_tree, str(cg_mp))
        if _cgd["gpurec_unmapped"]:
            raise SystemExit(f"--clade-groups: {len(_cgd['gpurec_unmapped'])} unmapped "
                             f"nodes (tree mismatch): {_cgd['gpurec_unmapped'][:4]}")
        group_index = group_index.to(device)
        n_dtl = int(group_index.max().item()) + 1
        n_o = 0
        if args.origination == "optimize" and not args.free_origination:
            omega_group_index, n_o = omega_group_index_for_species_helpers(
                species_helpers, cg_tree, str(cg_mp))
            if omega_group_index is not None:
                omega_group_index = omega_group_index.to(device)
        elif args.origination == "optimize":
            n_o = -1  # FREE per-branch O (omega_group_index=None) -- needed for the
                      # root-mass / depth origination priors (which require free omega).
        if args.init_from_alerax:
            _rb, _ob, _ = branch_params_from_alerax(species_helpers, cg_tree, str(cg_mp))
            _alerax_theta_init = torch.log2(_rb.clamp_min(1e-10)).to(device=device, dtype=dtype)
            if _ob is not None:
                _alerax_omega_init = torch.log2(_ob.clamp_min(1e-12)).to(device=device, dtype=dtype)
        _odesc = "FREE per-branch" if n_o < 0 else f"{n_o} classes"
        print(f"      ALERAX-EXACT GROUPING: {n_dtl} DTL classes, O: {_odesc} "
              f"(from {cg_dir.name}); init_from_alerax={args.init_from_alerax}", flush=True)
    elif args.clade_groups_from_tree:
        from clade_groups import (tree_clade_group_index, tree_origination_group_index,
                                  BIGTREE_DTL_CLADES)
        _wl = None if args.all_named_clades else BIGTREE_DTL_CLADES
        group_index, _cg_labels = tree_clade_group_index(
            species_helpers, str(tree_path), S, clade_whitelist=_wl)
        group_index = group_index.to(device)
        n_dtl = int(group_index.max().item()) + 1
        n_o = 0
        if args.origination == "optimize":
            omega_group_index, n_o = tree_origination_group_index(
                species_helpers, str(tree_path), S)
            omega_group_index = omega_group_index.to(device)
        print(f"      CLADE-GROUPED DTL: {n_dtl} classes from named clades; "
              f"O: {n_o} classes (DPANN/Eury/TackA/rest, AleRax DTL_br1_O structure)",
              flush=True)
        if omega_group_index is not None:
            import collections as _coll
            print("      O class sizes (branches per origination class):",
                  dict(sorted(_coll.Counter(omega_group_index.tolist()).items())),
                  flush=True)
        if args.preflight_groups:
            import collections
            print("      DTL clade labels:", _cg_labels)
            if omega_group_index is not None:
                print("      O class sizes:", dict(sorted(
                    collections.Counter(omega_group_index.tolist()).items())))
            raise SystemExit("--preflight-groups: grouping built OK, exiting before fit")
    brownian_sigma = float(args.brownian_sigma) if args.prior == "brownian" else None
    brownian_root_sigma = args.brownian_root_sigma
    prior_info = {"prior": args.prior}
    if args.prior == "brownian":
        prior_info.update({"brownian_sigma": brownian_sigma,
                           "brownian_root_sigma": brownian_root_sigma,
                           "units": "log2"})

    # WARM-START from a previous fit's sidecar (for the annealing chain): seed theta
    # (and free omega) at the previous stage's MAP so each relaxed-lambda stage
    # continues from the last -> graduated optimization into the deep basin.
    _rates_theta_init = None
    _rates_omega_init = None
    if args.init_from_rates:
        _sc = json.load(open(args.init_from_rates))
        _rates_theta_init = torch.tensor(_sc["theta_log2"], dtype=dtype, device=device)
        _om = _sc.get("origination", {}).get("omega_log2")
        if _om is not None:
            _rates_omega_init = torch.tensor(_om, dtype=dtype, device=device)
        print(f"      WARM-START from {Path(args.init_from_rates).name} "
              f"(theta{'+omega' if _om is not None else ''})", flush=True)

    omega_init = None
    origination_l2 = 0.0
    origination_info = {"origination": args.origination}
    if args.origination == "optimize":
        omega_init = (_rates_omega_init if _rates_omega_init is not None
                      else _alerax_omega_init if _alerax_omega_init is not None
                      else torch.zeros(S, dtype=dtype, device=device))
        # STRUCTURED-O SEED (the principled basin-finding init): when no warm-start
        # omega is given, seed origination CONCENTRATED at the deep/root region --
        # omega = -depth*scale, so softmax(omega) peaks at depth 0 (the root) and
        # decays outward. This encodes the vertical-evolution / LACA hypothesis (genes
        # originate once, deep) as a STARTING POINT (not a destructive penalty), placing
        # the optimizer in the deep/Eury basin's attractor. scale=0 -> uniform (current).
        if (float(args.init_omega_depth_scale) != 0.0
                and _rates_omega_init is None and _alerax_omega_init is None):
            _par = parent_index.tolist()
            _dep = [0] * S
            for e in range(S):
                d, cur, seen = 0, e, 0
                while _par[cur] >= 0 and _par[cur] != cur and seen <= S:
                    d += 1; cur = _par[cur]; seen += 1
                _dep[e] = d
            omega_init = (-float(args.init_omega_depth_scale)
                          * torch.tensor(_dep, dtype=dtype, device=device))
            print(f"      STRUCTURED-O SEED: omega=-depth*{args.init_omega_depth_scale} "
                  f"(O concentrated at the root/deep region; depth 0..{max(_dep)})", flush=True)
        # p^O = softmax(omega) is a probability distribution -> L2 ridge on the
        # logits (shrink toward uniform), NOT a Brownian prior. 0 = free.
        origination_l2 = float(args.origination_l2)
        origination_info.update({
            "regularization": ("l2" if origination_l2 > 0 else "free"),
            "origination_l2": origination_l2})

    # Vertical-evolution origination prior: per-branch DEPTH = #edges from the
    # root (root=0); penalize below-root origination via --origination-depth-lambda.
    origination_depth = None
    if args.origination == "optimize" and args.origination_depth_lambda > 0:
        _par = parent_index.tolist()
        depth = [0] * S
        for e in range(S):
            d, cur, seen = 0, e, 0
            while _par[cur] >= 0 and seen <= S:
                d += 1; cur = _par[cur]; seen += 1
            depth[e] = d
        origination_depth = torch.tensor(depth, dtype=dtype, device=device)
        origination_info.update({
            "origination_depth_lambda": float(args.origination_depth_lambda),
            "depth_range": [min(depth), max(depth)]})
        print(f"      VERTICAL-EVOLUTION O prior: lambda={args.origination_depth_lambda} "
              f"(depth range {min(depth)}..{max(depth)})", flush=True)

    # Root branch index (parent<0 or self-loop) for the ROOT-MASS origination prior.
    origination_root_index = None
    if args.origination == "optimize" and args.origination_root_lambda > 0:
        _par = parent_index.tolist()
        _roots = [e for e in range(S) if _par[e] < 0 or _par[e] == e]
        if len(_roots) != 1:
            raise SystemExit(f"expected exactly 1 root branch, got {_roots}")
        origination_root_index = _roots[0]
        origination_info.update({
            "origination_root_lambda": float(args.origination_root_lambda),
            "origination_root_index": origination_root_index})
        print(f"      ROOT-MASS O prior: lambda={args.origination_root_lambda} "
              f"(penalize 1-p^O_root; root branch index {origination_root_index})",
              flush=True)

    # ANTI-CONCENTRATION origination barrier (the validated cure). pi target:
    # rho=0 -> uniform (pure anti-concentration); rho>0 -> root-vertical. Penalty
    # strength c with divergence --origination-barrier-kind (use 'simpson').
    origination_vertical_pi = None
    if args.origination == "optimize" and args.origination_dirichlet > 0:
        _par = parent_index.tolist()
        _depth = [0] * S
        for e in range(S):
            d, cur, seen = 0, e, 0
            while _par[cur] >= 0 and _par[cur] != cur and seen <= S:
                d += 1; cur = _par[cur]; seen += 1
            _depth[e] = d
        _dec = float(args.origination_vertical_decay)
        _rho = float(args.origination_vertical_rho)
        _v = torch.tensor([_dec ** (-d) for d in _depth], dtype=dtype, device=device)
        _v = _v / _v.sum()
        origination_vertical_pi = (1.0 - _rho) / S + _rho * _v          # [S], sum=1, >0
        origination_info.update({
            "origination_dirichlet_c": float(args.origination_dirichlet),
            "origination_barrier_kind": args.origination_barrier_kind,
            "origination_vertical_rho": _rho})
        print(f"      O BARRIER: kind={args.origination_barrier_kind} "
              f"c={args.origination_dirichlet} rho={_rho} "
              f"(pi_min={float(origination_vertical_pi.min()):.2e})", flush=True)

    if _rates_theta_init is not None:
        theta_init = _rates_theta_init
    elif _alerax_theta_init is not None:
        theta_init = _alerax_theta_init
    elif args.init_dlt:
        # GLOBAL-RATE WARM-UP: seed every branch at a single (D,L,T) (e.g. AleRax's
        # global-model rates) instead of uniform 0.1 -> tests whether good DTL
        # magnitudes drop the optimizer into the deep (Eury) basin. O still uniform.
        _dlt = [float(x) for x in args.init_dlt.split(",")]
        if len(_dlt) != 3:
            raise SystemExit("--init-dlt expects 'D,L,T' (3 comma-sep rates)")
        theta_init = torch.log2(torch.tensor(_dlt, dtype=dtype, device=device)
                                .clamp_min(1e-10)).repeat(S, 1)
        print(f"      GLOBAL-RATE init: D,L,T={_dlt} (uniform across branches)", flush=True)
    else:
        theta_init = math.log2(args.init_rate) * torch.ones(S, 3, dtype=dtype, device=device)

    print(f"[4/5] Optimizing specieswise theta [S={S},3]  "
          f"(optimizer={args.optimizer}, fm-mode={args.fm_mode}, "
          f"origination={args.origination}, prior={args.prior}, "
          f"steps={args.steps}) ...", flush=True)
    t0 = time.time()
    result = optimize_theta_wave(
        wave_layout=wave_layout,
        species_helpers=sp_helpers_gpu,
        root_clade_ids=root_clade_ids,
        unnorm_row_max=unnorm_row_max,
        theta_init=theta_init,
        steps=args.steps,
        optimizer=args.optimizer,
        specieswise=True,
        pibar_mode=args.pibar_mode,
        families=fam_use,
        family_batch_size=args.family_batch_size,
        device=device,
        dtype=dtype,
        leaf_E=leaf_E,
        leaf_obs_log=leaf_obs_log,
        verbose=(rank == 0),
        distributed=(world > 1),
        ddp_device=device,
        grad_reduction="sum",   # mini-batched L-BFGS needs Σ (not mean/n_batch) so the
                                # gradient is consistent with the (summed) NLL — else the
                                # line search stalls and omega freezes at uniform.
        brownian_sigma=brownian_sigma,
        dtl_tv_lambda=args.dtl_tv_lambda,
        dtl_tv_eps=args.dtl_tv_eps,
        brownian_root_sigma=brownian_root_sigma,
        parent_index=parent_index,
        origination=args.origination,
        omega_init=omega_init,
        origination_l2=origination_l2,
        origination_depth=origination_depth,
        origination_depth_lambda=args.origination_depth_lambda,
        origination_root_index=origination_root_index,
        origination_root_lambda=args.origination_root_lambda,
        origination_dirichlet_c=args.origination_dirichlet,
        origination_vertical_pi=origination_vertical_pi,
        origination_barrier_kind=args.origination_barrier_kind,
        group_index=group_index,
        omega_group_index=omega_group_index,
    )
    elapsed = time.time() - t0

    theta = result["theta"]
    rates = result["rates"]
    if args.origination == "optimize":
        origination_prob = result["origination"]
        omega = result["omega"]
        origination_info.update({"origination": "optimize",
                                 "omega_log2": omega.tolist(),
                                 "origination_prob": origination_prob.tolist()})
        if "origination_prior" in result:
            origination_info["prior"] = result["origination_prior"]
    else:
        origination_prob = torch.full((S,), 1.0 / S, dtype=torch.float64)
    nll = float(result["negative_log_likelihood"])
    logL = float(result["log_likelihood"])
    data_nll = float(result.get("data_negative_log_likelihood", nll))
    data_logL = -data_nll

    # DDP: the optimizer all-reduced nll+grad, so result is GLOBAL (all families);
    # only rank 0 reports + writes. Non-zero ranks have done their collective work.
    if rank != 0:
        if world > 1:
            ddp.cleanup()
        return 0

    print(f"\n{'=' * 60}")
    print(f"  DONE  root={args.root}  families={len(families)}  "
          f"time={elapsed:.1f}s")
    print(f"  NLL(log2)={nll:.4f}  data_logL(log2)={data_logL:.4f}  "
          f"data_logL(ln)={data_logL * math.log(2.0):.4f}")
    for col, lab in ((0, "D"), (1, "L"), (2, "T")):
        c = rates[:, col]
        print(f"  {lab}: min={float(c.min()):.6g} median={float(c.median()):.6g} "
              f"max={float(c.max()):.6g}")
    if args.origination == "optimize":
        o = origination_prob
        print(f"  O: min={float(o.min()):.6g} median={float(o.median()):.6g} "
              f"max={float(o.max()):.6g} (sum={float(o.sum()):.6g})")
    print(f"{'=' * 60}")

    # 5. write rates (AleRax-comparable) + JSON sidecar.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        if args.origination == "optimize":
            fh.write("# node D L T O\n")
            for s in range(S):
                fh.write(f"{names[s]} {float(rates[s,0]):.10g} {float(rates[s,1]):.10g} "
                         f"{float(rates[s,2]):.10g} {float(origination_prob[s]):.10g}\n")
        else:
            fh.write("# node D L T\n")
            for s in range(S):
                fh.write(f"{names[s]} {float(rates[s,0]):.10g} {float(rates[s,1]):.10g} "
                         f"{float(rates[s,2]):.10g}\n")
    print(f"  rates written -> {out_path}", flush=True)

    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    payload = {
        "command": " ".join(sys.argv), "dataset": "This_study/Undine_C60",
        "root": args.root, "species_tree": str(tree_path), "S": S, "names": names,
        "n_families_kept": len(families),
        "n_families_dropped_min_species": stats["dropped_min_species"],
        "min_species": args.min_species, "init_rate": args.init_rate,
        "optimizer": args.optimizer, "pibar_mode": args.pibar_mode,
        "steps": args.steps, "dtype": args.dtype, "fm_mode": args.fm_mode,
        "fraction_missing": fm_info, "prior": prior_info, "origination": origination_info,
        "negative_log_likelihood_log2": nll, "log_likelihood_log2": logL,
        "negative_log_likelihood_ln": nll * math.log(2.0),
        "data_log_likelihood_log2": data_logL,
        "data_log_likelihood_ln": data_logL * math.log(2.0),
        "data_negative_log_likelihood_log2": data_nll,
        "theta_log2": theta.tolist(), "rates": rates.tolist(),
        "elapsed_s": elapsed, "n_steps": len(result.get("history", [])),
    }
    with open(sidecar, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  sidecar written -> {sidecar}", flush=True)
    if world > 1:
        ddp.cleanup()
    return 0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Branch-wise DTL rates + rooting for the This_study/Undine "
                    "big tree from UFBoot samples (gpurec specieswise wave).")
    p.add_argument("--root", required=True, help=f"one of {KNOWN_ROOTS}")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--ccp-dir", default=DEFAULT_CCP_SUBDIR,
                   help="dir (under --data-dir) of ALEobserve .ale CCPs; "
                        f"default {DEFAULT_CCP_SUBDIR}")
    p.add_argument("--min-species", type=int, default=1)
    p.add_argument("--families", type=int, default=0,
                   help="limit #families (0=all; for quick tests)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "adam", "sgd"])
    p.add_argument("--pibar-mode", default="uniform", choices=["uniform", "dense", "topk"])
    p.add_argument("--init-rate", type=float, default=0.1)
    p.add_argument("--init-dlt", default=None,
                   help="GLOBAL-RATE warm-up: 'D,L,T' to seed every branch at (instead "
                        "of uniform --init-rate). Basin-finding: does a good global DTL "
                        "init reach the deep/Eury basin? e.g. '0.07,0.30,0.17'.")
    p.add_argument("--init-from-rates", default=None,
                   help="Warm-start theta (+free omega) from a previous fit's "
                        "*.rates.txt.json sidecar. Used to chain the ANNEALING stages "
                        "(each relaxed-lambda stage continues from the previous MAP).")
    p.add_argument("--init-omega-depth-scale", type=float, default=0.0,
                   help="STRUCTURED-O SEED: init free omega=-depth*scale so origination "
                        "starts CONCENTRATED at the deep/root region (vertical/LACA prior "
                        "as an init). Places the optimizer in the deep/Eury basin attractor. "
                        "0=uniform. Multi-start over a few scales + pick the best logL.")
    p.add_argument("--fm-mode", default="off", choices=["both", "e-only", "off"])
    p.add_argument("--no-fraction-missing", action="store_true")
    p.add_argument("--origination", default="uniform", choices=["uniform", "optimize"])
    p.add_argument("--origination-l2", type=float, default=0.0,
                   help="L2/ridge on origination logits (lambda*sum(omega^2), "
                        "shrink p^O toward uniform). 0=free. p^O is a softmax "
                        "distribution, so L2 is correct (not a Brownian prior).")
    p.add_argument("--origination-depth-lambda", type=float, default=0.0,
                   help="VERTICAL-EVOLUTION origination prior: penalty "
                        "lambda*E_{p^O}[depth] where depth=#edges below the root "
                        "(root=0). Pulls origination mass to the root, taxing the "
                        "below-root loss-saving (small-genome) artefact. 0=off.")
    p.add_argument("--origination-root-lambda", type=float, default=0.0,
                   help="ROOT-MASS origination prior: penalty lambda*(1 - p^O_root). "
                        "Flat tax on all non-root mass (depth-agnostic) -- rewards "
                        "root origination without taxing the deep tail's shape. Use "
                        "INSTEAD of --origination-depth-lambda. 0=off.")
    p.add_argument("--origination-dirichlet", type=float, default=0.0,
                   help="ANTI-CONCENTRATION origination barrier strength c. Forbids the "
                        "degenerate single-branch origination vertex (reroot degeneracy). "
                        "0=off. Divergence set by --origination-barrier-kind.")
    p.add_argument("--origination-barrier-kind", type=str, default="meanlog",
                   choices=["meanlog", "simpson", "renyi2"],
                   help="Barrier divergence. 'simpson' (c*sum p^O^2, PEAK-weighted, "
                        "tolerates structured zeros -- the validated barrier); 'meanlog' "
                        "(reverse KL, zero-weighted); 'renyi2' (c*log2 sum p^O^2).")
    p.add_argument("--origination-vertical-rho", type=float, default=0.0,
                   help="Verticality of the barrier target pi=(1-rho)/S+rho*decay^-depth/Z. "
                        "0 (default) = uniform target (pure anti-concentration).")
    p.add_argument("--origination-vertical-decay", type=float, default=2.0,
                   help="Geometric decay base of the vertical target with depth (rho>0).")
    p.add_argument("--clade-groups-from-tree", action="store_true",
                   help="CONSTRAINED channel-1: clade-grouped DTL from the labeled "
                        "species tree (each branch -> smallest enclosing NAMED clade; "
                        "the paper's per-clade branch-wise DTL_br) + structured O "
                        "{root, 2 children, DPANN, rest}. No AleRax output needed.")
    p.add_argument("--preflight-groups", action="store_true",
                   help="With --clade-groups-from-tree: build + print the grouping "
                        "(DTL labels, O class sizes) and exit before fitting.")
    p.add_argument("--all-named-clades", action="store_true",
                   help="Use ALL named internal clades (~30) instead of the paper's "
                        "explicit DTL_br2 whitelist (22). Richer DTL_br model.")
    p.add_argument("--clade-groups", default=None,
                   help="AleRax's EXACT grouping: a per-root dir with "
                        "species_trees/starting_species_tree.newick + "
                        "model_parameters/model_parameters.txt (group branches by "
                        "AleRax's distinct rate tuples). Use INSTEAD of "
                        "--clade-groups-from-tree to match AleRax's parametrization.")
    p.add_argument("--init-from-alerax", action="store_true",
                   help="With --clade-groups: seed theta (and omega) at AleRax's "
                        "fitted per-branch rates instead of uniform -> tests whether "
                        "gpurec stays in AleRax's (Eury) basin or descends to SGA.")
    p.add_argument("--free-origination", action="store_true",
                   help="With --clade-groups: keep FREE per-branch omega (do NOT group "
                        "O into AleRax's O categories). Required for the root-mass / "
                        "depth origination priors, which need free omega.")
    p.add_argument("--family-batch-size", type=int, default=0,
                   help="mini-batch families for forward/backward to bound GPU "
                        "memory (0=all at once). Use a few hundred for the big tree.")
    p.add_argument("--prior", default="none", choices=["none", "brownian"])
    p.add_argument("--brownian-sigma", type=float, default=1.0)
    p.add_argument("--dtl-tv-lambda", type=float, default=0.0,
                   help="FUSED-LASSO / total-variation prior on per-branch DTL rates "
                        "(smoothed-L1 on parent-child log2-rate diffs) -> PIECEWISE-CONSTANT "
                        "rates = data-driven clade grouping (SOTA branchwise w/o overparam). "
                        "0=off. Use WITHOUT --clade-groups (full per-branch theta).")
    p.add_argument("--dtl-tv-eps", type=float, default=1e-3,
                   help="pseudo-Huber smoothing scale (log2 units) for --dtl-tv-lambda.")
    p.add_argument("--brownian-root-sigma", type=float, default=5.0)
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--out", default=None)
    p.add_argument("--preflight", action="store_true",
                   help="load a sample of .ale + check species mapping, then exit (CPU)")
    args = p.parse_args(argv)
    if args.no_fraction_missing:
        args.fm_mode = "off"
    if args.out is None:
        args.out = f"results/undine_{args.root}_branchwise.rates.txt"
    return args


def main(argv=None):
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    if args.preflight:
        return _preflight(args, data_dir)
    return _run(args, data_dir)


if __name__ == "__main__":
    sys.exit(main())
