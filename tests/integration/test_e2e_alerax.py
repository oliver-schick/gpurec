#!/usr/bin/env python3
"""
End-to-end validation: simulate trees, run AleRax, compare gpurec likelihoods.

1. Simulate a 60-leaf species tree (birth=1, death=0) using rustree
2. Simulate 100 gene trees (d=0.05, t=0.05, l=0.05)
3. Run AleRax with MPI to optimize DTL parameters
4. Run gpurec with AleRax's inferred parameters
5. Compare per-family likelihoods
"""

import csv
import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import torch

# Ensure project root is on path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# rustree was removed in the Rust-free build (it provided the tree simulator
# this end-to-end test uses to generate data). Skip cleanly when it is absent.
rustree = pytest.importorskip(
    "rustree",
    reason="rustree removed in the Rust-free build (was used for tree simulation)",
)
from gpurec.core.preprocess_cpp import _load_extension as _load_cpp_ext
from gpurec.core.extract_parameters import extract_parameters
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.legacy import Pi_fixed_point

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SPECIES = 60
N_GENES = 100
SEED = 42
LAMBDA_BIRTH = 1.0
MU_DEATH = 0.0
LAMBDA_D = 0.05
LAMBDA_T = 0.05
LAMBDA_L = 0.05

ALERAX_BINARY = shutil.which("alerax") or str(
    PROJECT_ROOT / "extra" / "AleRax_modified" / "build" / "bin" / "alerax"
)
MPI_BINARY = shutil.which("mpiexec") or shutil.which("mpirun")
N_MPI_PROCS = 4

# Tolerance for likelihood comparison (nats)
TOLERANCE = 1e-3


def simulate_trees(out_dir: pathlib.Path):
    """Simulate species tree and gene trees, write newick files + families.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating species tree ({N_SPECIES} leaves, birth={LAMBDA_BIRTH}, death={MU_DEATH})...")
    sp = rustree.simulate_species_tree(
        n=N_SPECIES, lambda_=LAMBDA_BIRTH, mu=MU_DEATH, seed=SEED
    )
    sp.save_newick(str(out_dir / "sp.nwk"))
    print(f"  Saved to {out_dir / 'sp.nwk'}")

    print(f"Simulating {N_GENES} gene trees (d={LAMBDA_D}, t={LAMBDA_T}, l={LAMBDA_L})...")
    families_lines = ["[FAMILIES]"]
    gene_paths = []
    rows = []

    for i, gt in enumerate(sp.simulate_dtl_per_species_iter(
        lambda_d=LAMBDA_D,
        lambda_t=LAMBDA_T,
        lambda_l=LAMBDA_L,
        require_extant=True,
        n=N_GENES,
        seed=SEED + 1,
    )):
        g_name = f"g_{i:04d}.nwk"
        # Drop gene copies at internal species nodes so AleRax can map all
        # gene leaves to species tree leaves.
        gt_ext = gt.sample_extant()
        gt_ext.save_newick(str(out_dir / g_name))
        gene_paths.append(str(out_dir / g_name))

        families_lines.append(f"- family_{i:04d}")
        families_lines.append(f"gene_tree = {g_name}")

        events = gt.count_events()
        rows.append({"family": f"family_{i:04d}",
                      "n_leaves": int(events["leaves"]),
                      "speciations": int(events["speciations"]),
                      "duplications": int(events["duplications"]),
                      "transfers": int(events["transfers"]),
                      "losses": int(events["losses"])})

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{N_GENES}")

    fam_path = out_dir / "families.txt"
    fam_path.write_text("\n".join(families_lines) + "\n")
    print(f"  Wrote {fam_path}")

    # Stats
    import statistics
    print(f"\n{'Field':14s}  {'mean':>8s}  {'median':>6s}  {'min':>5s}  {'max':>6s}")
    for field in ("n_leaves", "speciations", "duplications", "transfers", "losses"):
        vals = [r[field] for r in rows]
        if len(vals) > 1:
            print(f"{field:14s}  {statistics.mean(vals):8.1f}  {statistics.median(vals):6.0f}"
                  f"  {min(vals):5d}  {max(vals):6d}")

    return str(out_dir / "sp.nwk"), gene_paths, str(fam_path)


def run_alerax(sp_path: str, families_path: str, output_dir: str):
    """Run AleRax with MPI, return output directory."""
    if not pathlib.Path(ALERAX_BINARY).exists():
        raise FileNotFoundError(f"AleRax binary not found at {ALERAX_BINARY}")

    cmd = []
    if MPI_BINARY:
        cmd += [MPI_BINARY, "-np", str(N_MPI_PROCS), "--oversubscribe"]

    cmd += [
        ALERAX_BINARY,
        "-f", families_path,
        "-s", sp_path,
        "-p", output_dir,
        "--gene-tree-samples", "0",
        "--species-tree-search", "SKIP",
        "--rec-model", "UndatedDTL",
        "--model-parametrization", "GLOBAL",
    ]

    # Run from the directory containing families.txt so relative gene_tree paths resolve
    cwd = str(pathlib.Path(families_path).parent)
    print(f"\nRunning AleRax (cwd={cwd}): {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=cwd)

    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:] if result.stdout else "(empty)")
        print("STDERR:", result.stderr[-2000:] if result.stderr else "(empty)")
        raise RuntimeError(f"AleRax failed with return code {result.returncode}")

    print("  AleRax completed successfully.")
    return output_dir


def parse_alerax_parameters(output_dir: str) -> dict:
    """Parse DTL parameters from AleRax model_parameters.txt.

    Returns dict with 'D', 'L', 'T' (rates, not log).
    With GLOBAL parametrization all branches have the same values.
    """
    params_file = pathlib.Path(output_dir) / "model_parameters" / "model_parameters.txt"
    if not params_file.exists():
        raise FileNotFoundError(f"Not found: {params_file}")

    with open(params_file) as f:
        lines = f.readlines()

    # Format: "# node D L T" header, then "node_name D L T" per line
    for line in lines[1:]:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                D = float(parts[1])
                L = float(parts[2])
                T = float(parts[3])
                return {'D': D, 'L': L, 'T': T}

    raise ValueError(f"No valid parameter lines in {params_file}")


def parse_alerax_likelihoods(output_dir: str) -> dict:
    """Parse per-family likelihoods from AleRax output.

    Returns dict mapping family_name -> log-likelihood (nats).
    """
    lik_file = pathlib.Path(output_dir) / "per_fam_likelihoods.txt"
    if not lik_file.exists():
        raise FileNotFoundError(f"Not found: {lik_file}")

    result = {}
    with open(lik_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                family_name = parts[0]
                likelihood = float(parts[1])
                result[family_name] = likelihood

    return result


def run_gpurec(sp_path: str, gene_paths: list, D: float, T: float, L: float,
               device="cpu", dtype=torch.float64) -> list:
    """Run gpurec with given DTL parameters.

    Returns list of per-family log-likelihoods in log2 units (bits).
    """
    _INV_LN2 = 1.0 / math.log(2.0)
    device = torch.device(device)

    print(f"\nRunning gpurec (D={D:.6e}, T={T:.6e}, L={L:.6e})...")

    ext = _load_cpp_ext()

    # theta = ln([D, L, T])  (extract_parameters expects natural-log logits)
    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))

    likelihoods = []
    for i, gpath in enumerate(gene_paths):
        raw = ext.preprocess(sp_path, [gpath])
        species_raw = raw['species']
        ccp_raw = raw['ccp']

        # Build species_helpers
        species_helpers = {
            'S': int(species_raw['S']),
            'names': species_raw['names'],
            's_P_indexes': species_raw['s_P_indexes'].to(device=device),
            's_C12_indexes': species_raw['s_C12_indexes'].to(device=device),
            'Recipients_mat': species_raw['Recipients_mat'].to(dtype=dtype, device=device),
        }

        # Build ccp_helpers
        ccp_helpers = {
            'split_leftrights_sorted': ccp_raw['split_leftrights_sorted'].to(device=device),
            'log_split_probs_sorted': (ccp_raw['log_split_probs_sorted'].to(dtype=dtype, device=device) * _INV_LN2),
            'seg_parent_ids': ccp_raw['seg_parent_ids'].to(device=device),
            'ptr_ge2': ccp_raw['ptr_ge2'].to(device=device),
            'num_segs_ge2': int(ccp_raw['num_segs_ge2']),
            'num_segs_eq1': int(ccp_raw['num_segs_eq1']),
            'num_segs_eq0': int(ccp_raw['num_segs_eq0']),
            'stop_reduce_ptr_idx': int(ccp_raw['stop_reduce_ptr_idx']),
            'end_rows_ge2': int(ccp_raw['end_rows_ge2']),
            'C': int(ccp_raw['C']),
            'N_splits': int(ccp_raw['N_splits']),
        }

        root_clade_id = int(ccp_raw['root_clade_id'])
        leaf_row_index = raw['leaf_row_index'].to(torch.long).to(device)
        leaf_col_index = raw['leaf_col_index'].to(torch.long).to(device)

        tr_mat_unnorm = torch.log2(species_helpers['Recipients_mat'])

        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
            theta, tr_mat_unnorm, genewise=False, specieswise=False, pairwise=False)

        if max_transfer_mat.ndim == 2 and max_transfer_mat.shape[-1] == 1:
            max_transfer_vec = max_transfer_mat.squeeze(-1)
        else:
            max_transfer_vec = max_transfer_mat

        E_out = E_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            max_iters=2000, tolerance=1e-12,
            warm_start_E=None, dtype=dtype, device=device,
        )

        Pi_out = Pi_fixed_point(
            ccp_helpers=ccp_helpers,
            species_helpers=species_helpers,
            leaf_row_index=leaf_row_index,
            leaf_col_index=leaf_col_index,
            E=E_out['E'], Ebar=E_out['E_bar'],
            E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat_T=transfer_mat.transpose(-1, -2),
            max_transfer_mat=max_transfer_vec,
            max_iters=2000, tolerance=1e-12,
            warm_start_Pi=None, device=device, dtype=dtype,
        )

        logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_id)
        likelihoods.append(float(logL))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(gene_paths)}")

    print(f"  Done. Computed {len(likelihoods)} likelihoods.")
    return likelihoods


def compare_likelihoods(alerax_liks: dict, gpurec_liks_bits: list, gene_names: list,
                        output_csv: str | None = None):
    """Compare AleRax (nats) vs gpurec (bits) likelihoods.

    Converts gpurec from log2 to ln for comparison.
    Saves full results table to output_csv if provided.
    """
    LN2 = math.log(2.0)
    rows = []
    max_diff = 0.0
    max_diff_family = ""

    for i, name in enumerate(gene_names):
        alerax_val = alerax_liks.get(name)
        if alerax_val is None:
            continue

        gpurec_nats = -gpurec_liks_bits[i] * LN2
        diff = abs(gpurec_nats - alerax_val)
        rows.append({
            'family': name,
            'alerax_nats': alerax_val,
            'gpurec_nats': gpurec_nats,
            'abs_diff': diff,
            'status': 'OK' if diff < TOLERANCE else 'FAIL',
        })

        if diff > max_diff:
            max_diff = diff
            max_diff_family = name

    diffs = [r['abs_diff'] for r in rows]
    mean_diff = sum(diffs) / len(diffs) if diffs else float('inf')
    n_pass = sum(1 for d in diffs if d < TOLERANCE)
    n_fail = len(diffs) - n_pass

    # Print summary to stdout
    print(f"\nMean abs diff:  {mean_diff:.2e}")
    print(f"Max abs diff:   {max_diff:.2e}  ({max_diff_family})")
    print(f"Tolerance:      {TOLERANCE:.2e}")
    print(f"Pass: {n_pass}/{len(diffs)}, Fail: {n_fail}/{len(diffs)}")

    # Save full table to CSV
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['family', 'alerax_nats', 'gpurec_nats', 'abs_diff', 'status'])
            writer.writeheader()
            writer.writerows(rows)
            # Summary row
            total_alerax = sum(r['alerax_nats'] for r in rows)
            total_gpurec = sum(r['gpurec_nats'] for r in rows)
            writer.writerow({
                'family': 'TOTAL',
                'alerax_nats': total_alerax,
                'gpurec_nats': total_gpurec,
                'abs_diff': abs(total_gpurec - total_alerax),
                'status': '',
            })
        print(f"Results saved to {output_csv}")

    return n_fail == 0, mean_diff, max_diff


def main():
    with tempfile.TemporaryDirectory(prefix="gpurec_e2e_") as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        data_dir = tmpdir / "data"
        alerax_out = str(tmpdir / "alerax_output")

        # 1. Simulate trees
        sp_path, gene_paths, families_path = simulate_trees(data_dir)

        # 2. Run AleRax
        run_alerax(sp_path, families_path, alerax_out)

        # 3. Parse AleRax results
        params = parse_alerax_parameters(alerax_out)
        print(f"\nAleRax inferred parameters: D={params['D']:.6e}, L={params['L']:.6e}, T={params['T']:.6e}")

        alerax_liks = parse_alerax_likelihoods(alerax_out)
        print(f"AleRax computed {len(alerax_liks)} family likelihoods")

        # 4. Run gpurec with AleRax's parameters
        gpurec_liks = run_gpurec(
            sp_path, gene_paths,
            D=params['D'], T=params['T'], L=params['L'],
        )

        # 5. Compare and save results
        output_csv = str(PROJECT_ROOT / "tests" / "integration" / "likelihood_comparison.csv")
        gene_names = [f"family_{i:04d}" for i in range(N_GENES)]
        passed, mean_diff, max_diff = compare_likelihoods(
            alerax_liks, gpurec_liks, gene_names, output_csv=output_csv)

        if passed:
            print(f"\nVALIDATION PASSED: all {N_GENES} families within tolerance {TOLERANCE:.0e}")
        else:
            print(f"\nVALIDATION FAILED: some families exceed tolerance {TOLERANCE:.0e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
