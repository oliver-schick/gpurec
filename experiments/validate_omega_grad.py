"""Finite-difference validation of the recipient-omega gradient.

The transfer-to model sets transfer_mat_unnormalized[d,r] = omega_r at valid (non
-ancestor) recipients, -inf elsewhere. The efficient implicit backward now returns
dL/d(transfer_mat_unnormalized) (return_grad_tmu=True); chaining through the omega
broadcast gives dL/d omega. This script compares that analytic dL/d omega against
central finite differences on the small bundled test tree (dense pibar_mode, the
near-exact path). Run on a CUDA node (the Triton wave forward needs it).

  python experiments/validate_omega_grad.py
"""
from __future__ import annotations
import glob, math, sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "experiments"))

from gpurec.core.extract_parameters import extract_parameters               # noqa: E402
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood    # noqa: E402
from gpurec.core.forward import Pi_wave_forward                             # noqa: E402
from gpurec.core.scheduling import compute_clade_waves                      # noqa: E402
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout  # noqa: E402
from gpurec.optimization.implicit_grad import implicit_grad_loglik_vjp_wave  # noqa: E402

NEG = float("-inf")
_WDD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/Williams_et_al_2017")


def _load(device, dtype, n_fam=3):
    from run_williams_branchwise import _load_species_helpers, _load_families
    tree = sorted(glob.glob(str(_WDD / "rooted_phylogeny" / "*")))[0]
    sp = _load_species_helpers(tree); s2i = sp["species_name_to_index"]
    ale = sorted(p for p in glob.glob(str(_WDD / "ccps" / "*.ale")) if not Path(p).name.startswith("._"))
    fams, _ = _load_families(ale, s2i, min_species=1, dtype=dtype, limit=n_fam)
    sh = {"S": int(sp["S"]), "names": sp.get("names"),
          "s_P_indexes": sp["s_P_indexes"].to(device), "s_C12_indexes": sp["s_C12_indexes"].to(device),
          "Recipients_mat": sp["Recipients_mat"].to(dtype=dtype, device=device)}
    items = [{"ccp": f["ccp_helpers"], "leaf_row_index": f["leaf_row_index"],
              "leaf_col_index": f["leaf_col_index"], "root_clade_id": int(f["root_clade_id"])}
             for f in fams]
    return sh, items


def _build_tmu(omega, valid):
    """transfer_mat_unnormalized[d,r] = omega_r on valid recipients, -inf else (diff'able in omega)."""
    S = omega.shape[0]
    neg = torch.full((S, S), NEG, dtype=omega.dtype, device=omega.device)
    return torch.where(valid, omega.unsqueeze(0).expand(S, S), neg)


def _per_family_forward(theta, tmu, sh, items, device, dtype):
    """sum logL over families (dense pibar mode). Returns (logL, per-family fwd intermediates)."""
    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters(
        theta, tmu, genewise=False, specieswise=True, pairwise=False)
    if mt.ndim == 2:
        mt = mt.squeeze(-1)
    E_out = E_fixed_point(species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                          transfer_mat=transfer_mat, max_transfer_mat=mt, max_iters=4000,
                          tolerance=1e-12, warm_start_E=None, dtype=dtype, device=device,
                          pibar_mode="dense")
    total = 0.0; perfam = []
    for bi in items:
        sb = collate_gene_families([bi], dtype=dtype, device=device)
        w, p = compute_clade_waves(bi["ccp"]); cw = collate_wave([w], [0])
        wl = build_wave_layout(waves=cw, phases=p, ccp_helpers=sb["ccp"],
                               leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
                               root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)
        po = Pi_wave_forward(wave_layout=wl, species_helpers=sh, E=E_out["E"], Ebar=E_out["E_bar"],
                             E_s1=E_out["E_s1"], E_s2=E_out["E_s2"], log_pS=log_pS, log_pD=log_pD,
                             log_pL=log_pL, transfer_mat=transfer_mat, max_transfer_mat=mt,
                             device=device, dtype=dtype, pibar_mode="dense")
        total += compute_log_likelihood(po["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()
        perfam.append((wl, po))
    return total, (log_pS, log_pD, log_pL, transfer_mat, mt, E_out, perfam)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (Triton wave forward).")
    device = torch.device("cuda"); dtype = torch.float64
    sh, items = _load(device, dtype)
    S = sh["S"]
    valid = sh["Recipients_mat"] > 0
    tm_unnorm = torch.log2(sh["Recipients_mat"]).to(device=device, dtype=dtype)
    unnorm_row_max = tm_unnorm.max(dim=-1).values

    base = torch.log2(torch.tensor([0.08, 0.20, 0.10], dtype=dtype, device=device))
    theta = base.unsqueeze(0).expand(S, -1).contiguous()
    torch.manual_seed(0)
    omega = (0.3 * torch.randn(S, dtype=dtype, device=device))               # nontrivial recipient weights

    # analytic dL/d omega
    omega_req = omega.clone().requires_grad_(True)
    tmu = _build_tmu(omega_req, valid)
    logL, fwd = _per_family_forward(theta, tmu.detach(), sh, items, device, dtype)
    log_pS, log_pD, log_pL, transfer_mat, mt, E_out, perfam = fwd
    grad_tmu_total = torch.zeros(S, S, dtype=dtype, device=device)
    for (wl, po) in perfam:
        _, _, gtmu = implicit_grad_loglik_vjp_wave(
            wave_layout=wl, species_helpers=sh,
            Pi_star_wave=po["Pi_wave_ordered"], Pibar_star_wave=po["Pibar_wave_ordered"],
            E_star=E_out["E"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"], Ebar=E_out["E_bar"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, max_transfer_mat=mt,
            root_clade_ids_perm=wl["root_clade_ids"], theta=theta, unnorm_row_max=unnorm_row_max,
            specieswise=True, device=device, dtype=dtype, neumann_terms=4, use_pruning=False,
            cg_tol=1e-12, cg_maxiter=2000, pibar_mode="dense",
            transfer_mat=transfer_mat, transfer_mat_unnormalized=tmu.detach(),
            return_grad_tmu=True)
        grad_tmu_total = grad_tmu_total + gtmu
    domega = torch.autograd.grad(tmu, omega_req, grad_outputs=grad_tmu_total)[0].detach()

    # central FD on a sample of omega entries
    eps = 1e-5
    idxs = list(range(min(S, 12)))
    print(f"base logL = {logL:.8f}   S={S}")
    print(f"{'r':>3s} {'analytic':>14s} {'FD':>14s} {'rel_err':>10s}")
    maxrel = 0.0
    for r in idxs:
        op = omega.clone(); op[r] += eps
        lp, _ = _per_family_forward(theta, _build_tmu(op, valid), sh, items, device, dtype)
        om = omega.clone(); om[r] -= eps
        lm, _ = _per_family_forward(theta, _build_tmu(om, valid), sh, items, device, dtype)
        fd = (lp - lm) / (2 * eps); ana = float(domega[r])
        rel = abs(ana - fd) / max(abs(fd), 1e-7); maxrel = max(maxrel, rel)
        print(f"{r:3d} {ana:14.6e} {fd:14.6e} {rel:10.2e}")
    print(f"\nmax rel_err = {maxrel:.3e}   {'PASS' if maxrel < 2e-3 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
