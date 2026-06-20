"""Barrier-CONSISTENT empirical-Bayes rooting on the Dirichlet-barrier (cmlD0) MAPs.

The plain Laplace evidence uses a tau-ridge prior, so it just rewards data fit ->
prefers WEAK barrier (more degenerate p^O). Fix: use the SAME Dirichlet barrier the
MAP was fit with as the prior. Dominant change = the prior log-density:

    log Z_barrier = data_logL_ln  +  ln2 * c * sum_e pi_e log2 p^O*_e   +  occam
                    \___sidecar__/   \______barrier log-prior (nats)____/  \_EBev_/

with pi uniform over S branches (rho=0 runs). This PENALIZES concentrated p^O*
(weak-c MAPs) and credits spread p^O* (strong-c). occam = data-Hessian Occam from
the EBev evidence (a tau-Hessian approximation; the barrier-Hessian refinement is
secondary to the prior term). Reuses existing files -- no new GPU runs.
"""
import json, glob, os, re, math, collections
CM = "/work/SzollosiU/gergely-szollosi/williams_run/cleanml"
CS = [1000, 3000]  # Dirichlet barrier strengths (cmlD0)
LN2 = math.log(2.0)
ROOTS = ["Alti","AMD","Asgard","Cluster2","DPANN","Eury","HaloThermoplas","Kor","TAC","TackA"]

def barrier_logprior_nats(omega, c):
    # pi uniform over S branches; sum_e pi_e log2 p^O_e = mean_e log2 p^O_e
    mx = max(omega); ex = [2.0**(w-mx) for w in omega]; Z = sum(ex)
    log2p = [(w-mx) - math.log2(Z) for w in omega]   # log2 p^O_e
    mean_log2p = sum(log2p)/len(log2p)
    return LN2 * c * mean_log2p

data = collections.defaultdict(dict)  # root -> {c: logZ_barrier}
for c in CS:
    for r in ROOTS:
        sc = f"{CM}/cmlD0_{c}_{r}.rates.txt.json"
        ev = f"{CM}/EBev_{c}_{r}.evidence.json"
        if not (os.path.exists(sc) and os.path.exists(ev)):
            continue
        d = json.load(open(sc)); e = json.load(open(ev))
        om = (d.get("origination") or {}).get("omega_log2")
        if not om or e.get("occam_factor") is None:
            continue
        dL = d["data_log_likelihood_ln"]; occ = e["occam_factor"]
        logZ = dL + barrier_logprior_nats(om, c) + occ
        data[r][c] = logZ

print("=== BARRIER-CONSISTENT evidence log_Z by (root, Dirichlet c) ; EB = argmax_c ===")
print("%-16s %14s %14s | %12s %s" % ("root", "c=1000", "c=3000", "EB max", "c*"))
ebmax = {}
for r in ROOTS:
    row = data.get(r, {})
    if not row: continue
    cells = [("%14.1f" % row[c]) if c in row else "%14s" % "-" for c in CS]
    best = max(row.values()); cstar = [c for c in CS if row.get(c) == best][0]
    ebmax[r] = (best, cstar)
    print("%-16s %s | %12.1f c=%s" % (r, " ".join(cells), best, cstar))

print("\n=== BARRIER-CONSISTENT EB rooting ===")
done = sorted(ebmax.items(), key=lambda kv: -kv[1][0])
if done:
    b = done[0][1][0]
    for i, (r, (v, cstar)) in enumerate(done, 1):
        tag = "[deep]" if r in ("Eury","DPANN","TackA") else ("[MHH]" if r == "HaloThermoplas" else "")
        print("  %2d. %-16s logZ=%14.1f  dZ=%9.1f  (c*=%s) %s" % (i, r, v, v-b, cstar, tag))
