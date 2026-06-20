import json, glob, os, re, collections
CM="/work/SzollosiU/gergely-szollosi/williams_run/cleanml"
CS=[10,100,1000,3000]
ROOTS=["Alti","AMD","Asgard","Cluster2","DPANN","Eury","HaloThermoplas","Kor","TAC","TackA"]
data=collections.defaultdict(dict)  # root -> {c: logZ}
for c in CS:
    for r in ROOTS:
        f=f"{CM}/EBev_{c}_{r}.evidence.json"
        if os.path.exists(f):
            try: data[r][c]=json.load(open(f)).get("log_Z")
            except: pass
print("=== evidence log_Z by (root, anti-concentration c) ; EB picks argmax_c ===")
print("%-16s %12s %12s %12s %12s | %10s"%("root","c=10","c=100","c=1000","c=3000","EB max"))
ebmax={}
for r in ROOTS:
    row=data.get(r,{})
    if not row: continue
    cells=[("%12.1f"%row[c]) if c in row and row[c] is not None else "%12s"%"-" for c in CS]
    best=max((v for v in row.values() if v is not None), default=None)
    ebmax[r]=best
    print("%-16s %s | %10s"%(r," ".join(cells), ("%.1f"%best) if best else "-"))
print("\n=== EB-selected rooting (rank by max-evidence log_Z) ===")
done=[(r,v) for r,v in ebmax.items() if v is not None]
done.sort(key=lambda x:-x[1]); 
if done:
    b=done[0][1]
    for i,(r,v) in enumerate(done,1):
        tag="[deep]" if r in ("Eury","DPANN","TackA") else ("[MHH-ish]" if r in("HaloThermoplas",) else "")
        bc=[c for c in CS if data[r].get(c)==v]
        print("  %2d. %-16s logZ=%12.1f  dZ=%9.1f  (c*=%s) %s"%(i,r,v,v-b,bc[0] if bc else "?",tag))
