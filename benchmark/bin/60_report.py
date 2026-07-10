#!/usr/bin/env python3
"""Build the gpurec-vs-AleRax runtime comparison from collected timing
records. Reads results/{deigo,saion}/**/timing*.json (+ the gpurec rate
sidecar for n_families_kept and the tool's own elapsed_s, and ruse.txt for
peak memory). Writes results/comparison.md and results/comparison.csv.

Pure stdlib; run after bin/50_collect.sh:
    python3 bin/60_report.py [results_dir]
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else
               Path(__file__).resolve().parent.parent / "results")


def _collect(results: Path) -> list[dict]:
    rows = []
    for tj in sorted(results.rglob("timing*.json")):
        try:
            rec = json.loads(tj.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {tj}: {exc}", file=sys.stderr)
            continue
        # n_families: prefer the gpurec sidecar's kept count; else AleRax's.
        n_fam = rec.get("n_families")
        elapsed_tool = None
        gpu_peak_gib = None
        side = rec.get("rates_sidecar")
        if side:
            sp = (tj.parent / Path(side).name)
            if sp.exists():
                try:
                    sj = json.loads(sp.read_text())
                    n_fam = sj.get("n_families_kept", n_fam)
                    elapsed_tool = sj.get("elapsed_s")
                    gpu_peak_gib = sj.get("peak_gib")   # true GPU peak (torch max_memory_allocated)
                except Exception:  # noqa: BLE001
                    pass
        # NOTE: we deliberately do NOT report ruse's "Memory" here -- it's host RSS,
        # not GPU memory (and for AleRax it's just the MPI launcher). The meaningful
        # figure is gpurec's GPU peak from its sidecar; AleRax (CPU/MPI) has none.
        wall = rec.get("wall_seconds")
        # Pairing tag from the output dir name: <tool>_<mode>_<root>_<phase>_<N>
        # -> pair gpurec vs alerax on the shared <...root_phase_N> suffix.
        m = re.match(r"(?:alerax|gpurec)_(.*)$", tj.parent.name)
        tag = m.group(1) if m else tj.parent.name
        rows.append({
            "tag": tag,
            "tool": rec.get("tool", "?"),
            "cluster": rec.get("cluster", "?"),
            "mode": rec.get("mode", "?"),
            "root": rec.get("root", ""),
            "n_families": n_fam,
            "wall_s": wall,
            "tool_elapsed_s": elapsed_tool,
            "peak_gpu_gib": round(gpu_peak_gib, 1) if gpu_peak_gib else None,
            "ms_per_family": round(1000.0 * wall / n_fam, 1)
                              if (wall and n_fam) else None,
            "job_id": rec.get("job_id"),
            "source": str(tj.relative_to(results)),
        })
    return rows


def main() -> int:
    if not RESULTS.exists():
        print(f"no results dir: {RESULTS}", file=sys.stderr)
        return 1
    rows = _collect(RESULTS)
    if not rows:
        print("No timing*.json found. Run bin/50_collect.sh after jobs finish.",
              file=sys.stderr)
        return 1

    cols = ["tag", "tool", "cluster", "mode", "root", "n_families", "wall_s",
            "tool_elapsed_s", "peak_gpu_gib", "ms_per_family", "job_id", "source"]
    csv_path = RESULTS / "comparison.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Markdown table, sorted by n_families then tool.
    rows.sort(key=lambda r: (r["n_families"] or 0, r["tool"]))
    md = ["# gpurec vs AleRax -- runtime comparison", "",
          "Same .ale gene families, same rooted species tree, same DTL model. "
          "AleRax = Deigo CPU/MPI; gpurec = Saion A100. `wall_s` is the ruse/srun "
          "wall time; `tool_elapsed_s` is gpurec's own optimize-loop timer. "
          "`GPU peak` is gpurec's `torch.cuda.max_memory_allocated` (from its rate "
          "sidecar); AleRax is CPU/MPI so it has no GPU-memory figure (shown `--`).", "",
          "| tool | cluster | mode | families | wall (s) | tool elapsed (s) | GPU peak (GiB) | ms/family |",
          "|------|---------|------|----------|----------|------------------|----------------|-----------|"]
    _d = lambda v: "--" if v is None else v
    for r in rows:
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["tool"], r["cluster"], r["mode"], _d(r["n_families"]), _d(r["wall_s"]),
            _d(r["tool_elapsed_s"]), _d(r["peak_gpu_gib"]), _d(r["ms_per_family"])))

    # Speedup lines: pair gpurec vs alerax on the shared run tag (same root +
    # phase + requested N), robust to gpurec's <4-species family drops.
    by_tag: dict = {}
    for r in rows:
        by_tag.setdefault(r["tag"], {})[r["tool"]] = r
    md += ["", "## Speedup (AleRax wall / gpurec wall) at matched runs", ""]
    any_pair = False
    for tag, d in sorted(by_tag.items()):
        if "gpurec" in d and "alerax" in d and d["gpurec"]["wall_s"] and d["alerax"]["wall_s"]:
            sp = d["alerax"]["wall_s"] / d["gpurec"]["wall_s"]
            md.append(f"- **{tag}**: AleRax {d['alerax']['wall_s']}s "
                      f"({d['alerax']['n_families']} fam) vs gpurec "
                      f"{d['gpurec']['wall_s']}s ({d['gpurec']['n_families']} fam)"
                      f"  ->  **{sp:.1f}x**")
            any_pair = True
    if not any_pair:
        md.append("- (no matched gpurec/AleRax pair yet -- run both tools at the "
                  "same phase, then re-collect)")

    (RESULTS / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {RESULTS/'comparison.md'} and {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
