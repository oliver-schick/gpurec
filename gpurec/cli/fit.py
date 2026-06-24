"""General, dataset-agnostic ``gpurec fit`` command.

Optimise per-branch (or clade-grouped) DTL + origination rates for ANY rooted
species tree and a directory of gene families (ALEobserve ``.ale`` CCPs), with the
full set of model / optimisation / prior flags. This is the general interface to the
proven branch-wise fitter: instead of the dataset-specific
``run_undine_branchwise.py --root <name>`` convention, you pass the inputs explicitly:

    gpurec fit --species-tree sp.nwk --ale-dir ale/ --origination optimize \\
               --fm-mode e-only --family-batch-size 500 --steps 250 --out fit.rates.txt

Run ``gpurec fit --help`` for every flag. Requires a CUDA GPU (Triton). The command
must be run from a gpurec repository checkout (it reuses the validated fit driver).
"""
import os
import sys


def _locate_driver():
    """Put the repo's experiments/ dir on the path and import the fit driver."""
    here = os.path.dirname(os.path.abspath(__file__))     # <repo>/gpurec/cli
    repo = os.path.dirname(os.path.dirname(here))         # <repo>
    exp = os.path.join(repo, "experiments")
    if os.path.isdir(exp) and exp not in sys.path:
        sys.path.insert(0, exp)
    try:
        import run_undine_branchwise as drv               # noqa: E402
    except ImportError as e:                              # pragma: no cover
        raise SystemExit(
            f"gpurec fit: could not import the fit driver from {exp!r}: {e}\n"
            "Run `gpurec fit` from inside a gpurec repository checkout."
        )
    return drv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    drv = _locate_driver()
    # The general path needs explicit inputs; nudge the user if neither is given.
    if argv and argv[0] not in ("-h", "--help"):
        if not any(a == "--species-tree" for a in argv) and \
           not any(a == "--root" for a in argv):
            sys.stderr.write(
                "gpurec fit: pass --species-tree <Newick> and --ale-dir <dir> "
                "(see `gpurec fit --help`).\n")
    return drv.main(argv)


if __name__ == "__main__":
    sys.exit(main())
