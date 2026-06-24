"""
Command-line interface for gene tree / species tree reconciliation.

Evaluates the log-likelihood of gene trees given a species tree and DTL parameters.
"""

import argparse
import sys

import torch

from ..core.model import GeneDataset


def main():
    """Main CLI function.

    Subcommands:
      gpurec fit ...        general DTL+origination RATE OPTIMISATION (the fitter)
      gpurec reconcile ...  forward likelihood at fixed rates (the default below)
    A bare ``gpurec --species ... --gene ...`` keeps the legacy forward behaviour.
    """
    argv = sys.argv[1:]
    if argv and argv[0] == "fit":
        from .fit import main as fit_main
        return fit_main(argv[1:])
    if argv and argv[0] == "reconcile":
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="GPU-accelerated DTL reconciliation via CCP likelihood"
    )
    parser.add_argument("--species", required=True, help="Species tree file (.nwk)")
    parser.add_argument(
        "--gene", required=True, nargs="+", help="Gene tree file(s) (.nwk)"
    )
    parser.add_argument(
        "--delta", type=float, default=1e-10, help="Duplication rate (default: 1e-10)"
    )
    parser.add_argument(
        "--tau", type=float, default=1e-10, help="Transfer rate (default: 1e-10)"
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_param",
        type=float,
        default=1e-10,
        help="Loss rate (default: 1e-10)",
    )
    parser.add_argument(
        "--iters", type=int, default=2000, help="Maximum iterations (default: 2000)"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Computation dtype (default: float64)",
    )
    args = parser.parse_args(argv)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    dataset = GeneDataset(
        species_tree_path=args.species,
        gene_tree_paths=args.gene,
        genewise=False,
        specieswise=False,
        pairwise=False,
        dtype=dtype,
        device=device,
    )

    # Set DTL parameters
    for i in range(len(dataset)):
        dataset.set_params(i, D=args.delta, T=args.tau, L=args.lambda_param)

    # Compute likelihoods
    for i in range(len(dataset)):
        result = dataset.compute_likelihood(
            i,
            max_iters_E=args.iters,
            max_iters_Pi=args.iters,
            device=device,
            dtype=dtype,
        )
        print(f"Family {i}: log-likelihood = {result['log_likelihood']:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
