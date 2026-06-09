"""I/O adapters for external phylogenetic formats.

Currently provides :mod:`gpurec.io.ale`, a loader for classic ALEobserve
``.ale`` conditional-clade-probability files.
"""
from .ale import AleData, parse_ale_file, build_family_from_ale, default_species_of_leaf

__all__ = [
    "AleData",
    "parse_ale_file",
    "build_family_from_ale",
    "default_species_of_leaf",
]
