"""
Dataset loaders and procedural generators for Syntax vs. Semantics analysis:
- blimp: 10 advanced linguistic minimal pair subsets
- stsb: Semantic textual similarity continuous human ratings (0-5)
- wic: Words in Context contextual polysemy disambiguation
- hellaswag: Physical and commonsense plausibility 4-way choice
- chomsky: Colorless-Green Type A/B/C/D procedural generator
"""

from .blimp import load_blimp_subset, BLIMP_ADVANCED_SUBSETS
from .stsb import load_stsb
from .wic import load_wic
from .hellaswag import load_hellaswag
from .chomsky import generate_chomsky_pairs

__all__ = [
    "load_blimp_subset",
    "BLIMP_ADVANCED_SUBSETS",
    "load_stsb",
    "load_wic",
    "load_hellaswag",
    "generate_chomsky_pairs",
]
