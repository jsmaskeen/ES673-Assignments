"""
Advanced linguistic representation measures:
- surprisal: Word-by-word surprisal profiles and violation spike analysis
- preference: Forced-choice accuracy and standardized Cohen's d effect size
- geometry: Anisotropy-corrected cosine and Representational Similarity Matrices
- layerscope: 27-layer representational similarity analysis (RSA) and probing
"""

from .surprisal import compute_word_surprisal, sentence_perplexity
from .preference import evaluate_pairwise_preference, compute_cohens_d
from .geometry import anisotropy_corrected_cosine, compute_rsm
from .layerscope import evaluate_stsb_across_layers, evaluate_wic_across_layers, compute_layer_rsa_matrix

__all__ = [
    "compute_word_surprisal",
    "sentence_perplexity",
    "evaluate_pairwise_preference",
    "compute_cohens_d",
    "anisotropy_corrected_cosine",
    "compute_rsm",
    "evaluate_stsb_across_layers",
    "evaluate_wic_across_layers",
    "compute_layer_rsa_matrix",
]
