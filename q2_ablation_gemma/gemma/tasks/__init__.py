"""
Downstream evaluation tasks for Gemma-2 ablation studies:
- fluency: WikiText-2 language modeling (PPL & BPC)
- valence: IMDB review sentiment classification (generative forced-choice)
- structural: Garden-path relative clause syntactic parsing probe
"""

from .fluency import evaluate_fluency
from .valence import evaluate_valence
from .structural import evaluate_garden_path

__all__ = ["evaluate_fluency", "evaluate_valence", "evaluate_garden_path"]
