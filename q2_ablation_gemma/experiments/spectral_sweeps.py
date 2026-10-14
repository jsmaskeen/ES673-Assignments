"""
Spectral PCA Subspace Elimination Sweeps:
Evaluates geometric fragility by projecting out top-k principal variance directions
from the token embedding space.
"""
from typing import List, Tuple
from gemma.surgery import SurgerySpec

SEVERITIES = [0.05, 0.1, 0.25, 0.5]
TASKS = ["fluency", "valence", "structural"]

EXPERIMENTS: List[Tuple[SurgerySpec, str]] = []

for task in TASKS:
    for sev in SEVERITIES:
        spec = SurgerySpec(
            surgery_type="emb_spectral",
            severity=sev,
            seed=42,
        )
        EXPERIMENTS.append((spec, task))
