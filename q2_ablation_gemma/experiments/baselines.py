from typing import List, Tuple
from gemma.surgery import SurgerySpec

# Baseline evaluations with zero ablation
EXPERIMENTS: List[Tuple[SurgerySpec, str]] = [
    (SurgerySpec(surgery_type="none", severity=0.0, seed=42), "fluency"),
    (SurgerySpec(surgery_type="none", severity=0.0, seed=42), "valence"),
    (SurgerySpec(surgery_type="none", severity=0.0, seed=42), "structural"),
]
