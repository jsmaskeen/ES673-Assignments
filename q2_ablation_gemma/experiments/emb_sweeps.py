from typing import List, Tuple
from gemma.surgery import SurgerySpec

SEVERITIES = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
TASKS = ["fluency", "valence", "structural"]

EXPERIMENTS: List[Tuple[SurgerySpec, str]] = []

for task in TASKS:
    for sev in SEVERITIES:
        spec = SurgerySpec(
            surgery_type="emb_mask",
            severity=sev,
            seed=42,
        )
        EXPERIMENTS.append((spec, task))
