from typing import List, Tuple
from gemma.surgery import SurgerySpec

SEVERITIES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
POSITIONS = ["early", "middle", "late", "random"]
COMPONENTS = ["block", "attn", "mlp"]
TASKS = ["fluency", "valence", "structural"]

EXPERIMENTS: List[Tuple[SurgerySpec, str]] = []

for task in TASKS:
    for comp in COMPONENTS:
        for pos in POSITIONS:
            for sev in SEVERITIES:
                spec = SurgerySpec(
                    surgery_type="layers",
                    severity=sev,
                    component=comp,
                    position=pos,
                    seed=42,
                )
                EXPERIMENTS.append((spec, task))
