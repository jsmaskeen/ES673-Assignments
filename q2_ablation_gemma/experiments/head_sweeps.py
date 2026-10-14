from typing import List, Tuple
from gemma.surgery import SurgerySpec

SEVERITIES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
HEAD_MODES = ["q_only", "kv_group", "full_head"]
TASKS = ["fluency", "valence", "structural"]
SEEDS = [42, 101, 202]

EXPERIMENTS: List[Tuple[SurgerySpec, str]] = []

for task in TASKS:
    for mode in HEAD_MODES:
        for sev in SEVERITIES:
            # Baseline (0.0) needs only 1 seed
            seeds_to_run = [42] if sev == 0.0 else SEEDS
            for s in seeds_to_run:
                spec = SurgerySpec(
                    surgery_type="heads",
                    severity=sev,
                    head_mode=mode,
                    position="random",
                    seed=s,
                )
                EXPERIMENTS.append((spec, task))
