"""
Perturbation Modality Sweeps:
Compares structural necrosis ('zero') vs synaptic jitter ('gaussian_noise')
vs tonic baseline clamping ('mean_clamp') across Attention Heads and Layers.
"""
from typing import List, Tuple
from gemma.surgery import SurgerySpec

PERTURBATION_MODES = ["zero", "gaussian_noise", "mean_clamp"]
SEVERITIES = [0.1, 0.25, 0.5]
TASKS = ["fluency", "valence", "structural"]

EXPERIMENTS: List[Tuple[SurgerySpec, str]] = []

for p_mode in PERTURBATION_MODES:
    for task in TASKS:
        for sev in SEVERITIES:
            # 1. Attention Head Perturbation (q_only)
            spec_heads = SurgerySpec(
                surgery_type="heads",
                severity=sev,
                head_mode="q_only",
                perturbation_type=p_mode,
                seed=42,
            )
            EXPERIMENTS.append((spec_heads, task))

            # 2. Attention Layer Perturbation (middle position)
            spec_layers = SurgerySpec(
                surgery_type="layers",
                severity=sev,
                component="attn",
                position="middle",
                perturbation_type=p_mode,
                seed=42,
            )
            EXPERIMENTS.append((spec_layers, task))
