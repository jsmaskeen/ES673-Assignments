import sys
import os

_curr = os.path.abspath(os.path.dirname(__file__))
while _curr and not os.path.exists(os.path.join(_curr, "common_gemma")):
    _parent = os.path.dirname(_curr)
    if _parent == _curr:
        break
    _curr = _parent
if os.path.exists(os.path.join(_curr, "common_gemma")) and _curr not in sys.path:
    sys.path.insert(0, _curr)

from common_gemma.surgery import SurgerySpec

# Causal interventions to evaluate dissociation between syntax and semantics
CAUSAL_CONDITIONS = [
    {"label": "Baseline (Intact)", "spec": SurgerySpec(surgery_type="none", severity=0.0)},
    {"label": "Early Layers Ablated (0..6)", "spec": SurgerySpec(surgery_type="layers", severity=0.25, component="block", position="early")},
    {"label": "Middle Layers Ablated (10..16)", "spec": SurgerySpec(surgery_type="layers", severity=0.25, component="block", position="middle")},
    {"label": "Late Layers Ablated (20..26)", "spec": SurgerySpec(surgery_type="layers", severity=0.25, component="block", position="late")},
    {"label": "Q-Heads Ablated (50%)", "spec": SurgerySpec(surgery_type="heads", severity=0.5, head_mode="q_only")},
    {"label": "KV-Groups Ablated (50%)", "spec": SurgerySpec(surgery_type="heads", severity=0.5, head_mode="kv_group")},
    {"label": "Embedding Masked (25%)", "spec": SurgerySpec(surgery_type="emb_mask", severity=0.25)},
]
