"""
Common Gemma-2 Foundation Engine
---------------------------------
Unified core module providing:
- backbone: Gemma-2 loading in bfloat16 with eager attention
- surgery: GQA head surgery, position-aware layer surgery, embedding masking
- prober: 27-layer hidden state extraction, token-level embeddings, and pooling
"""

from .backbone import load_backbone, GemmaTelemetry, get_hf_token
from .surgery import SurgerySpec, apply_surgery
from .prober import (
    extract_all_layer_hidden_states,
    get_sentence_embeddings_all_layers,
    get_token_embeddings_all_layers,
)

__all__ = [
    "load_backbone",
    "GemmaTelemetry",
    "get_hf_token",
    "SurgerySpec",
    "apply_surgery",
    "extract_all_layer_hidden_states",
    "get_sentence_embeddings_all_layers",
    "get_token_embeddings_all_layers",
]
