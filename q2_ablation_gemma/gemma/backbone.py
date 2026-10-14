"""
Gemma-2 Backbone Loader & Architecture Inspector (Question 2)
--------------------------------------------------------------
Imports the core Gemma-2 loader and telemetry from the unified
common_gemma foundation, and provides Q2-specific architectural
inspection utilities.
"""

import os
import sys
from typing import Dict, Any, List

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common_gemma.backbone import load_backbone, GemmaTelemetry, get_hf_token


def inspect_gqa_layout(model) -> Dict[str, Any]:
    """
    Q2 Extension: Inspects and returns the Grouped Query Attention (GQA)
    layout of Gemma-2, mapping every Query head to its governing Key/Value group.
    """
    cfg = model.config
    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // num_q
    ratio = num_q // num_kv

    group_mapping: Dict[int, List[int]] = {}
    for g in range(num_kv):
        group_mapping[g] = list(range(g * ratio, (g + 1) * ratio))

    return {
        "num_layers": cfg.num_hidden_layers,
        "num_query_heads": num_q,
        "num_key_value_heads": num_kv,
        "gqa_ratio": ratio,
        "head_dimension": head_dim,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": getattr(cfg, "intermediate_size", 4 * cfg.hidden_size),
        "kv_to_query_mapping": group_mapping,
    }


def summarize_model_parameters(model) -> Dict[str, Any]:
    """
    Q2 Extension: Returns a fine-grained parameter count breakdown across
    embedding, attention, and MLP submodules.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if total_params == 0 and hasattr(model, "model") and hasattr(model.model, "parameters"):
        total_params = sum(p.numel() for p in model.model.parameters())
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters()) if hasattr(model.model, "embed_tokens") else 0

    attn_params = 0
    mlp_params = 0
    if hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "parameters"):
                attn_params += sum(p.numel() for p in layer.self_attn.parameters())
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "parameters"):
                mlp_params += sum(p.numel() for p in layer.mlp.parameters())

    return {
        "total_parameters": total_params,
        "embedding_parameters": embed_params,
        "attention_parameters": attn_params,
        "mlp_parameters": mlp_params,
        "other_parameters": total_params - (embed_params + attn_params + mlp_params),
        "attention_pct": float(attn_params / max(1, total_params) * 100),
        "mlp_pct": float(mlp_params / max(1, total_params) * 100),
    }


__all__ = [
    "load_backbone",
    "GemmaTelemetry",
    "get_hf_token",
    "inspect_gqa_layout",
    "summarize_model_parameters",
]
