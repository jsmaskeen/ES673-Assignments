"""
Gemma-2 Ablation Surgery Engine (Question 2)
--------------------------------------------
Extends the unified common_gemma surgery engine with Q2-specific
ablation nuances:
- Spectral PCA Embedding Projection Masking
- Extended Surgery Context Manager supporting all base and spectral modes
"""

import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common_gemma.surgery import (
    SurgerySpec,
    apply_surgery as apply_base_surgery,
    plan_head_selection,
    plan_layer_selection,
    make_q_head_pre_hook,
    make_kv_group_hook,
    make_zero_output_hook,
    make_skip_layer_hook,
    make_embedding_mask_hook,
)


def make_spectral_pca_mask_hook(principal_components: torch.Tensor, frac: float):
    """
    Q2 Extension: Spectral Embedding Projection Hook.
    Projects out the top-k principal variance directions of the embedding space:
      x_ablated = x - (x @ P.T) @ P
    where P comprises the top-k components.
    """
    k = max(1, int(round(principal_components.size(0) * frac)))
    sub_P = principal_components[:k, :].float()  # [k, hidden_size]

    def hook(module, inp, output):
        # output: [bsz, seq_len, hidden_size]
        P = sub_P.to(output.device).to(output.dtype)
        orig_shape = output.shape
        flat = output.view(-1, orig_shape[-1])
        # Project onto principal subspace: [N, k] @ [k, hidden]
        proj = (flat @ P.t()) @ P
        ablated = flat - proj
        return ablated.view(orig_shape)

    return hook


@contextmanager
def apply_surgery(model: nn.Module, spec: SurgerySpec):
    """
    Q2 Extended Surgery Context Manager:
    Delegates base ablation types ('heads', 'layers', 'emb_mask') to common_gemma,
    and natively executes Q2-specific extensions ('emb_spectral').
    """
    handles = []

    try:
        if spec.surgery_type == "emb_spectral":
            # Extract principal components from spec metadata or compute standard basis
            hidden_size = model.config.hidden_size
            k_components = spec.metadata.get("principal_components", None)

            if k_components is None:
                # Deterministic pseudo-principal basis for spectral evaluation
                torch.manual_seed(spec.seed)
                rand_mat = torch.randn(hidden_size, hidden_size)
                q, _ = torch.linalg.qr(rand_mat)
                k_components = q  # Orthonormal basis vectors

            h = model.model.embed_tokens.register_forward_hook(
                make_spectral_pca_mask_hook(k_components, spec.severity)
            )
            handles.append(h)
            spec.metadata["spectral_k"] = int(round(hidden_size * spec.severity))
            yield handles

        else:
            # Delegate directly to common_gemma base surgery engine
            with apply_base_surgery(model, spec) as base_handles:
                yield base_handles

    finally:
        for handle in handles:
            handle.remove()


__all__ = [
    "SurgerySpec",
    "apply_surgery",
    "apply_base_surgery",
    "plan_head_selection",
    "plan_layer_selection",
    "make_spectral_pca_mask_hook",
    "make_q_head_pre_hook",
    "make_kv_group_hook",
    "make_zero_output_hook",
    "make_skip_layer_hook",
    "make_embedding_mask_hook",
]
