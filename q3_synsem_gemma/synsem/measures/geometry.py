from typing import List, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F


def anisotropy_corrected_cosine(
    u: Any,
    v: Any,
    mean_vec: Optional[torch.Tensor] = None,
) -> float:
    """
    Computes cosine similarity with anisotropy correction:
    Subtracting the mean embedding vector eliminates the representational cone effect
    common in deep transformers, yielding calibrated similarity metrics.
    """
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)

    u = u.view(-1).float()
    v = v.view(-1).float()

    if mean_vec is not None:
        m = mean_vec.view(-1).float()
        u = u - m
        v = v - m

    norm_u = torch.linalg.norm(u)
    norm_v = torch.linalg.norm(v)

    if norm_u < 1e-8 or norm_v < 1e-8:
        return 0.0

    cos = (u @ v) / (norm_u * norm_v)
    return float(cos.clamp(-1.0, 1.0).item())


def compute_rsm(
    vectors: List[torch.Tensor],
    mean_vec: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Computes an N x N Representational Similarity Matrix (RSM)
    for a list of N vectors.
    """
    n = len(vectors)
    rsm = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i, n):
            sim = anisotropy_corrected_cosine(vectors[i], vectors[j], mean_vec=mean_vec)
            rsm[i, j] = sim
            rsm[j, i] = sim

    return rsm
