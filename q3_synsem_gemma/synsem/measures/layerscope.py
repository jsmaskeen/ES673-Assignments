from typing import Dict, List, Any, Tuple

import numpy as np
import torch
try:
    import scipy.stats as stats
except ImportError:
    stats = None

# Import prober from common_gemma
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

from common_gemma.prober import get_sentence_embeddings_all_layers, get_token_embeddings_all_layers
from .geometry import anisotropy_corrected_cosine, compute_rsm


def evaluate_stsb_across_layers(
    model,
    tokenizer,
    stsb_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates STS-B semantic textual similarity at every layer (0 to 26).
    Computes Spearman rank correlation between model cosine and human gold rating.
    """
    num_layers = 27
    gold_scores = [item["score"] for item in stsb_items]
    layer_predicted_similarities: List[List[float]] = [[] for _ in range(num_layers)]

    for item in stsb_items:
        layers1 = get_sentence_embeddings_all_layers(model, tokenizer, item["sentence1"])
        layers2 = get_sentence_embeddings_all_layers(model, tokenizer, item["sentence2"])

        for l in range(min(len(layers1), len(layers2), num_layers)):
            sim = anisotropy_corrected_cosine(layers1[l], layers2[l])
            layer_predicted_similarities[l].append(sim)

    spearman_rhos: List[float] = []
    p_values: List[float] = []

    for l in range(num_layers):
        if len(layer_predicted_similarities[l]) >= 5:
            rho, pval = stats.spearmanr(layer_predicted_similarities[l], gold_scores)
            spearman_rhos.append(float(rho) if not np.isnan(rho) else 0.0)
            p_values.append(float(pval) if not np.isnan(pval) else 1.0)
        else:
            spearman_rhos.append(0.0)
            p_values.append(1.0)

    best_layer = int(np.argmax(spearman_rhos))
    peak_rho = spearman_rhos[best_layer]

    return {
        "num_layers": num_layers,
        "spearman_rhos": spearman_rhos,
        "p_values": p_values,
        "best_layer": best_layer,
        "peak_rho": peak_rho,
        "sample_count": len(stsb_items),
    }


def evaluate_wic_across_layers(
    model,
    tokenizer,
    wic_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates SuperGLUE WiC polysemy disambiguation at every layer (0 to 26).
    Compares contextual token similarity for same-sense (1) vs different-sense (0) pairs.
    """
    num_layers = 27
    same_sense_sims: List[List[float]] = [[] for _ in range(num_layers)]
    diff_sense_sims: List[List[float]] = [[] for _ in range(num_layers)]

    for item in wic_items:
        word = item["word"]
        s1, s2 = item["sentence1"], item["sentence2"]
        st1, end1 = item.get("start1", None), item.get("end1", None)
        st2, end2 = item.get("start2", None), item.get("end2", None)
        is_same = (item["label"] == 1)

        vecs1 = get_token_embeddings_all_layers(model, tokenizer, s1, word, char_start=st1, char_end=end1)
        vecs2 = get_token_embeddings_all_layers(model, tokenizer, s2, word, char_start=st2, char_end=end2)

        if vecs1 and vecs2:
            for l in range(min(len(vecs1), len(vecs2), num_layers)):
                sim = anisotropy_corrected_cosine(vecs1[l], vecs2[l])
                if is_same:
                    same_sense_sims[l].append(sim)
                else:
                    diff_sense_sims[l].append(sim)

    same_means = [float(np.mean(sims)) if sims else 0.0 for sims in same_sense_sims]
    diff_means = [float(np.mean(sims)) if sims else 0.0 for sims in diff_sense_sims]
    divergence = [s - d for s, d in zip(same_means, diff_means)]

    return {
        "num_layers": num_layers,
        "same_sense_cosine_by_layer": same_means,
        "diff_sense_cosine_by_layer": diff_means,
        "divergence_by_layer": divergence,
        "peak_divergence_layer": int(np.argmax(divergence)),
        "max_divergence": float(max(divergence)) if divergence else 0.0,
    }


def compute_layer_rsa_matrix(
    model,
    tokenizer,
    probe_sentences: List[str],
) -> np.ndarray:
    """
    Computes a 27 x 27 Representational Similarity Analysis (RSA) matrix.
    Each cell (i, j) is the Spearman correlation between the Representational
    Similarity Matrix (RSM) at Layer i and Layer j.
    """
    num_layers = 27
    layer_sentence_embeddings: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

    for sent in probe_sentences:
        layer_vecs = get_sentence_embeddings_all_layers(model, tokenizer, sent)
        for l in range(min(len(layer_vecs), num_layers)):
            layer_sentence_embeddings[l].append(layer_vecs[l])

    # Compute RSM for each layer
    layer_rsms: List[np.ndarray] = []
    for l in range(num_layers):
        rsm = compute_rsm(layer_sentence_embeddings[l])
        layer_rsms.append(rsm)

    # Cross-correlate upper-triangle vectors of each layer's RSM
    rsa_matrix = np.zeros((num_layers, num_layers), dtype=np.float32)
    n = len(probe_sentences)
    triu_indices = np.triu_indices(n, k=1)

    for i in range(num_layers):
        vec_i = layer_rsms[i][triu_indices]
        for j in range(i, num_layers):
            vec_j = layer_rsms[j][triu_indices]
            rho, _ = stats.spearmanr(vec_i, vec_j)
            val = float(rho) if not np.isnan(rho) else 0.0
            rsa_matrix[i, j] = val
            rsa_matrix[j, i] = val

    return rsa_matrix
