import math
from typing import Dict, List, Any

import numpy as np
import torch
try:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel
except ImportError:
    PreTrainedTokenizerBase, PreTrainedModel = Any, Any


def compute_cohens_d(values: List[float]) -> float:
    """Computes standardized Cohen's d effect size for a sample mean against 0."""
    if not values or len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-8:
        return float(np.sign(mean) * 10.0)
    return float(mean / std)


@torch.no_grad()
def _sentence_nll(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> float:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if input_ids.size(1) <= 1:
        return 0.0
    outputs = model(input_ids=input_ids, labels=input_ids)
    # Total sequence NLL = mean_loss * (seq_len - 1)
    return outputs.loss.item() * (input_ids.size(1) - 1)


@torch.no_grad()
def evaluate_pairwise_preference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    minimal_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates minimal pair forced choice (e.g. BLiMP).
    Correct if P(sentence_good) > P(sentence_bad), i.e. NLL(good) < NLL(bad).
    Reports accuracy, raw margins, and standardized Cohen's d effect size.
    """
    model.eval()
    device = next(model.parameters()).device

    correct_count = 0
    margins: List[float] = []
    item_results: List[Dict[str, Any]] = []

    for item in minimal_pairs:
        good = item["sentence_good"]
        bad = item["sentence_bad"]

        nll_good = _sentence_nll(model, tokenizer, good, device)
        nll_bad = _sentence_nll(model, tokenizer, bad, device)

        # Margin: positive indicates model prefers the grammatical sentence
        margin = nll_bad - nll_good
        is_correct = 1 if margin > 0 else 0

        if is_correct:
            correct_count += 1
        margins.append(margin)

        item_results.append({
            "id": item.get("id", ""),
            "nll_good": float(nll_good),
            "nll_bad": float(nll_bad),
            "margin": float(margin),
            "correct": is_correct,
        })

    total = len(minimal_pairs)
    accuracy = correct_count / max(1, total)
    cohens_d = compute_cohens_d(margins)
    avg_margin = float(np.mean(margins)) if margins else 0.0

    return {
        "total_pairs": total,
        "correct_count": correct_count,
        "accuracy": float(accuracy),
        "avg_margin": avg_margin,
        "cohens_d": cohens_d,
        "item_results": item_results,
    }
