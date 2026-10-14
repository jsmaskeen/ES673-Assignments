import math
from typing import Dict, List, Any, Tuple

import torch
import torch.nn.functional as F
try:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel
except ImportError:
    PreTrainedTokenizerBase, PreTrainedModel = Any, Any


@torch.no_grad()
def compute_word_surprisal(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentence: str,
) -> Dict[str, Any]:
    """
    Computes token-by-token surprisal in bits: I(w_t) = -log2 P(w_t | w_{<t}).
    Returns tokens, surprisal values, mean surprisal, and peak surprisal spike.
    """
    model.eval()
    device = next(model.parameters()).device

    enc = tokenizer(sentence, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    seq_len = input_ids.size(1)

    if seq_len <= 1:
        return {
            "tokens": [sentence],
            "surprisals": [0.0],
            "mean_surprisal": 0.0,
            "peak_surprisal": 0.0,
            "peak_token": sentence,
        }

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Log softmax over vocab in base 2 (bits)
    log2_probs = F.log_softmax(logits[:-1], dim=-1) / math.log(2.0)

    surprisals: List[float] = []
    tokens: List[str] = []

    for t in range(1, seq_len):
        target_token_id = input_ids[0, t].item()
        tok_str = tokenizer.decode([target_token_id])
        tokens.append(tok_str)
        # Token at position t is predicted by logits at position t - 1
        surprisal_val = -log2_probs[t - 1, target_token_id].item()
        surprisals.append(max(0.0, surprisal_val))

    mean_surprisal = sum(surprisals) / max(1, len(surprisals))
    max_idx = int(torch.tensor(surprisals).argmax().item())
    peak_surprisal = surprisals[max_idx]
    peak_token = tokens[max_idx]

    return {
        "tokens": tokens,
        "surprisals": [float(round(s, 3)) for s in surprisals],
        "mean_surprisal": float(mean_surprisal),
        "peak_surprisal": float(peak_surprisal),
        "peak_token": peak_token,
    }


@torch.no_grad()
def sentence_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentence: str,
) -> float:
    """Computes overall sentence perplexity."""
    enc = tokenizer(sentence, return_tensors="pt")
    input_ids = enc["input_ids"].to(next(model.parameters()).device)
    if input_ids.size(1) <= 1:
        return 1.0

    outputs = model(input_ids=input_ids, labels=input_ids)
    nll = outputs.loss.item()
    try:
        return float(math.exp(nll))
    except OverflowError:
        return 1e6
