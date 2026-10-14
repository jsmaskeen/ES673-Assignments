import math
import time
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel


# ---------------------------------------------------------------------------
# Hand-crafted Syntactic Probe Stimuli
# ---------------------------------------------------------------------------

GARDEN_PATH_PAIRS: List[Tuple[str, str]] = [
    # (Reduced / Garden-Path, Overt / Disambiguated)
    ("The horse raced past the barn fell.", "The horse that was raced past the barn fell."),
    ("The patient persuaded the doctor was incompetent.", "The patient persuaded that the doctor was incompetent."),
    ("The cotton clothing is made of grows in Mississippi.", "The cotton that clothing is made of grows in Mississippi."),
    ("The florist sent the flowers was very pleased.", "The florist who was sent the flowers was very pleased."),
    ("The artist painted the portrait died last night.", "The artist who was painted in the portrait died last night."),
    ("The girl told the story cried aloud.", "The girl who was told the story cried aloud."),
    ("The criminal warned about the ambush fled immediately.", "The criminal who was warned about the ambush fled immediately."),
    ("The lawyer defended the defendant won the case.", "The lawyer who defended the defendant won the case."),
    ("The dog walked through the park barked loudly.", "The dog that was walked through the park barked loudly."),
    ("The child shown the puzzle solved it quickly.", "The child who was shown the puzzle solved it quickly."),
]

AGREEMENT_ATTRACTOR_PAIRS: List[Dict[str, str]] = [
    # Long-range agreement with intervenor attractor nouns
    {"prefix": "The key to the cabinets ", "correct": "is", "incorrect": "are", "suffix": " on the shelf."},
    {"prefix": "The keys to the cabinet ", "correct": "are", "incorrect": "is", "suffix": " on the shelf."},
    {"prefix": "The chef near the ovens ", "correct": "cooks", "incorrect": "cook", "suffix": " every morning."},
    {"prefix": "The chefs near the oven ", "correct": "cook", "incorrect": "cooks", "suffix": " every morning."},
    {"prefix": "The book about the ancient empires ", "correct": "was", "incorrect": "were", "suffix": " fascinating."},
    {"prefix": "The books about the ancient empire ", "correct": "were", "incorrect": "was", "suffix": " fascinating."},
    {"prefix": "The scientist behind the revolutionary discoveries ", "correct": "has", "incorrect": "have", "suffix": " retired."},
    {"prefix": "The scientists behind the revolutionary discovery ", "correct": "have", "incorrect": "has", "suffix": " retired."},
    {"prefix": "The singer surrounded by adoring fans ", "correct": "smiles", "incorrect": "smile", "suffix": " brightly."},
    {"prefix": "The singers surrounded by an adoring fan ", "correct": "smile", "incorrect": "smiles", "suffix": " brightly."},
    {"prefix": "The architect of the towering skyscrapers ", "correct": "lives", "incorrect": "live", "suffix": " in Chicago."},
    {"prefix": "The architects of the towering skyscraper ", "correct": "live", "incorrect": "lives", "suffix": " in Chicago."},
    {"prefix": "The dog chasing the terrified cats ", "correct": "runs", "incorrect": "run", "suffix": " through the yard."},
    {"prefix": "The dogs chasing the terrified cat ", "correct": "run", "incorrect": "runs", "suffix": " through the yard."},
    {"prefix": "The author of those contentious articles ", "correct": "writes", "incorrect": "write", "suffix": " with passion."},
    {"prefix": "The authors of that contentious article ", "correct": "write", "incorrect": "writes", "suffix": " with passion."},
]


def _sentence_nll(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> float:
    """Computes total Negative Log Likelihood of a full sentence."""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if input_ids.size(1) <= 1:
        return 0.0

    outputs = model(input_ids=input_ids, labels=input_ids)
    return outputs.loss.item() * (input_ids.size(1) - 1)


def _token_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefix: str,
    continuation: str,
    device: torch.device,
) -> float:
    """Computes log P(continuation | prefix)."""
    pref_ids = tokenizer.encode(prefix, add_special_tokens=True)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    joint_ids = pref_ids + cont_ids

    input_tensor = torch.tensor([joint_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_tensor)
    logits = outputs.logits[0]
    log_probs = F.log_softmax(logits, dim=-1)

    total_lp = 0.0
    for i, tok in enumerate(cont_ids):
        pos = (len(pref_ids) - 1) + i
        total_lp += log_probs[pos, tok].item()
    return total_lp


@torch.no_grad()
def evaluate_garden_path(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_samples: int = 50,
) -> Dict[str, Any]:
    """
    Evaluates syntactic processing resilience:
    1. Garden-path penalty ratio (PPL of reduced vs overt relative clauses).
    2. Long-range Subject-Verb Agreement under attractor interference.
    """
    model.eval()
    start_time = time.time()
    device = next(model.parameters()).device

    # Part 1: Garden-Path Disambiguation
    gp_penalties: List[float] = []
    for gp_text, overt_text in GARDEN_PATH_PAIRS[:max_samples]:
        nll_gp = _sentence_nll(model, tokenizer, gp_text, device)
        nll_ov = _sentence_nll(model, tokenizer, overt_text, device)

        # Penalty ratio: PPL(reduced) / PPL(overt)
        # Expected > 1.0 in standard models (overt structure is easier to parse)
        try:
            ppl_gp = math.exp(nll_gp / max(1, len(tokenizer.encode(gp_text))))
            ppl_ov = math.exp(nll_ov / max(1, len(tokenizer.encode(overt_text))))
            ratio = ppl_gp / max(1e-4, ppl_ov)
        except OverflowError:
            ratio = 100.0
        gp_penalties.append(ratio)

    avg_gp_ratio = sum(gp_penalties) / max(1, len(gp_penalties))

    # Part 2: Long-Range Subject-Verb Agreement with Attractors
    correct_agreement = 0
    agreement_margins: List[float] = []

    for item in AGREEMENT_ATTRACTOR_PAIRS[:max_samples]:
        pref = item["prefix"]
        cor = " " + item["correct"].strip()
        inc = " " + item["incorrect"].strip()

        lp_cor = _token_logprob(model, tokenizer, pref, cor, device)
        lp_inc = _token_logprob(model, tokenizer, pref, inc, device)

        if lp_cor > lp_inc:
            correct_agreement += 1
        agreement_margins.append(lp_cor - lp_inc)

    total_aggr = len(AGREEMENT_ATTRACTOR_PAIRS[:max_samples])
    aggr_accuracy = correct_agreement / max(1, total_aggr)
    avg_margin = sum(agreement_margins) / max(1, len(agreement_margins))

    runtime_s = time.time() - start_time

    return {
        "task": "structural",
        "garden_path_ratio": float(avg_gp_ratio),
        "agreement_accuracy": float(aggr_accuracy),
        "agreement_margin": float(avg_margin),
        "num_gp_pairs": len(gp_penalties),
        "num_agreement_pairs": total_aggr,
        "runtime_s": float(runtime_s),
    }
