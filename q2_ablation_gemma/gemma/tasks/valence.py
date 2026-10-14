import time
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, PreTrainedModel


PROMPT_TEMPLATE = "Review: {text}\nQuestion: Is the overall sentiment of this review positive or negative?\nAnswer:"
LABELS = ["negative", "positive"]


def _sequence_logprob(
    model: PreTrainedModel,
    prompt_ids: List[int],
    completion_ids: List[int],
    device: torch.device,
) -> float:
    """
    Computes log P(completion_ids | prompt_ids) under teacher forcing.
    """
    joint_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor([joint_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    comp_len = len(completion_ids)

    outputs = model(input_ids=input_tensor)
    logits = outputs.logits[0]  # [seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    # Next token at pos t is predicted by logits at t - 1
    total_lp = 0.0
    for i in range(comp_len):
        target_token = completion_ids[i]
        logit_pos = (prompt_len - 1) + i
        total_lp += log_probs[logit_pos, target_token].item()

    return total_lp


@torch.no_grad()
def evaluate_valence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "test",
    dataset_name: str = "stanfordnlp/imdb",
    max_samples: int = 200,
    max_text_chars: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates semantic sentiment classification via generative forced choice
    on the IMDB dataset.
    """
    model.eval()
    start_time = time.time()
    device = next(model.parameters()).device

    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception:
        try:
            ds = load_dataset("stanfordnlp/imdb", split=split)
        except Exception:
            ds = load_dataset("imdb", split=split)

    # Deterministically sample examples
    ds = ds.shuffle(seed=seed)
    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(len(ds), max_samples)))

    # Encode label tokens with leading space
    label_ids = {
        0: tokenizer.encode(" negative", add_special_tokens=False),
        1: tokenizer.encode(" positive", add_special_tokens=False),
    }

    correct_predictions = 0
    total_evaluated = 0
    confidence_margins: List[float] = []

    for item in ds:
        text = item["text"][:max_text_chars].strip()
        gold_label = int(item["label"])

        prompt_str = PROMPT_TEMPLATE.format(text=text)
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)

        lp_neg = _sequence_logprob(model, prompt_ids, label_ids[0], device)
        lp_pos = _sequence_logprob(model, prompt_ids, label_ids[1], device)

        pred_label = 1 if lp_pos > lp_neg else 0
        margin = lp_pos - lp_neg if gold_label == 1 else lp_neg - lp_pos

        if pred_label == gold_label:
            correct_predictions += 1
        confidence_margins.append(margin)
        total_evaluated += 1

    runtime_s = time.time() - start_time
    accuracy = correct_predictions / max(1, total_evaluated)
    avg_margin = sum(confidence_margins) / max(1, len(confidence_margins))

    return {
        "task": "valence",
        "dataset": dataset_name,
        "split": split,
        "accuracy": float(accuracy),
        "avg_confidence_margin": float(avg_margin),
        "correct_count": int(correct_predictions),
        "total_evaluated": int(total_evaluated),
        "runtime_s": float(runtime_s),
    }
