import math
import time
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, PreTrainedModel


@torch.no_grad()
def evaluate_fluency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "validation",
    dataset_name: str = "wikitext-2-raw-v1",
    stride: int = 256,
    max_length: int = 512,
    max_tokens: Optional[int] = 10000,
) -> Dict[str, Any]:
    """
    Evaluates language modeling fluency on WikiText-2.
    Computes both Perplexity (PPL) and Bits-Per-Character (BPC) using
    sliding window token evaluation.
    """
    model.eval()
    start_time = time.time()
    device = next(model.parameters()).device

    # Load wikitext dataset (Salesforce/wikitext namespace for modern HF Hub)
    try:
        ds = load_dataset("Salesforce/wikitext", dataset_name, split=split)
    except Exception:
        ds = load_dataset("wikitext", dataset_name, split=split)
    full_text = "\n\n".join([row["text"] for row in ds if row["text"].strip()])

    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    total_seq_len = input_ids.size(0)

    if max_tokens is not None and max_tokens > 0:
        total_seq_len = min(total_seq_len, max_tokens)
        input_ids = input_ids[:total_seq_len]

    total_loss = 0.0
    evaluated_tokens = 0
    total_characters = 0

    # Sliding-window evaluation
    prev_end = 0
    for begin_loc in range(0, total_seq_len, stride):
        end_loc = min(begin_loc + max_length, total_seq_len)
        trg_len = end_loc - prev_end
        input_ids_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)

        target_ids = input_ids_chunk.clone()
        # Only compute loss on target tokens (those not evaluated in previous stride)
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids=input_ids_chunk, labels=target_ids)
        # outputs.loss is mean over unmasked target tokens
        neg_log_likelihood = outputs.loss.item()

        # Count characters for BPC
        chunk_text = tokenizer.decode(input_ids[prev_end:end_loc], skip_special_tokens=True)
        total_characters += len(chunk_text)

        total_loss += neg_log_likelihood * trg_len
        evaluated_tokens += trg_len
        prev_end = end_loc

        if end_loc >= total_seq_len:
            break

    runtime_s = time.time() - start_time
    avg_nll = total_loss / max(1, evaluated_tokens)

    # Prevent overflow on severed models
    try:
        ppl = math.exp(avg_nll)
    except OverflowError:
        ppl = float("inf")

    avg_chars_per_token = total_characters / max(1, evaluated_tokens)
    bpc = (avg_nll / math.log(2)) / max(0.1, avg_chars_per_token)

    return {
        "task": "fluency",
        "dataset": dataset_name,
        "split": split,
        "nll": float(avg_nll),
        "perplexity": float(ppl),
        "bits_per_character": float(bpc),
        "evaluated_tokens": int(evaluated_tokens),
        "total_characters": int(total_characters),
        "avg_chars_per_token": float(avg_chars_per_token),
        "runtime_s": float(runtime_s),
    }
