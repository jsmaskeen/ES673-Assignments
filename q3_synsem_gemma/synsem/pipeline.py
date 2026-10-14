import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

# Ensure access to common_gemma and q3 root
_curr = os.path.abspath(os.path.dirname(__file__))
_repo_root = None
_q3_root = None
while _curr:
    if os.path.exists(os.path.join(_curr, "common_gemma")):
        _repo_root = _curr
    if os.path.exists(os.path.join(_curr, "synsem")) and os.path.exists(os.path.join(_curr, "experiments")):
        _q3_root = _curr
    _parent = os.path.dirname(_curr)
    if _parent == _curr:
        break
    _curr = _parent

for _p in [_repo_root, _q3_root]:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from common_gemma.backbone import load_backbone
from common_gemma.surgery import apply_surgery, SurgerySpec

from .datasets.blimp import load_blimp_subset, BLIMP_ADVANCED_SUBSETS
from .datasets.stsb import load_stsb
from .datasets.wic import load_wic
from .datasets.hellaswag import load_hellaswag
from .datasets.chomsky import generate_chomsky_pairs

from .measures.surprisal import compute_word_surprisal, sentence_perplexity
from .measures.preference import evaluate_pairwise_preference
from .measures.layerscope import (
    evaluate_stsb_across_layers,
    evaluate_wic_across_layers,
    compute_layer_rsa_matrix,
)
from experiments.causal_dissociation import CAUSAL_CONDITIONS


def run_blimp_evaluation(model, tokenizer, max_samples: int = 50) -> Dict[str, Any]:
    print("\n--- Running BLiMP Syntactic Minimal Pair Battery (10 Subsets) ---")
    results = {}
    total_correct = 0
    total_pairs = 0

    for subset_name, category in BLIMP_ADVANCED_SUBSETS.items():
        pairs = load_blimp_subset(subset_name, max_samples=max_samples)
        pref_res = evaluate_pairwise_preference(model, tokenizer, pairs)

        results[subset_name] = {
            "category": category,
            "accuracy": pref_res["accuracy"],
            "avg_margin": pref_res["avg_margin"],
            "cohens_d": pref_res["cohens_d"],
            "count": pref_res["total_pairs"],
        }
        total_correct += pref_res["correct_count"]
        total_pairs += pref_res["total_pairs"]
        print(f"  BLiMP [{category} - {subset_name}]: Acc = {pref_res['accuracy']:.3f}, Cohen's d = {pref_res['cohens_d']:.2f}")

    macro_acc = float(np.mean([r["accuracy"] for r in results.values()]))
    overall_acc = total_correct / max(1, total_pairs)
    print(f"BLiMP Battery Summary: Macro Acc = {macro_acc:.3f}, Overall Acc = {overall_acc:.3f}")

    return {
        "task": "blimp_battery",
        "macro_accuracy": macro_acc,
        "overall_accuracy": overall_acc,
        "subsets": results,
    }


def run_stsb_evaluation(model, tokenizer, max_samples: int = 100) -> Dict[str, Any]:
    print("\n--- Running STS-B Semantic Textual Similarity Across 27 Layers ---")
    stsb_items = load_stsb(max_samples=max_samples)
    res = evaluate_stsb_across_layers(model, tokenizer, stsb_items)
    print(f"STS-B Peak Spearman rho = {res['peak_rho']:.4f} at Layer {res['best_layer']} (out of 26)")
    return {
        "task": "stsb_layerwise",
        **res,
    }


def run_wic_evaluation(model, tokenizer, max_samples: int = 100) -> Dict[str, Any]:
    print("\n--- Running SuperGLUE WiC Polysemy Disambiguation Across 27 Layers ---")
    wic_items = load_wic(max_samples=max_samples)
    res = evaluate_wic_across_layers(model, tokenizer, wic_items)
    print(f"WiC Polysemy Peak Divergence = {res['max_divergence']:.4f} at Layer {res['peak_divergence_layer']}")
    return {
        "task": "wic_polysemy",
        **res,
    }


def run_hellaswag_evaluation(model, tokenizer, max_samples: int = 50) -> Dict[str, Any]:
    print("\n--- Running HellaSwag Commonsense Physical Plausibility ---")
    items = load_hellaswag(max_samples=max_samples)
    device = next(model.parameters()).device

    correct = 0
    total = 0
    by_category: Dict[str, Dict[str, int]] = {}

    for item in items:
        ctx = item["ctx"]
        endings = item["endings"]
        gold_label = item["label"]
        cat = item["activity_label"]

        # Compute log P(ending | ctx) for each ending
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=True)
        ending_logprobs = []

        for end_text in endings:
            end_ids = tokenizer.encode(" " + end_text.strip(), add_special_tokens=False)
            joint_ids = ctx_ids + end_ids
            input_tensor = torch.tensor([joint_ids], device=device)

            with torch.no_grad():
                out = model(input_ids=input_tensor)
                logits = out.logits[0]
                lp = torch.nn.functional.log_softmax(logits, dim=-1)

            # Sum logprobs of ending tokens
            total_lp = 0.0
            ctx_len = len(ctx_ids)
            for i, tok in enumerate(end_ids):
                total_lp += lp[ctx_len - 1 + i, tok].item()
            ending_logprobs.append(total_lp)

        pred_label = int(np.argmax(ending_logprobs))
        is_correct = (pred_label == gold_label)
        if is_correct:
            correct += 1
        total += 1

        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if is_correct:
            by_category[cat]["correct"] += 1

    overall_acc = correct / max(1, total)
    print(f"HellaSwag 4-way Forced Choice Accuracy = {overall_acc:.3f} ({correct}/{total})")

    return {
        "task": "hellaswag",
        "accuracy": float(overall_acc),
        "total": total,
        "correct": correct,
        "by_category": {k: v["correct"] / max(1, v["total"]) for k, v in by_category.items()},
    }


def run_chomsky_evaluation(model, tokenizer, max_samples: int = 100) -> Dict[str, Any]:
    print("\n--- Running Colorless-Green Chomskyan Dissociation Suite ---")
    pairs = generate_chomsky_pairs(num_samples=max_samples)

    ppl_by_type: Dict[str, List[float]] = {"A": [], "B": [], "C": [], "D": []}
    surprisal_samples: List[Dict[str, Any]] = []

    for item in pairs:
        sent = item["sentence"]
        t = item["type"]
        ppl = sentence_perplexity(model, tokenizer, sent)
        ppl_by_type[t].append(ppl)

        # Record detailed surprisal profiles for a subset
        if len(surprisal_samples) < 8:
            surp = compute_word_surprisal(model, tokenizer, sent)
            surprisal_samples.append({
                "type": t,
                "sentence": sent,
                "surprisal_profile": surp,
            })

    mean_ppl = {t: float(np.mean(vals)) if vals else 0.0 for t, vals in ppl_by_type.items()}
    median_ppl = {t: float(np.median(vals)) if vals else 0.0 for t, vals in ppl_by_type.items()}

    print(f"Chomskyan Perplexity Results:")
    print(f"  Type A (Syntax OK, Semantics Bad):  Mean PPL = {mean_ppl['A']:.2f}")
    print(f"  Type B (Syntax Bad, Semantics OK):  Mean PPL = {mean_ppl['B']:.2f}")
    print(f"  Type C (Syntax OK, Semantics OK):   Mean PPL = {mean_ppl['C']:.2f}")
    print(f"  Type D (Syntax Bad, Semantics Bad): Mean PPL = {mean_ppl['D']:.2f}")

    return {
        "task": "chomsky_dissociation",
        "mean_perplexity": mean_ppl,
        "median_perplexity": median_ppl,
        "surprisal_samples": surprisal_samples,
    }


def run_rsa_evaluation(model, tokenizer, max_samples: int = 24) -> Dict[str, Any]:
    print("\n--- Computing 27 x 27 Inter-Layer Representational Similarity (RSA) ---")
    chomsky_items = generate_chomsky_pairs(num_samples=max_samples)
    probe_sentences = [it["sentence"] for it in chomsky_items[:max_samples]]

    rsa_matrix = compute_layer_rsa_matrix(model, tokenizer, probe_sentences)
    print(f"RSA Matrix successfully computed: shape = {rsa_matrix.shape}")

    return {
        "task": "layer_rsa",
        "rsa_matrix": rsa_matrix.tolist(),
    }


def run_causal_dissociation(model, tokenizer, max_samples: int = 50) -> Dict[str, Any]:
    print("\n=== Running Causal Dissociation under Ablation Surgery ===")
    causal_results = []

    # Probe 1 subset of BLiMP (Island constraints) and STS-B under surgery
    blimp_probe = load_blimp_subset("coordinate_structure_constraint_object_extraction", max_samples=max_samples)
    stsb_probe = load_stsb(max_samples=max_samples)

    for cond in CAUSAL_CONDITIONS:
        label = cond["label"]
        spec: SurgerySpec = cond["spec"]
        print(f"\nCondition: {label}")

        with apply_surgery(model, spec):
            b_res = evaluate_pairwise_preference(model, tokenizer, blimp_probe)
            s_res = evaluate_stsb_across_layers(model, tokenizer, stsb_probe)

        rec = {
            "condition": label,
            "surgery_type": spec.surgery_type,
            "severity": spec.severity,
            "position": spec.position,
            "head_mode": spec.head_mode,
            "blimp_syntax_accuracy": b_res["accuracy"],
            "stsb_peak_semantic_rho": s_res["peak_rho"],
            "stsb_best_layer": s_res["best_layer"],
        }
        print(f"  -> BLiMP Syntax Acc = {b_res['accuracy']:.3f} | STS-B Peak Semantic rho = {s_res['peak_rho']:.3f}")
        causal_results.append(rec)

    return {
        "task": "causal_dissociation",
        "conditions": causal_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Gemma-2 Syntax vs. Semantics Experiment Pipeline")
    parser.add_argument("--model-id", type=str, default="google/gemma-2-2b")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "layer_probes", "causal_dissociation", "blimp", "stsb", "wic", "hellaswag", "chomsky", "rsa"])
    parser.add_argument("--max-samples", type=int, default=50, help="Sample limit per dataset for fast execution")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save Parquet & JSON results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model, tokenizer, telemetry = load_backbone(model_id=args.model_id, token=args.token)

    all_outputs = {}
    exp = args.experiment

    if exp in ("all", "layer_probes", "blimp"):
        all_outputs["blimp"] = run_blimp_evaluation(model, tokenizer, max_samples=args.max_samples)

    if exp in ("all", "layer_probes", "stsb"):
        all_outputs["stsb"] = run_stsb_evaluation(model, tokenizer, max_samples=args.max_samples)

    if exp in ("all", "layer_probes", "wic"):
        all_outputs["wic"] = run_wic_evaluation(model, tokenizer, max_samples=args.max_samples)

    if exp in ("all", "layer_probes", "hellaswag"):
        all_outputs["hellaswag"] = run_hellaswag_evaluation(model, tokenizer, max_samples=args.max_samples)

    if exp in ("all", "layer_probes", "chomsky"):
        all_outputs["chomsky"] = run_chomsky_evaluation(model, tokenizer, max_samples=args.max_samples)

    if exp in ("all", "layer_probes", "rsa"):
        all_outputs["rsa"] = run_rsa_evaluation(model, tokenizer, max_samples=min(24, args.max_samples))

    if exp in ("all", "causal_dissociation"):
        all_outputs["causal_dissociation"] = run_causal_dissociation(model, tokenizer, max_samples=args.max_samples)

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.outdir, f"synsem_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2)

    print(f"\n[pipeline] Successfully completed all requested evaluations!")
    print(f"[pipeline] JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
