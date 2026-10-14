import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
import torch

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from .backbone import load_backbone
from .surgery import SurgerySpec, apply_surgery
from .tasks.fluency import evaluate_fluency
from .tasks.valence import evaluate_valence
from .tasks.structural import evaluate_garden_path


def evaluate_task(task_name: str, model, tokenizer, max_samples: int = 100) -> Dict[str, Any]:
    """Dispatch evaluation to appropriate task module."""
    if task_name == "fluency":
        return evaluate_fluency(model, tokenizer, max_tokens=max_samples * 20)
    elif task_name == "valence":
        return evaluate_valence(model, tokenizer, max_samples=max_samples)
    elif task_name == "structural":
        return evaluate_garden_path(model, tokenizer, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def run_single_ablation(
    model,
    tokenizer,
    spec: SurgerySpec,
    task_name: str,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """Execute a single ablation evaluation run."""
    run_start = time.time()

    with apply_surgery(model, spec):
        task_metrics = evaluate_task(task_name, model, tokenizer, max_samples=max_samples)

    total_runtime = time.time() - run_start

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task_name,
        "surgery_type": spec.surgery_type,
        "severity": float(spec.severity),
        "head_mode": spec.head_mode,
        "component": spec.component,
        "position": spec.position,
        "perturbation_type": getattr(spec, "perturbation_type", "zero"),
        "seed": int(spec.seed),
        "total_runtime_s": float(total_runtime),
    }

    # Flatten task metrics
    for k, v in task_metrics.items():
        if isinstance(v, (int, float, str, bool)):
            record[f"metric_{k}"] = v

    # Record metadata
    for k, v in spec.metadata.items():
        if isinstance(v, (int, float, str, bool)):
            record[f"meta_{k}"] = v
        elif isinstance(v, list):
            record[f"meta_{k}"] = str(v)

    return record


def run_sanity_check(model_id: str = "google/gemma-2-2b", token: str = None):
    """Verify loading, hook installation, and hook detachment."""
    print("\n" + "=" * 60)
    print("RUNNING SANITY CHECK FOR GEMMA-2 SURGERY FRAMEWORK")
    print("=" * 60)

    model, tokenizer, telemetry = load_backbone(model_id=model_id, token=token)
    device = next(model.parameters()).device

    test_prompt = "NeuroAI combines neuroscience and artificial intelligence."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    # 1. Baseline forward
    with torch.no_grad():
        base_logits = model(**inputs).logits

    # 2. Test Q-head surgery hook
    spec_q = SurgerySpec(surgery_type="heads", severity=0.25, head_mode="q_only", seed=42)
    with apply_surgery(model, spec_q):
        with torch.no_grad():
            q_logits = model(**inputs).logits
        diff_q = (base_logits - q_logits).abs().max().item()
        print(f"[Sanity] Q-head hook active: max logit change = {diff_q:.4f}")
        assert diff_q > 1e-4, "Q-head surgery hook did not modify logits!"

    # 3. Verify clean detachment
    with torch.no_grad():
        restored_logits = model(**inputs).logits
    detachment_diff = (base_logits - restored_logits).abs().max().item()
    print(f"[Sanity] Post-detachment diff from baseline = {detachment_diff:.6f}")
    assert detachment_diff < 1e-5, "Hooks failed to detach cleanly!"

    # 4. Test Structural task smoke evaluation
    print("[Sanity] Testing structural task smoke evaluation...")
    struct_metrics = evaluate_garden_path(model, tokenizer, max_samples=2)
    print(f"[Sanity] Structural metrics: {struct_metrics}")

    print("=" * 60)
    print("SANITY CHECK PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Gemma-2 Ablation Orchestrator")
    parser.add_argument("--model-id", type=str, default="google/gemma-2-2b")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--experiment", type=str, default=None, help="Module path to experiment suite (e.g. experiments.baselines)")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save Parquet & JSON results")
    parser.add_argument("--max-samples", type=int, default=100, help="Sample limit per task evaluation")
    parser.add_argument("--sanity-check", action="store_true", help="Run sanity check and exit")
    args = parser.parse_args()

    if args.sanity_check:
        run_sanity_check(model_id=args.model_id, token=args.token)
        return

    os.makedirs(args.outdir, exist_ok=True)

    # Load experiment specifications
    if args.experiment is None:
        print("[Error] Please specify --experiment (e.g., experiments.baselines or experiments.head_sweeps) or --sanity-check")
        sys.exit(1)

    try:
        exp_module = importlib.import_module(args.experiment)
        experiment_specs = exp_module.EXPERIMENTS
    except Exception as e:
        print(f"[Error] Failed to load experiment module '{args.experiment}': {e}")
        sys.exit(1)

    print(f"[Orchestrator] Loaded {len(experiment_specs)} experiment sweeps from {args.experiment}")

    # Load model once for all sweeps
    model, tokenizer, telemetry = load_backbone(model_id=args.model_id, token=args.token)

    results: List[Dict[str, Any]] = []

    for idx, (spec, task_name) in enumerate(experiment_specs):
        print(f"\n[{idx+1}/{len(experiment_specs)}] Running Task: {task_name.upper()} | "
              f"Type: {spec.surgery_type} | Mode: {spec.head_mode} | "
              f"Severity: {spec.severity:.2f} | Pos: {spec.position} | Seed: {spec.seed}")

        record = run_single_ablation(model, tokenizer, spec, task_name, max_samples=args.max_samples)
        results.append(record)

        print(f"  -> Metrics: {[f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' for k, v in record.items() if k.startswith('metric_')]}")

    # Export to Parquet and JSON
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment.split(".")[-1]
    parquet_path = os.path.join(args.outdir, f"{exp_name}_{timestamp}.parquet")
    json_path = os.path.join(args.outdir, f"{exp_name}_{timestamp}.json")

    df = pd.DataFrame(results)
    df.to_parquet(parquet_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Orchestrator] Successfully saved {len(results)} run records to:")
    print(f"  Parquet: {parquet_path}")
    print(f"  JSON:    {json_path}")


if __name__ == "__main__":
    main()
