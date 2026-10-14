import argparse
import glob
import json
import os
from typing import Dict, Any, List

import pandas as pd


def collate_q3_outputs(input_dir: str) -> Dict[str, Any]:
    """Load and merge all synsem_results JSON files from input_dir (chronological order)."""
    files = sorted(glob.glob(os.path.join(input_dir, "synsem_results_*.json")))
    if not files:
        # Fallback to any json
        files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    if not files:
        raise ValueError(f"No results JSON files found in '{input_dir}'.")

    merged: Dict[str, Any] = {}
    print(f"[collate] Found {len(files)} JSON result file(s) in '{input_dir}'")
    for fpath in files:
        print(f"[collate] Merging results from: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged.update(data)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Collate Question 3 results")
    parser.add_argument("--input-dir", type=str, default="outputs", help="Directory containing Q3 JSON outputs")
    parser.add_argument("--output-md", type=str, default="outputs/q3_summary_table.md", help="Markdown summary table path")
    args = parser.parse_args()

    data = collate_q3_outputs(args.input_dir)

    lines = ["# Question 3: Syntax vs. Semantics Comprehensive Summary\n"]

    if "blimp" in data:
        blimp = data["blimp"]
        lines.append(f"## 1. BLiMP Battery (Overall Acc: {blimp['overall_accuracy']:.3f}, Macro Acc: {blimp['macro_accuracy']:.3f})\n")
        rows = []
        for name, sub in blimp.get("subsets", {}).items():
            rows.append({
                "Subset": name,
                "Category": sub["category"],
                "Accuracy": f"{sub['accuracy']:.3f}",
                "Cohen's d": f"{sub['cohens_d']:.2f}",
                "Samples": sub["count"],
            })
        df_b = pd.DataFrame(rows)
        lines.append(df_b.to_markdown(index=False) + "\n")

    if "stsb" in data:
        stsb = data["stsb"]
        lines.append(f"## 2. STS-B Semantic Similarity\n- Peak Spearman $\\rho$: **{stsb['peak_rho']:.4f}** at **Layer {stsb['best_layer']}**\n")

    if "wic" in data:
        wic = data["wic"]
        lines.append(f"## 3. WiC Polysemy Disambiguation\n- Max Polysemy Divergence: **{wic['max_divergence']:.4f}** at **Layer {wic['peak_divergence_layer']}**\n")

    if "chomsky" in data:
        chom = data["chomsky"]
        lines.append("## 4. Colorless-Green Chomskyan Dissociation\n")
        rows_c = [{"Type": k, "Mean PPL": f"{v:.2f}"} for k, v in chom.get("mean_perplexity", {}).items()]
        lines.append(pd.DataFrame(rows_c).to_markdown(index=False) + "\n")

    if "causal_dissociation" in data:
        causal = data["causal_dissociation"]
        lines.append("## 5. Causal Dissociation Under Ablation Surgery\n")
        lines.append(pd.DataFrame(causal.get("conditions", [])).to_markdown(index=False) + "\n")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[collate] Successfully exported markdown summary to: {args.output_md}")


if __name__ == "__main__":
    main()
