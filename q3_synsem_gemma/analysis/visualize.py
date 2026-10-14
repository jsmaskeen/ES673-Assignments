import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .collate import collate_q3_outputs

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_fig1_blimp(data: dict, out_path: str):
    """Figure 1: BLiMP Syntactic Battery Across 10 Advanced Subsets."""
    blimp = data.get("blimp", {})
    subsets = blimp.get("subsets", {})
    if not subsets:
        print("[visualize] Skip Fig 1: No BLiMP data found.")
        return

    df = pd.DataFrame([
        {"Subset": k.replace("_", " ").title(), "Category": v["category"], "Accuracy": v["accuracy"], "Cohen_d": v["cohens_d"]}
        for k, v in subsets.items()
    ]).sort_values("Accuracy", ascending=True)

    plt.figure(figsize=(11, 6))
    palette = sns.color_palette("Set2", len(df["Category"].unique()))
    ax = sns.barplot(data=df, y="Subset", x="Accuracy", hue="Category", palette="deep", dodge=False)

    plt.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance Parity (50%)")
    plt.title("Figure 1: BLiMP Syntactic Battery Performance on Gemma-2-2B", fontsize=13, fontweight="bold")
    plt.xlabel("Pairwise Forced-Choice Accuracy", fontweight="bold")
    plt.xlim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Linguistic Phenomenon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 1: {out_path}")


def plot_fig2_stsb(data: dict, out_path: str):
    """Figure 2: STS-B Semantic Textual Similarity Evolution Across 27 Layers."""
    stsb = data.get("stsb", {})
    rhos = stsb.get("spearman_rhos", [])
    if not rhos:
        print("[visualize] Skip Fig 2: No STS-B layer data found.")
        return

    layers = list(range(len(rhos)))
    best_layer = int(np.argmax(rhos))

    plt.figure(figsize=(10, 5))
    plt.plot(layers, rhos, marker="o", color="#2b5c8f", linewidth=2.5, markersize=6, label="Spearman rho")
    plt.axvline(best_layer, color="#d95f02", linestyle="--", alpha=0.8, label=f"Peak Crystallization: Layer {best_layer} (rho={rhos[best_layer]:.3f})")
    plt.scatter([best_layer], [rhos[best_layer]], color="#d95f02", s=120, zorder=5)

    plt.title("Figure 2: Semantic Similarity Crystallization Across Gemma-2-2B Layers (STS-B)", fontsize=13, fontweight="bold")
    plt.xlabel("Layer Index (0 = Embedding, 1..26 = Transformer Layers)", fontweight="bold")
    plt.ylabel("Spearman Rank Correlation (rho)", fontweight="bold")
    plt.ylim(-0.05, max(0.9, max(rhos) + 0.1))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 2: {out_path}")


def plot_fig3_wic(data: dict, out_path: str):
    """Figure 3: WiC Polysemy Disambiguation Across 27 Layers."""
    wic = data.get("wic", {})
    same = wic.get("same_sense_cosine_by_layer", [])
    diff = wic.get("diff_sense_cosine_by_layer", [])
    if not same or not diff:
        print("[visualize] Skip Fig 3: No WiC layer data found.")
        return

    layers = list(range(len(same)))
    peak_layer = int(np.argmax(wic.get("divergence_by_layer", [0])))

    plt.figure(figsize=(10, 5.5))
    plt.plot(layers, same, marker="s", color="#2ca02c", linewidth=2.2, label="Same Word Sense Context")
    plt.plot(layers, diff, marker="^", color="#d62728", linewidth=2.2, label="Different Word Sense Context")
    plt.axvline(peak_layer, color="purple", linestyle=":", label=f"Max Polysemy Divergence: Layer {peak_layer}")

    plt.title("Figure 3: Contextual Polysemy Resolution Across Layers (SuperGLUE WiC)", fontsize=13, fontweight="bold")
    plt.xlabel("Layer Index (0 = Embedding, 1..26 = Transformer Layers)", fontweight="bold")
    plt.ylabel("Contextual Token Cosine Similarity", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 3: {out_path}")


def plot_fig4_chomsky(data: dict, out_path: str):
    """Figure 4: Colorless-Green Chomskyan Dissociation (Type A vs B vs C vs D)."""
    chomsky = data.get("chomsky", {})
    mean_ppl = chomsky.get("mean_perplexity", {})
    if not mean_ppl:
        print("[visualize] Skip Fig 4: No Chomsky data found.")
        return

    df = pd.DataFrame([
        {"Type": "Type A\n(Syntax OK,\nSemantics Bad)", "PPL": mean_ppl.get("A", 0), "Category": "Chomsky Quadrant"},
        {"Type": "Type B\n(Syntax Bad,\nSemantics OK)", "PPL": mean_ppl.get("B", 0), "Category": "Chomsky Quadrant"},
        {"Type": "Type C\n(Syntax OK,\nSemantics OK)", "PPL": mean_ppl.get("C", 0), "Category": "Control (Natural)"},
        {"Type": "Type D\n(Syntax Bad,\nSemantics Bad)", "PPL": mean_ppl.get("D", 0), "Category": "Negative Control"},
    ])

    plt.figure(figsize=(10, 6))
    palette = ["#386cb0", "#f0027f", "#7fc97f", "#beaed4"]
    ax = sns.barplot(data=df, x="Type", y="PPL", palette=palette)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}", (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom", fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")

    plt.title("Syntax-Semantics Dissociation via Perplexity", fontsize=13, fontweight="bold")
    plt.xlabel("Sentence Quadrant", fontweight="bold")
    plt.ylabel("Sentence Perplexity (PPL)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 4: {out_path}")


def plot_fig5_surprisal(data: dict, out_path: str):
    """Figure 5: Word-by-Word Surprisal Profile and Violation Spikes."""
    chomsky = data.get("chomsky", {})
    samples = chomsky.get("surprisal_samples", [])
    if not samples:
        print("[visualize] Skip Fig 5: No surprisal samples found.")
        return

    fig, axes = plt.subplots(min(2, len(samples)), 2, figsize=(12, 6), sharex=False)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for idx, sample in enumerate(samples[:2]):
        profile = sample.get("surprisal_profile", {})
        tokens = profile.get("tokens", [])
        surp = profile.get("surprisals", [])
        t_label = sample.get("type", "")

        axes[0,idx].step(range(len(tokens)), surp, where="mid", color="#1f77b4", linewidth=2.5)
        axes[0,idx].set_xticks(range(len(tokens)))
        axes[0,idx].set_xticklabels(tokens, rotation=35, ha="right", fontsize=15)
        axes[0,idx].set_ylabel("Surprisal (Bits)", fontweight="bold")
        axes[0,idx].set_title(f"Type {t_label}: \"{sample.get('sentence', '')}\"", fontsize=11, fontweight="bold")
        axes[0,idx].grid(True, linestyle="--", alpha=0.5)
    
    for idx, sample in enumerate(samples[2:4]):
            profile = sample.get("surprisal_profile", {})
            tokens = profile.get("tokens", [])
            surp = profile.get("surprisals", [])
            t_label = sample.get("type", "")
    
            axes[1,idx].step(range(len(tokens)), surp, where="mid", color="#1f77b4", linewidth=2.5)
            axes[1,idx].set_xticks(range(len(tokens)))
            axes[1,idx].set_xticklabels(tokens, rotation=35, ha="right", fontsize=9)
            axes[1,idx].set_ylabel("Surprisal (Bits)", fontweight="bold")
            axes[1,idx].set_title(f"Type {t_label}: \"{sample.get('sentence', '')}\"", fontsize=11, fontweight="bold")
            axes[1,idx].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Word-by-Word Surprisal Spike Profile", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 5: {out_path}")


def plot_fig7_rsa_matrix(data: dict, out_path: str):
    """Figure 7: 27 x 27 Inter-Layer Representational Similarity (RSA) Heatmap."""
    rsa_data = data.get("rsa", {}).get("rsa_matrix", [])
    if not rsa_data:
        print("[visualize] Skip Fig 7: No RSA matrix found.")
        return

    mat = np.array(rsa_data)
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(mat, cmap="magma", vmin=0.0, vmax=1.0, cbar_kws={"label": "Second-Order Spearman rho"})

    plt.title("Figure 7: Gemma-2-2B Inter-Layer Representational Geometry (RSA)", fontsize=13, fontweight="bold")
    plt.xlabel("Layer Index (0 = Embedding, 1..26 = Layers)", fontweight="bold")
    plt.ylabel("Layer Index (0 = Embedding, 1..26 = Layers)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 7: {out_path}")


def plot_fig9_causal_dissociation(data: dict, out_path: str):
    """Figure 9: Causal Dissociation under Ablation Surgery (Syntax vs Semantics)."""
    causal = data.get("causal_dissociation", {}).get("conditions", [])
    if not causal:
        print("[visualize] Skip Fig 9: No causal surgery data found.")
        return

    df = pd.DataFrame(causal)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(df))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df["blimp_syntax_accuracy"], width, label="BLiMP Syntax Accuracy", color="#2b5c8f")
    ax1.set_ylabel("Syntax Accuracy", fontweight="bold", color="#2b5c8f")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, df["stsb_peak_semantic_rho"], width, label="STS-B Semantic rho", color="#d95f02")
    ax2.set_ylabel("Semantic Spearman rho", fontweight="bold", color="#d95f02")
    ax2.set_ylim(0, 1.05)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["condition"], rotation=25, ha="right", fontsize=9, fontweight="bold")
    plt.title("Figure 9: Causal Dissociation: Ablation Surgery on Syntax vs. Semantics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 9: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures for Question 3")
    parser.add_argument("--input-dir", type=str, default="outputs", help="Directory containing Q3 JSON outputs")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save PNG plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        data = collate_q3_outputs(args.input_dir)
    except Exception as e:
        print(f"[visualize] Error reading Q3 outputs: {e}")
        sys.exit(1)

    print(f"[visualize] Generating creative publication figures in '{args.output_dir}'...")

    plot_fig1_blimp(data, os.path.join(args.output_dir, "fig1_blimp_accuracy.png"))
    plot_fig2_stsb(data, os.path.join(args.output_dir, "fig2_stsb_layer_evolution.png"))
    plot_fig3_wic(data, os.path.join(args.output_dir, "fig3_wic_polysemy_resolution.png"))
    plot_fig4_chomsky(data, os.path.join(args.output_dir, "fig4_colorless_green_dissociation.png"))
    plot_fig5_surprisal(data, os.path.join(args.output_dir, "fig5_word_surprisal_profiles.png"))
    plot_fig7_rsa_matrix(data, os.path.join(args.output_dir, "fig7_layer_rsa_matrix.png"))
    plot_fig9_causal_dissociation(data, os.path.join(args.output_dir, "fig9_causal_dissociation.png"))

    print(f"[visualize] All available Question 3 figures successfully created in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
