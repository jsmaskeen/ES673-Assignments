import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .collate import load_all_records

# Apply publication aesthetic
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def prepare_with_baseline(df: pd.DataFrame, surgery_types: list = None) -> pd.DataFrame:
    """
    Synthesizes pristine baseline (alpha=0.0) records into active surgery types,
    ensuring continuous degradation curves originating from the identical pristine baseline.
    """
    base_rows = df[df["surgery_type"] == "none"].copy()
    if base_rows.empty:
        return df
    if surgery_types is None:
        surgery_types = [st for st in df["surgery_type"].unique() if st not in ("none",)]

    parts = [df[df["surgery_type"] != "none"]]
    for st in surgery_types:
        b = base_rows.copy()
        b["surgery_type"] = st
        b["severity"] = 0.0
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def plot_waterfall_degradation(df: pd.DataFrame, out_path: str):
    """
    Figure 1: Multi-Panel Waterfall Degradation Curves.
    Plots WikiText-2 PPL, IMDB Valence Accuracy, and Garden-Path Ratio vs Severity.
    """
    df_clean = prepare_with_baseline(df)
    active_types = [t for t in df_clean["surgery_type"].unique() if t != "none"]
    palette = sns.color_palette("deep", len(active_types))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Fluency (Perplexity)
    df_flu = df_clean[df_clean["task"] == "fluency"]
    if not df_flu.empty and "metric_perplexity" in df_flu.columns:
        sns.lineplot(
            data=df_flu,
            x="severity",
            y="metric_perplexity",
            hue="surgery_type",
            style="surgery_type",
            markers=True,
            dashes=False,
            palette=palette,
            ax=axes[0],
            linewidth=2.5,
        )
        axes[0].set_yscale("log")
        axes[0].set_title("A. General Fluency (WikiText-2)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Ablation Severity (Fraction Zeroed)", fontweight="bold")
        axes[0].set_ylabel("Perplexity (Log Scale)", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "No Fluency Data", ha="center")

    # Panel B: Valence Accuracy (IMDB)
    df_val = df_clean[df_clean["task"] == "valence"]
    if not df_val.empty and "metric_accuracy" in df_val.columns:
        sns.lineplot(
            data=df_val,
            x="severity",
            y="metric_accuracy",
            hue="surgery_type",
            style="surgery_type",
            markers=True,
            dashes=False,
            palette=palette,
            ax=axes[1],
            linewidth=2.5,
        )
        axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (50%)")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_title("B. Semantic Sentiment (IMDB)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Ablation Severity (Fraction Zeroed)", fontweight="bold")
        axes[1].set_ylabel("Classification Accuracy", fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "No Valence Data", ha="center")

    # Panel C: Syntactic Parsing (Garden-Path Ratio)
    df_syn = df_clean[df_clean["task"] == "structural"]
    if not df_syn.empty and "metric_garden_path_ratio" in df_syn.columns:
        sns.lineplot(
            data=df_syn,
            x="severity",
            y="metric_garden_path_ratio",
            hue="surgery_type",
            style="surgery_type",
            markers=True,
            dashes=False,
            palette=palette,
            ax=axes[2],
            linewidth=2.5,
        )
        axes[2].axhline(1.0, color="red", linestyle=":", alpha=0.8, label="Parity Threshold (1.0)")
        axes[2].set_title("C. Syntactic Parsing (Garden-Path Ratio)", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Ablation Severity (Fraction Zeroed)", fontweight="bold")
        axes[2].set_ylabel("Garden-Path Penalty Ratio", fontweight="bold")
    else:
        axes[2].text(0.5, 0.5, "No Structural Data", ha="center")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Cross-Task Performance Degradation under Ablation Surgery", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 1: {out_path}")


def plot_gqa_sensitivity(df: pd.DataFrame, out_path: str):
    """
    Figure 2: GQA Architectural Sensitivity (Q-only vs KV-group vs Full-head).
    """
    df_heads = df[df["surgery_type"] == "heads"]
    if df_heads.empty:
        print("[visualize] Skip Figure 2: No head ablation data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = sns.color_palette("Set2", 3)

    # Panel A: Fluency by GQA Head Mode
    df_flu = df_heads[df_heads["task"] == "fluency"]
    if not df_flu.empty and "metric_perplexity" in df_flu.columns:
        sns.lineplot(
            data=df_flu,
            x="severity",
            y="metric_perplexity",
            hue="head_mode",
            style="head_mode",
            markers=True,
            palette=palette,
            ax=axes[0],
            linewidth=2.5,
        )
        axes[0].set_yscale("log")
        axes[0].set_title("Fluency Impact (PPL) by GQA Mode", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Head Severity (Fraction)", fontweight="bold")
        axes[0].set_ylabel("Perplexity (Log Scale)", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "No Fluency Heads Data", ha="center")

    # Panel B: Valence Accuracy by GQA Head Mode
    df_val = df_heads[df_heads["task"] == "valence"]
    if not df_val.empty and "metric_accuracy" in df_val.columns:
        sns.lineplot(
            data=df_val,
            x="severity",
            y="metric_accuracy",
            hue="head_mode",
            style="head_mode",
            markers=True,
            palette=palette,
            ax=axes[1],
            linewidth=2.5,
        )
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
        axes[1].set_title("Semantic Valence Accuracy by GQA Mode", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Head Severity (Fraction)", fontweight="bold")
        axes[1].set_ylabel("Classification Accuracy", fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "No Valence Heads Data", ha="center")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Grouped Query Attention (GQA) in Gemma-2-2B", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 2: {out_path}")


def plot_position_resilience_profile(df: pd.DataFrame, out_path: str):
    """
    Figure 3: Position-Aware Resilience Comparison (Early vs Mid vs Late vs Random).
    Evaluates depth resilience across all swept severity levels [0.10, 0.25, 0.50, 0.75].
    """
    df_pos = df[(df["surgery_type"] == "layers") & (df["component"] == "block")]
    if df_pos.empty or "position" not in df_pos.columns:
        print("[visualize] Skip Figure 3: No position layer data found.")
        return

    eval_sevs = [s for s in [0.10, 0.25, 0.50, 0.75] if s in df_pos["severity"].values]
    if not eval_sevs:
        eval_sevs = sorted([s for s in df_pos["severity"].unique() if s > 0.0])

    df_sub = df_pos[df_pos["severity"].isin(eval_sevs)].copy()
    df_sub["Severity Level"] = df_sub["severity"].apply(lambda s: f"{s:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    palette = sns.color_palette("mako", len(eval_sevs))
    order = ["early", "middle", "late", "random"]

    # 1. Fluency PPL
    df_f = df_sub[df_sub["task"] == "fluency"]
    if not df_f.empty and "metric_perplexity" in df_f.columns:
        sns.barplot(data=df_f, x="position", y="metric_perplexity", hue="Severity Level", order=order, palette=palette, ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_title("A. Fluency Perplexity across Cortical Depth", fontweight="bold", fontsize=13)
        axes[0].set_xlabel("Layer Depth Position", fontweight="bold")
        axes[0].set_ylabel("Perplexity (Log Scale)", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "No Position Fluency Data", ha="center")

    # 2. Valence Accuracy
    df_v = df_sub[df_sub["task"] == "valence"]
    if not df_v.empty and "metric_accuracy" in df_v.columns:
        sns.barplot(data=df_v, x="position", y="metric_accuracy", hue="Severity Level", order=order, palette=palette, ax=axes[1])
        axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (50%)")
        axes[1].set_title("B. Semantic Valence Accuracy across Depth", fontweight="bold", fontsize=13)
        axes[1].set_xlabel("Layer Depth Position", fontweight="bold")
        axes[1].set_ylabel("Classification Accuracy", fontweight="bold")
        axes[1].set_ylim(0, 1.05)
    else:
        axes[1].text(0.5, 0.5, "No Position Valence Data", ha="center")

    # 3. Structural Accuracy / Ratio
    df_s = df_sub[df_sub["task"] == "structural"]
    if not df_s.empty and "metric_agreement_accuracy" in df_s.columns:
        sns.barplot(data=df_s, x="position", y="metric_agreement_accuracy", hue="Severity Level", order=order, palette=palette, ax=axes[2])
        axes[2].set_title("C. Syntactic Agreement Accuracy across Depth", fontweight="bold", fontsize=13)
        axes[2].set_xlabel("Layer Depth Position", fontweight="bold")
        axes[2].set_ylabel("Agreement Accuracy", fontweight="bold")
        axes[2].set_ylim(0, 1.05)
    else:
        axes[2].text(0.5, 0.5, "No Position Syntax Data", ha="center")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Position (Depth) Resilience Across Ablation Severities", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 3: {out_path}")


def plot_functional_dissociation(df: pd.DataFrame, out_path: str):
    """
    Figure 4: Syntactic vs. Semantic Functional Dissociation (Scatter Plot).
    Compares degradation on Valence vs. degradation on Syntax across configurations.
    """
    group_cols = ["surgery_type", "severity", "head_mode", "component", "position"]
    valid_cols = [c for c in group_cols if c in df.columns]

    val_sub = df[df["task"] == "valence"].groupby(valid_cols, as_index=False)["metric_accuracy"].mean()
    syn_sub = df[df["task"] == "structural"].groupby(valid_cols, as_index=False)["metric_agreement_accuracy"].mean()

    if val_sub.empty or syn_sub.empty:
        print("[visualize] Skip Figure 4: Incomplete pair data for dissociation plot.")
        return

    merged = pd.merge(val_sub, syn_sub, on=valid_cols, suffixes=("_val", "_syn"))
    if merged.empty:
        print("[visualize] Skip Figure 4: No intersecting configurations.")
        return

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=merged,
        x="metric_accuracy",
        y="metric_agreement_accuracy",
        hue="surgery_type",
        size="severity",
        sizes=(40, 200),
        alpha=0.85,
        palette="deep",
    )

    # Reference boundaries
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Valence Chance")
    plt.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Syntax Chance")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", alpha=0.4, label="Symmetric Parity")

    plt.title("Functional Dissociation: Semantic Valence vs. Syntactic Parsing", fontsize=13, fontweight="bold")
    plt.xlabel("Semantic Accuracy (IMDB Valence)", fontweight="bold")
    plt.ylabel("Syntactic Accuracy (Long-Range Agreement)", fontweight="bold")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 4: {out_path}")


def plot_failure_horizons(df: pd.DataFrame, out_path: str):
    """
    Figure 5: Critical Failure Horizons.
    Panel A: Exact critical severity threshold (alpha*) where semantic accuracy drops to <= 50% chance.
    Panel B: Continuous semantic degradation trajectories highlighting threshold crossings.
    """
    val = df[df["task"] == "valence"]
    if val.empty or "metric_accuracy" not in val.columns:
        print("[visualize] Skip Figure 5: No valence accuracy data.")
        return

    modalities = {
        "Early Layers": val[(val["surgery_type"] == "layers") & (val["position"] == "early") & (val["component"] == "block")],
        "Middle Layers": val[(val["surgery_type"] == "layers") & (val["position"] == "middle") & (val["component"] == "block")],
        "Late Layers": val[(val["surgery_type"] == "layers") & (val["position"] == "late") & (val["component"] == "block")],
        "Q-Heads": val[(val["surgery_type"] == "heads") & (val["head_mode"] == "q_only")],
        "KV-Heads": val[(val["surgery_type"] == "heads") & (val["head_mode"] == "kv_group")],
        "Full Heads": val[(val["surgery_type"] == "heads") & (val["head_mode"] == "full_head")],
        "Isotropic Emb": val[val["surgery_type"] == "emb_mask"],
        "Spectral Emb": val[val["surgery_type"] == "emb_spectral"],
    }

    horizons = {}
    curves = {}
    for name, mdf in modalities.items():
        if mdf.empty:
            continue
        grp = mdf.groupby("severity")["metric_accuracy"].mean().sort_index()
        if 0.0 not in grp.index and not df[df["surgery_type"] == "none"].empty:
            base_acc = df[(df["surgery_type"] == "none") & (df["task"] == "valence")]["metric_accuracy"].mean()
            grp.loc[0.0] = base_acc
            grp = grp.sort_index()
        curves[name] = grp

        sevs = grp.index.values
        accs = grp.values
        below = np.where(accs <= 0.50)[0]
        if len(below) > 0:
            idx = below[0]
            if idx == 0:
                h = float(sevs[0])
            else:
                x0, x1 = float(sevs[idx - 1]), float(sevs[idx])
                y0, y1 = float(accs[idx - 1]), float(accs[idx])
                h = x0 + (0.50 - y0) * (x1 - x0) / (y1 - y0) if (y1 != y0) else x1
            horizons[name] = min(max(h, 0.0), 1.0)
        else:
            horizons[name] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Horizontal Bar Chart of Critical Horizons
    h_df = pd.DataFrame(list(horizons.items()), columns=["Modality", "Failure Threshold (alpha*)"]).sort_values("Failure Threshold (alpha*)")
    colors = []
    for mod in h_df["Modality"]:
        if "Layer" in mod:
            colors.append("#8856a7")  # Purple
        elif "Head" in mod:
            colors.append("#e6550d")  # Orange
        else:
            colors.append("#31a354")  # Green

    bars = axes[0].barh(h_df["Modality"], h_df["Failure Threshold (alpha*)"], color=colors, height=0.6, edgecolor="#333333")
    axes[0].set_xlim(0, 1.15)
    axes[0].axvline(0.5, color="gray", linestyle=":", alpha=0.7, label="Moderate Lesion (50%)")
    axes[0].set_xlabel("Critical Severity Horizon (alpha* to reach <= 50% chance)", fontweight="bold")
    axes[0].set_title("A. Critical Failure Horizons by Architectural Modality", fontweight="bold", fontsize=13)

    for bar in bars:
        w = bar.get_width()
        val_str = f"{w:.2f}" if w < 1.0 else "> 1.0 (Immune)"
        axes[0].text(w + 0.02, bar.get_y() + bar.get_height() / 2, val_str, va="center", fontweight="bold", fontsize=10)

    # Panel B: Degradation Curves
    palette_b = sns.color_palette("tab10", len(curves))
    for i, (name, grp) in enumerate(curves.items()):
        axes[1].plot(grp.index, grp.values, marker="o", linewidth=2.2, label=name, color=palette_b[i])

    axes[1].axhline(0.50, color="red", linestyle="--", alpha=0.8, label="Chance Failure Threshold (50%)")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Ablation Severity (Fraction Zeroed)", fontweight="bold")
    axes[1].set_ylabel("IMDB Semantic Accuracy", fontweight="bold")
    axes[1].set_title("B. Semantic Degradation Trajectories", fontweight="bold", fontsize=13)
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Failure Horizons & Robustness Limits under Targeted Surgery", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 5: {out_path}")


def plot_depth_component_heatmap(df: pd.DataFrame, out_path: str):
    """
    Figure 6: Depth Position vs. Subcomponent Interaction Heatmaps.
    Panel A: Fluency NLL (Generative modeling degradation)
    Panel B: Semantic Valence Accuracy (Affective classification retention)
    """
    df_layers = df[df["surgery_type"] == "layers"]
    if df_layers.empty:
        print("[visualize] Skip Figure 6: No layer ablation data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: Fluency NLL
    flu = df_layers[df_layers["task"] == "fluency"]
    if not flu.empty and "metric_nll" in flu.columns:
        piv_nll = flu.pivot_table(index="position", columns="component", values="metric_nll", aggfunc="mean")
        piv_nll = piv_nll.reindex(["early", "middle", "late", "random"])
        sns.heatmap(piv_nll, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Mean NLL Loss'}, ax=axes[0])
        axes[0].set_title("A. Generative NLL Loss (Lower = Better)", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Cortical Depth Position", fontweight="bold")
        axes[0].set_xlabel("Layer Subcomponent", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "No Fluency Data", ha="center")

    # Panel B: Valence Accuracy
    val = df_layers[df_layers["task"] == "valence"]
    if not val.empty and "metric_accuracy" in val.columns:
        piv_acc = val.pivot_table(index="position", columns="component", values="metric_accuracy", aggfunc="mean")
        piv_acc = piv_acc.reindex(["early", "middle", "late", "random"])
        sns.heatmap(piv_acc, annot=True, fmt=".1%", cmap="Blues_r", vmin=0.45, vmax=0.75, cbar_kws={'label': 'Mean Accuracy'}, ax=axes[1])
        axes[1].set_title("B. Semantic Sentiment Accuracy (Higher = Better)", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Cortical Depth Position", fontweight="bold")
        axes[1].set_xlabel("Layer Subcomponent", fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "No Valence Data", ha="center")

    plt.suptitle("Depth Position vs. Subcomponent Interaction Heatmaps", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 6: {out_path}")


def plot_syntactic_attractor_dynamics(df: pd.DataFrame, out_path: str):
    """
    Figure 7: Psycholinguistic Structural Dynamics: Attractor Immunity & Ambiguity.
    Tracks Agreement Margin and Garden-Path Ratio across severities.
    """
    df_clean = prepare_with_baseline(df)
    df_syn = df_clean[df_clean["task"] == "structural"]
    if df_syn.empty:
        print("[visualize] Skip Figure 7: No structural data found.")
        return

    active_types = [t for t in df_syn["surgery_type"].unique() if t != "none"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    palette = sns.color_palette("tab10", len(active_types))

    # Panel A: Long-Range Agreement Margin
    if "metric_agreement_margin" in df_syn.columns:
        sns.lineplot(
            data=df_syn,
            x="severity",
            y="metric_agreement_margin",
            hue="surgery_type",
            style="surgery_type",
            markers=True,
            palette=palette,
            ax=axes[0],
            linewidth=2.5,
        )
        axes[0].axhline(0.0, color="red", linestyle="--", alpha=0.7, label="Attractor Reversal Threshold (0.0)")
        axes[0].set_title("A. Subject-Verb Agreement Margin (Log-Prob)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Ablation Severity", fontweight="bold")
        axes[0].set_ylabel("Log-Probability Margin (Gold - Attractor)", fontweight="bold")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No Agreement Margin Data", ha="center")

    # Panel B: Garden-Path Disruption Penalty Ratio
    if "metric_garden_path_ratio" in df_syn.columns:
        sns.lineplot(
            data=df_syn,
            x="severity",
            y="metric_garden_path_ratio",
            hue="surgery_type",
            style="surgery_type",
            markers=True,
            palette=palette,
            ax=axes[1],
            linewidth=2.5,
        )
        axes[1].axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Parity Baseline (1.0)")
        axes[1].set_title("B. Garden-Path Disruption Penalty Ratio", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Ablation Severity", fontweight="bold")
        axes[1].set_ylabel("Surprisal Ratio (Ambiguous / Control)", fontweight="bold")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No Garden Path Data", ha="center")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Structural Disruption under Causal Surgery", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 7: {out_path}")


def plot_embedding_percolation(df: pd.DataFrame, out_path: str):
    """
    Figure 8: Embedding Subspace Percolation Horizon.
    Dual-axis plot: Left y-axis WikiText-2 PPL (log scale), Right y-axis Agreement Margin
    across all 8 severities (0.0 to 1.0), showing the critical percolation threshold.
    """
    emb_df = df[df["surgery_type"] == "emb_mask"]
    if emb_df.empty:
        print("[visualize] Skip Figure 8: No embedding mask data found.")
        return

    flu = emb_df[emb_df["task"] == "fluency"].groupby("severity")["metric_perplexity"].mean()
    syn = emb_df[emb_df["task"] == "structural"].groupby("severity")["metric_agreement_margin"].mean()

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(flu.index, flu.values, "b-o", linewidth=2.5, markersize=7, label="Fluency Perplexity (Log Scale)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Embedding Channels Masked (Fraction)", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Perplexity (Log Scale)", color="blue", fontweight="bold", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="blue")

    l2 = ax2.plot(syn.index, syn.values, "r-s", linewidth=2.5, markersize=7, label="Syntactic Agreement Margin")
    ax2.axhline(0.0, color="gray", linestyle="--", alpha=0.6, label="Attractor Reversal Threshold")
    ax2.set_ylabel("Agreement Margin (Logits)", color="red", fontweight="bold", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="red")

    # Critical percolation shaded window
    ax1.axvspan(0.20, 0.50, color="orange", alpha=0.15, label="Critical Percolation Window (20-50%)")

    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", framealpha=0.9)

    plt.title("Embedding Subspace Percolation Horizon & Phase Transition", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 8: {out_path}")


def plot_perturbation_modalities(df: pd.DataFrame, out_path: str):
    """
    Figure 9: Perturbation Modalities: Zero vs. Gaussian Noise vs. Mean Clamp.
    Strictly isolates the layer/head perturbation experiment across its aligned severities [0.10, 0.25, 0.50].
    """
    if "perturbation_type" not in df.columns:
        return
    valid_modes = [m for m in df["perturbation_type"].dropna().unique() if m not in ("none", None)]
    if len(valid_modes) < 2:
        return

    # Filter strictly to the perturbation sweep (exclude emb_spectral)
    df_pert = df[
        (df["perturbation_type"].isin(["zero", "gaussian_noise", "mean_clamp"]))
        & (df["severity"].isin([0.10, 0.25, 0.50]))
        & (df["surgery_type"] != "emb_spectral")
    ].copy()

    if df_pert.empty:
        return

    df_pert["Severity Level"] = df_pert["severity"].apply(lambda s: f"{s:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = sns.color_palette("Set1", len(df_pert["perturbation_type"].unique()))

    # Panel A: Fluency
    df_flu = df_pert[df_pert["task"] == "fluency"]
    if not df_flu.empty and "metric_perplexity" in df_flu.columns:
        sns.barplot(data=df_flu, x="Severity Level", y="metric_perplexity", hue="perturbation_type", palette=palette, ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_title("A. Fluency Perplexity by Perturbation Modality", fontweight="bold", fontsize=13)
        axes[0].set_xlabel("Ablation Severity", fontweight="bold")
        axes[0].set_ylabel("Perplexity (Log Scale)", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "No Fluency Data", ha="center")

    # Panel B: Valence
    df_val = df_pert[df_pert["task"] == "valence"]
    if not df_val.empty and "metric_accuracy" in df_val.columns:
        sns.barplot(data=df_val, x="Severity Level", y="metric_accuracy", hue="perturbation_type", palette=palette, ax=axes[1])
        axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (50%)")
        axes[1].set_title("B. Valence Accuracy by Perturbation Modality", fontweight="bold", fontsize=13)
        axes[1].set_xlabel("Ablation Severity", fontweight="bold")
        axes[1].set_ylabel("Classification Accuracy", fontweight="bold")
        axes[1].set_ylim(0, 1.05)
    else:
        axes[1].text(0.5, 0.5, "No Valence Data", ha="center")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Perturbation Modality Resilience: Silence vs. Synaptic Jitter vs. Clamping", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 9: {out_path}")


def plot_spectral_vs_isotropic(df: pd.DataFrame, out_path: str):
    """
    Figure 10: Spectral PCA Subspace vs. Isotropic Channel Masking.
    """
    types = df["surgery_type"].unique()
    if "emb_spectral" not in types:
        return

    # Prepend baseline so both curves originate continuously at alpha=0.0
    df_sub = prepare_with_baseline(
        df[df["surgery_type"].isin(["emb_mask", "emb_spectral", "none"])],
        ["emb_mask", "emb_spectral"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = sns.color_palette("Set2", 2)

    df_flu = df_sub[df_sub["task"] == "fluency"]
    if not df_flu.empty and "metric_perplexity" in df_flu.columns:
        sns.lineplot(
            data=df_flu,
            x="severity",
            y="metric_perplexity",
            hue="surgery_type",
            style="surgery_type",
            markers=True,
            palette=palette,
            ax=axes[0],
            linewidth=2.5,
        )
        axes[0].set_yscale("log")
        axes[0].set_title("A. Fluency Perplexity: Spectral vs. Isotropic", fontweight="bold", fontsize=13)
        axes[0].set_xlabel("Severity (Fraction of Subspace Ablated)", fontweight="bold")
        axes[0].set_ylabel("Perplexity (Log Scale)", fontweight="bold")

    df_val = df_sub[df_sub["task"] == "valence"]
    if not df_val.empty and "metric_accuracy" in df_val.columns:
        sns.lineplot(
            data=df_val,
            x="severity",
            y="metric_accuracy",
            hue="surgery_type",
            style="surgery_type",
            markers=True,
            palette=palette,
            ax=axes[1],
            linewidth=2.5,
        )
        axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (50%)")
        axes[1].set_title("B. Valence Accuracy: Spectral vs. Isotropic", fontweight="bold", fontsize=13)
        axes[1].set_xlabel("Severity (Fraction of Subspace Ablated)", fontweight="bold")
        axes[1].set_ylabel("Classification Accuracy", fontweight="bold")
        axes[1].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Principal Subspace vs. Isotropic Noise", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Generated Figure 10: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate creative figures for Gemma-2 ablation study")
    parser.add_argument("--input-dir", type=str, default="outputs", help="Directory containing Parquet/JSON runs")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save PNG plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df = load_all_records(args.input_dir)
    except Exception as e:
        print(f"[visualize] Error loading records from {args.input_dir}: {e}")
        print("[visualize] Tip: Run an experiment first using `python -m gemma.orchestrator --experiment experiments.baselines`")
        sys.exit(1)

    print(f"[visualize] Loaded {len(df)} records. Generating creative figures...")

    # Core empirical figures (Figures 1 - 8)
    plot_waterfall_degradation(df, os.path.join(args.output_dir, "fig1_waterfall_degradation.png"))
    plot_gqa_sensitivity(df, os.path.join(args.output_dir, "fig2_gqa_sensitivity.png"))
    plot_position_resilience_profile(df, os.path.join(args.output_dir, "fig3_position_resilience.png"))
    plot_functional_dissociation(df, os.path.join(args.output_dir, "fig4_functional_dissociation.png"))
    plot_failure_horizons(df, os.path.join(args.output_dir, "fig5_failure_horizons.png"))
    plot_depth_component_heatmap(df, os.path.join(args.output_dir, "fig6_depth_component_heatmap.png"))
    plot_syntactic_attractor_dynamics(df, os.path.join(args.output_dir, "fig7_syntactic_attractor_dynamics.png"))
    plot_embedding_percolation(df, os.path.join(args.output_dir, "fig8_embedding_percolation.png"))

    # Extended sweeps figures (Figures 9 & 10)
    plot_perturbation_modalities(df, os.path.join(args.output_dir, "fig9_perturbation_modalities.png"))
    plot_spectral_vs_isotropic(df, os.path.join(args.output_dir, "fig10_spectral_vs_isotropic.png"))

    print(f"[visualize] All available figures generated in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
