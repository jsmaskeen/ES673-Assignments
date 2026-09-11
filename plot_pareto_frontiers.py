from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PARADIGM_METADATA: list[tuple[str, str, str]] = [
    ("backprop", "Backprop (MLP)", "#1e3a8a"),
    ("supervised_ff_margin", "Sup FF: margin", "#2563eb"),
    ("supervised_ff_squared", "Sup FF: squared", "#3b82f6"),
    ("supervised_ff_sparse", "Sup FF: sparse", "#60a5fa"),
    ("supervised_ff_spectral", "Sup FF: spectral", "#0284c7"),
    ("supervised_ff_hebbian_wire", "Sup FF: hebbian", "#0ea5e9"),
    ("supervised_ff_negative_squared", "Sup FF: neg_sq", "#94a3b8"),
    ("unsupervised_ff_spectral", "Unsup FF: spectral", "#047857"),
    ("unsupervised_ff_hebbian_wire", "Unsup FF: hebbian", "#065f46"),
    ("unsupervised_ff_sparse", "Unsup FF: sparse", "#059669"),
    ("unsupervised_ff_squared", "Unsup FF: squared", "#10b981"),
    ("unsupervised_ff_margin", "Unsup FF: margin", "#34d399"),
    ("unsupervised_ff_negative_squared", "Unsup FF: neg_sq", "#cbd5e1"),
    ("noprop", "NoProp-DT", "#f59e0b"),
    ("diffusion", "Diffusion Baseline", "#8b5cf6"),
]


def compute_pareto_frontier(costs: list[float], benefits: list[float]) -> list[int]:
    """
    Computes indices of points on the empirical Pareto frontier.
    
    A point i is Pareto-optimal (non-dominated) if there is no point j such that:
    costs[j] <= costs[i] AND benefits[j] >= benefits[i], with at least one strict inequality.
    """
    n = len(costs)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if (costs[j] <= costs[i] and benefits[j] >= benefits[i]) and (
                    costs[j] < costs[i] or benefits[j] > benefits[i]
                ):
                    is_pareto[i] = False
                    break
    return [i for i, p in enumerate(is_pareto) if p]


def extract_benchmark_series(results: dict[str, Any]) -> dict[str, list]:
    """Extracts aligned lists of names, accuracies, energies, memories, powers, and colors."""
    methods = []
    accuracies = []
    energies = []
    memories = []
    powers = []
    colors = []
    keys = []

    for key, name, color in PARADIGM_METADATA:
        acc_k = f"{key}_accuracy"
        en_k = f"{key}_energy_joules"
        mem_k = f"{key}_peak_memory_mb"
        pow_k = f"{key}_avg_power_w"
        time_k = f"{key}_time_s"

        if acc_k in results:
            methods.append(name)
            accuracies.append(results[acc_k] * 100.0)
            energies.append(results.get(en_k, results.get(time_k, 1.0) * 20.0))
            memories.append(results.get(mem_k, 15.0))
            powers.append(results.get(pow_k, 0.0))
            colors.append(color)
            keys.append(key)
        else:
            legacy_key = (
                key.replace("supervised_", "").replace("_squared", "")
                if "unsupervised" in key
                else key.replace("supervised_", "")
            )
            if f"{legacy_key}_accuracy" in results:
                methods.append(name)
                accuracies.append(results[f"{legacy_key}_accuracy"] * 100.0)
                energies.append(results.get(f"{legacy_key}_energy_joules", results.get(f"{legacy_key}_time_s", 1.0) * 20.0))
                memories.append(results.get(f"{legacy_key}_peak_memory_mb", 15.0))
                powers.append(results.get(f"{legacy_key}_avg_power_w", 0.0))
                colors.append(color)
                keys.append(key)

    return {
        "keys": keys,
        "methods": methods,
        "accuracies": accuracies,
        "energies": energies,
        "memories": memories,
        "powers": powers,
        "colors": colors,
    }


def draw_pareto_step_line(
    ax: plt.Axes,
    costs: list[float],
    benefits: list[float],
    pareto_indices: list[int],
    color: str = "#dc2626",
    label: str = "Pareto Frontier",
):
    """Draws a stepped Pareto frontier boundary line connecting non-dominated points."""
    if not pareto_indices:
        return

    sorted_pts = sorted([(costs[idx], benefits[idx]) for idx in pareto_indices], key=lambda pt: pt[0])
    px = [pt[0] for pt in sorted_pts]
    py = [pt[1] for pt in sorted_pts]

    # Stepped trade-off line
    ax.step(px, py, where="post", color=color, linestyle="--", linewidth=1.8, alpha=0.85, label=label, zorder=4)

    # Highlight Pareto optimal points
    ax.scatter(
        px,
        py,
        s=180,
        facecolors="none",
        edgecolors=color,
        linewidths=2.0,
        linestyle="-",
        zorder=6,
    )


def plot_energy_and_memory_comparison(
    results: dict[str, Any],
    output_path: str | Path,
    title_suffix: str = "",
) -> tuple[list[str], list[str]]:
    """
    Renders a comprehensive 2x2 publication figure comparing:
    1. Row 1, Left: Total Training Energy (Joules) [Bar Chart]
    2. Row 1, Right: Peak Allocated VRAM (MB) [Bar Chart]
    3. Row 2, Left: Energy-Accuracy Pareto Frontier (Test Accuracy % vs Energy Joules) [Scatter + Step Line]
    4. Row 2, Right: Power-Accuracy Pareto Frontier (Test Accuracy % vs Power W) [Scatter + Step Line]

    Returns:
        tuple of (pareto_optimal_energy_models, pareto_optimal_power_models)
    """
    data = extract_benchmark_series(results)
    methods = data["methods"]
    accuracies = data["accuracies"]
    energies = data["energies"]
    memories = data["memories"]
    powers = data["powers"]
    bar_colors = data["colors"]
    n_bars = len(methods)

    if n_bars == 0:
        print("[Plot] No valid benchmark results found to plot.")
        return [], []

    energy_pareto_idx = compute_pareto_frontier(energies, accuracies)
    power_pareto_idx = compute_pareto_frontier(powers, accuracies)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    bars1 = axes[0, 0].bar(
        range(n_bars),
        energies,
        color=bar_colors,
        alpha=0.90,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0, 0].set_ylabel("Energy Consumed (Joules)", fontsize=11, fontweight="bold")
    axes[0, 0].set_title(
        "Total Training Energy Consumption (Joules)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    max_en = max(energies) if energies else 1.0
    for bar in bars1:
        yval = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_en * 0.015,
            f"{yval:.1f} J",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    axes[0, 0].set_xticks(range(n_bars))
    axes[0, 0].set_xticklabels(methods, ha="right", fontsize=8.5, rotation=30)
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0, 0].set_ylim(0, max_en * 1.15)

    bars2 = axes[0, 1].bar(
        range(n_bars),
        memories,
        color=bar_colors,
        alpha=0.90,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0, 1].set_ylabel("Peak VRAM (MB)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title(
        "Peak Allocated Memory Footprint (MB)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    max_mem = max(memories) if memories else 1.0
    for bar in bars2:
        yval = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_mem * 0.015,
            f"{yval:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    axes[0, 1].set_xticks(range(n_bars))
    axes[0, 1].set_xticklabels(methods, ha="right", fontsize=8.5, rotation=30)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0, 1].set_ylim(0, max_mem * 1.15)

    axes[1, 0].set_title(
        "Energy-Accuracy Pareto Frontier (Test Accuracy % vs Energy Joules)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    for i, (m, en, acc, col) in enumerate(zip(methods, energies, accuracies, bar_colors)):
        is_opt = i in energy_pareto_idx
        axes[1, 0].scatter(
            en,
            acc,
            color=col,
            s=140 if is_opt else 100,
            edgecolor="black" if not is_opt else "#dc2626",
            linewidth=2.0 if is_opt else 1.0,
            zorder=5,
        )
        offset_y = 1.4 if i % 2 == 0 else -1.8
        offset_x = 0.5 if i % 2 == 0 else -0.5
        axes[1, 0].annotate(
            m,
            (en + offset_x, acc + offset_y),
            fontsize=8,
            fontweight="bold" if is_opt else "normal",
            alpha=0.95 if is_opt else 0.80,
        )

    draw_pareto_step_line(
        axes[1, 0],
        energies,
        accuracies,
        energy_pareto_idx,
        color="#dc2626",
        label="Pareto Optimal Boundary (Energy)",
    )
    axes[1, 0].set_xlabel("Total Energy Consumed (Joules)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="lower right", fontsize=8.5)
    axes[1, 0].set_ylim(0, 105)

    axes[1, 1].set_title(
        "Power-Accuracy Pareto Frontier (Test Accuracy % vs Power W)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    for i, (m, pw, acc, col) in enumerate(zip(methods, powers, accuracies, bar_colors)):
        is_opt = i in power_pareto_idx
        axes[1, 1].scatter(
            pw,
            acc,
            color=col,
            s=140 if is_opt else 100,
            edgecolor="black" if not is_opt else "#0284c7",
            linewidth=2.0 if is_opt else 1.0,
            zorder=5,
        )
        offset_y = 1.4 if i % 2 == 0 else -1.8
        offset_x = 0.08 if i % 2 == 0 else -0.08
        axes[1, 1].annotate(
            m,
            (pw + offset_x, acc + offset_y),
            fontsize=8,
            fontweight="bold" if is_opt else "normal",
            alpha=0.95 if is_opt else 0.80,
        )

    draw_pareto_step_line(
        axes[1, 1],
        powers,
        accuracies,
        power_pareto_idx,
        color="#0284c7",
        label="Pareto Optimal Boundary (Power)",
    )
    axes[1, 1].set_xlabel("Average Power Consumed (Watts)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="lower right", fontsize=8.5)
    axes[1, 1].set_ylim(0, 105)

    # Super Title
    full_title = "Hardware Profiling & Pareto Frontier Analysis"
    if title_suffix:
        full_title += f" ({title_suffix})"
    fig.suptitle(full_title, fontsize=14, fontweight="bold", y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved energy & power Pareto analysis to {output_path}")

    opt_energy_names = [methods[i] for i in energy_pareto_idx]
    opt_power_names = [methods[i] for i in power_pareto_idx]
    return opt_energy_names, opt_power_names


def main():
    parser = argparse.ArgumentParser(
        description="Extract and plot Energy and Power Pareto Frontiers from benchmark results.json"
    )
    parser.add_argument(
        "--results",
        "-r",
        type=str,
        default="results.json",
        help="Path to results JSON file (default: results.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="plots/energy_and_memory_comparison.png",
        help="Path to save output figure (default: plots/energy_and_memory_comparison.png)",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="",
        help="Optional title suffix (e.g. 'Iso-Parameter' or 'Optimised')",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[Pareto] Loading benchmark results from {results_path}...")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Render figure and compute frontiers
    opt_energy, opt_power = plot_energy_and_memory_comparison(
        results=results,
        output_path=args.output,
        title_suffix=args.title,
    )

    # Print summary table
    data = extract_benchmark_series(results)
    methods = data["methods"]
    accuracies = data["accuracies"]
    energies = data["energies"]
    memories = data["memories"]
    powers = data["powers"]

    print("\n" + "=" * 92)
    print(f"{'Paradigm / Model':<28} {'Acc (%)':>9} {'Energy (J)':>12} {'Peak VRAM':>12} {'Power (W)':>11} {'Pareto':>14}")
    print("=" * 92)

    for idx, name in enumerate(methods):
        is_en_opt = name in opt_energy
        is_pw_opt = name in opt_power
        tags = []
        if is_en_opt:
            tags.append("Energy")
        if is_pw_opt:
            tags.append("Power")
        status = "+".join(tags) if tags else "Dominated"

        print(
            f"{name:<28} {accuracies[idx]:>8.2f}% {energies[idx]:>10.2f} J "
            f"{memories[idx]:>9.2f} MB {powers[idx]:>9.2f} W  {status:>14}"
        )

    print("=" * 92)
    print(f"\n[Energy Pareto Frontier]: {', '.join(opt_energy)}")
    print(f"[Power Pareto Frontier]: {', '.join(opt_power)}")
    print(f"[Plot Generated]: {args.output}\n")


if __name__ == "__main__":
    main()
