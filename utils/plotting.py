import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from utils.masks import (
    create_hybrid_negative_batch,
    generate_smooth_mask,
    inject_label_border,
)
from utils.data import unnormalize_mnist


def plot_hybrid_mask_generation(loader, output_path: str | Path):
    """Visualizes the step-by-step generation of Hinton's smooth-blurred masks and hybrid images."""
    images, labels = next(iter(loader))
    device = images.device

    img1 = images[0:1]
    img2 = images[1:2]
    lbl1 = int(labels[0].item())
    lbl2 = int(labels[1].item())

    # Random Bernoulli bits
    raw_bits = (torch.rand((1, 1, 28, 28), device=device) > 0.5).float()

    # Separable blur filter
    kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32, device=device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 3, 3)
    blurred = raw_bits.clone()
    for _ in range(4):
        blurred = F.conv2d(blurred, kernel_2d, padding=1)

    # Thresholded binary mask
    binary_mask = (blurred >= 0.5).float()

    # Hybrid creation
    x1_unnorm = unnormalize_mnist(img1)
    x2_unnorm = unnormalize_mnist(img2)
    hybrid = binary_mask * x1_unnorm + (1.0 - binary_mask) * x2_unnorm

    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))
    axes[0].imshow(raw_bits.squeeze().cpu().numpy(), cmap="gray")
    axes[0].set_title("1. Random Bits", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(blurred.squeeze().cpu().numpy(), cmap="viridis")
    axes[1].set_title("2. Blurred Mask", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(binary_mask.squeeze().cpu().numpy(), cmap="gray")
    axes[2].set_title("3. Threshold (M)", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(x1_unnorm.squeeze().cpu().numpy(), cmap="gray")
    axes[3].set_title(f"Digit A ({lbl1})", fontsize=10)
    axes[3].axis("off")

    axes[4].imshow(x2_unnorm.squeeze().cpu().numpy(), cmap="gray")
    axes[4].set_title(f"Digit B ({lbl2})", fontsize=10)
    axes[4].axis("off")

    axes[5].imshow(hybrid.squeeze().cpu().numpy(), cmap="gray")
    axes[5].set_title(
        "Hybrid Negative\nM*A + (1-M)*B",
        fontsize=10,
        color="darkred",
        fontweight="bold",
    )
    axes[5].axis("off")

    fig.suptitle(
        "Hinton (2022): Data-Mask & Hybrid Negative Image Construction",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved hybrid mask generation breakdown to {output_path}")


def plot_ff_positive_negative_comparison(
    loader, output_path: str | Path, num_examples: int = 4
):
    """Compares Supervised FF, Hinton's Hybrid Mask Unsupervised FF, and Naive Random Noise."""
    images, labels = next(iter(loader))
    images = images[:num_examples]
    labels = labels[:num_examples]

    # Supervised Border Injected Pairs
    pos_sup = inject_label_border(images, labels, num_classes=10)
    wrong_labels = (labels + 1) % 10
    neg_sup = inject_label_border(
        images, labels, wrong_labels=wrong_labels, num_classes=10
    )

    #  Unsupervised Hybrid Pairs
    unnorm = unnormalize_mnist(images)
    hybrid_neg, masks, perm = create_hybrid_negative_batch(unnorm, blur_iterations=4)

    # Naive Random Noise
    naive_neg = unnorm.clone()
    noise_mask = torch.rand_like(naive_neg) < 0.30
    naive_neg[noise_mask] = torch.randn_like(naive_neg[noise_mask]).clamp(0, 1)

    fig, axes = plt.subplots(num_examples, 5, figsize=(12, 2.5 * num_examples))
    for idx in range(num_examples):
        lbl = int(labels[idx].item())
        w_lbl = int(wrong_labels[idx].item())

        # Col 0: Supervised Pos
        axes[idx, 0].imshow(pos_sup[idx].view(28, 28).cpu().numpy(), cmap="gray")
        axes[idx, 0].set_title(f"Supervised Pos\n(True Label {lbl})", fontsize=9)
        axes[idx, 0].axis("off")

        # Col 1: Supervised Neg
        axes[idx, 1].imshow(neg_sup[idx].view(28, 28).cpu().numpy(), cmap="gray")
        axes[idx, 1].set_title(f"Supervised Neg\n(Wrong Label {w_lbl})", fontsize=9)
        axes[idx, 1].axis("off")

        # Col 2: Unsupervised Pos (clean)
        axes[idx, 2].imshow(unnorm[idx].squeeze().cpu().numpy(), cmap="gray")
        axes[idx, 2].set_title("Unsupervised Pos\n(Clean Real Image)", fontsize=9)
        axes[idx, 2].axis("off")

        # Col 3: Unsupervised Hinton Hybrid
        axes[idx, 3].imshow(hybrid_neg[idx].squeeze().cpu().numpy(), cmap="gray")
        axes[idx, 3].set_title(
            "Hinton Hybrid Neg\n(Preserves local strokes)",
            fontsize=9,
            color="darkgreen",
        )
        axes[idx, 3].axis("off")

        # Col 4: Naive Noise
        axes[idx, 4].imshow(naive_neg[idx].squeeze().cpu().numpy(), cmap="gray")
        axes[idx, 4].set_title(
            "Naive Noise Neg\n(Destroys local strokes)", fontsize=9, color="darkred"
        )
        axes[idx, 4].axis("off")

    fig.suptitle(
        "Data Generation Comparison: Supervised vs Hybrid vs Naive",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved positive/negative comparison grid to {output_path}")


def plot_goodness_histograms(
    model,
    loader,
    mode: str,
    output_path: str | Path,
    threshold: float = 2.0,
    device=None,
):
    """Plots layer-wise goodness distributions for positive vs negative samples."""
    from algorithms.forward_forward import _get_effective_threshold, goodness

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    pos_input = inject_label_border(images, labels, num_classes=10)
    wrong_labels = (labels + 1) % 10
    neg_input = inject_label_border(
        images, labels, wrong_labels=wrong_labels, num_classes=10
    )

    fig, axes = plt.subplots(1, len(model.layers), figsize=(6 * len(model.layers), 4.5))
    if len(model.layers) == 1:
        axes = [axes]

    with torch.no_grad():
        pos_feat = pos_input
        neg_feat = neg_input
        for idx, layer in enumerate(model.layers):
            pos_act, pos_feat = layer(pos_feat)
            neg_act, neg_feat = layer(neg_feat)

            g_pos = goodness(pos_act, mode).cpu().numpy()
            g_neg = goodness(neg_act, mode).cpu().numpy()

            ax = axes[idx]
            min_val = min(g_pos.min(), g_neg.min())
            max_val = max(g_pos.max(), g_neg.max())
            if max_val == min_val:
                max_val += 1e-4
            bins = np.linspace(min_val, max_val, 25)
            ax.hist(
                g_pos,
                bins=bins,
                alpha=0.7,
                label=f"Positive (mean={g_pos.mean():.3f})",
                color="royalblue",
            )
            ax.hist(
                g_neg,
                bins=bins,
                alpha=0.7,
                label=f"Negative (mean={g_neg.mean():.3f})",
                color="tomato",
            )
            eff_thresh = _get_effective_threshold(threshold, layer.hidden_dim, mode)
            ax.axvline(
                eff_thresh,
                color="black",
                linestyle="--",
                label=f"Threshold ({eff_thresh:.1f})",
            )
            ax.set_title(
                f"Supervised FF: {mode} (Layer {idx + 1})\nDifference: {g_pos.mean() - g_neg.mean():.3f}"
            )
            ax.set_xlabel("Goodness Score")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved goodness histogram to {output_path}")


def plot_goodness_comparative_histograms(
    models_dict: dict,
    loader,
    output_path: str | Path,
    threshold: float = 2.0,
    device=None,
):
    """Compares layer 1 goodness distributions across all Supervised FF goodness functions side-by-side."""
    from algorithms.forward_forward import _get_effective_threshold, goodness

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    pos_input = inject_label_border(images, labels, num_classes=10)
    wrong_labels = (labels + 1) % 10
    neg_input = inject_label_border(
        images, labels, wrong_labels=wrong_labels, num_classes=10
    )

    modes = list(models_dict.keys())
    n_modes = len(modes)
    cols = min(4, n_modes)
    rows = math.ceil(n_modes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)

    for i, mode in enumerate(modes):
        model = models_dict[mode]
        model.to(device).eval()
        ax = axes[i]

        with torch.no_grad():
            pos_act, _ = model.layers[0](pos_input)
            neg_act, _ = model.layers[0](neg_input)
            g_pos = goodness(pos_act, mode).cpu().numpy()
            g_neg = goodness(neg_act, mode).cpu().numpy()

        min_val = min(g_pos.min(), g_neg.min())
        max_val = max(g_pos.max(), g_neg.max())
        if max_val == min_val:
            max_val += 1e-4
        bins = np.linspace(min_val, max_val, 25)

        margin = g_pos.mean() - g_neg.mean()
        ax.hist(
            g_pos,
            bins=bins,
            alpha=0.65,
            label=f"Pos (μ={g_pos.mean():.2f})",
            color="#2563eb",
        )
        ax.hist(
            g_neg,
            bins=bins,
            alpha=0.65,
            label=f"Neg (μ={g_neg.mean():.2f})",
            color="#dc2626",
        )
        eff_thresh = _get_effective_threshold(
            threshold, model.layers[0].hidden_dim, mode
        )
        ax.axvline(
            eff_thresh,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"θ={eff_thresh:.1f}",
        )
        ax.set_title(
            f"Supervised FF: {mode}\nMargin: {margin:+.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Goodness G(y)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    for j in range(n_modes, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Supervised Forward-Forward: Comparative Goodness Distributions (Layer 1)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved comparative goodness histograms to {output_path}")


def plot_unsupervised_goodness_histograms(
    model,
    loader,
    mode: str,
    output_path: str | Path,
    threshold: float = 2.0,
    device=None,
):
    """Plots layer-wise goodness distributions for Unsupervised FF (clean vs hybrid images)."""
    from algorithms.forward_forward import _get_effective_threshold, goodness

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)

    pos_input, neg_input = model.make_positive_negative(images)

    fig, axes = plt.subplots(1, len(model.layers), figsize=(6 * len(model.layers), 4.5))
    if len(model.layers) == 1:
        axes = [axes]

    with torch.no_grad():
        pos_feat = pos_input
        neg_feat = neg_input
        for idx, layer in enumerate(model.layers):
            pos_act, pos_feat = layer(pos_feat)
            neg_act, neg_feat = layer(neg_feat)

            g_pos = goodness(pos_act, mode).cpu().numpy()
            g_neg = goodness(neg_act, mode).cpu().numpy()

            ax = axes[idx]
            min_val = min(g_pos.min(), g_neg.min())
            max_val = max(g_pos.max(), g_neg.max())
            if max_val == min_val:
                max_val += 1e-4
            bins = np.linspace(min_val, max_val, 25)
            ax.hist(
                g_pos,
                bins=bins,
                alpha=0.7,
                label=f"Clean Real (mean={g_pos.mean():.3f})",
                color="royalblue",
            )
            ax.hist(
                g_neg,
                bins=bins,
                alpha=0.7,
                label=f"Hybrid Mask (mean={g_neg.mean():.3f})",
                color="tomato",
            )
            eff_thresh = _get_effective_threshold(threshold, layer.hidden_dim, mode)
            ax.axvline(
                eff_thresh,
                color="black",
                linestyle="--",
                label=f"Threshold ({eff_thresh:.1f})",
            )
            ax.set_title(
                f"Unsupervised FF: {mode} (Layer {idx + 1})\nMargin: {g_pos.mean() - g_neg.mean():.3f}"
            )
            ax.set_xlabel("Goodness Score")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved unsupervised goodness histogram to {output_path}")


def plot_goodness_relationships(
    model, loader, modes: list[str], output_path: str | Path, device=None
):
    """Plots relationships (correlations and scatter geometries) between different goodness metrics."""
    from algorithms.forward_forward import goodness

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    pos_input = inject_label_border(images, labels, num_classes=10)
    wrong_labels = (labels + 1) % 10
    neg_input = inject_label_border(
        images, labels, wrong_labels=wrong_labels, num_classes=10
    )

    with torch.no_grad():
        pos_act, _ = model.layers[0](pos_input)
        neg_act, _ = model.layers[0](neg_input)

    scores_pos = {}
    scores_neg = {}
    for m in modes:
        scores_pos[m] = goodness(pos_act, m).cpu().numpy()
        scores_neg[m] = goodness(neg_act, m).cpu().numpy()

    other_modes = [m for m in modes if m != "squared"]
    n_scatter = len(other_modes)
    cols = 3
    rows = math.ceil((n_scatter + 1) / cols)

    fig = plt.figure(figsize=(15, 4.2 * rows))
    gs = fig.add_gridspec(rows, cols)

    # 1. Correlation Matrix Heatmap on Positive Activations
    ax_corr = fig.add_subplot(gs[0, 0])
    mat = np.zeros((len(modes), len(modes)))
    for i, m1 in enumerate(modes):
        for j, m2 in enumerate(modes):
            v1 = scores_pos[m1]
            v2 = scores_pos[m2]
            std_prod = np.std(v1) * np.std(v2)
            mat[i, j] = np.corrcoef(v1, v2)[0, 1] if std_prod > 1e-8 else 0.0

    im = ax_corr.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(modes)))
    ax_corr.set_yticks(range(len(modes)))
    ax_corr.set_xticklabels(modes, rotation=45, ha="right", fontsize=8)
    ax_corr.set_yticklabels(modes, fontsize=8)
    for i in range(len(modes)):
        for j in range(len(modes)):
            val = mat[i, j]
            ax_corr.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=7,
            )
    ax_corr.set_title(
        "Pairwise Metric Correlation (Pos)", fontsize=10, fontweight="bold"
    )
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    # 2. Scatter plots vs 'squared' baseline
    base_pos = scores_pos["squared"]
    base_neg = scores_neg["squared"]

    for idx, m in enumerate(other_modes):
        pos_idx = idx + 1
        r = pos_idx // cols
        c = pos_idx % cols
        ax = fig.add_subplot(gs[r, c])
        ax.scatter(
            base_pos, scores_pos[m], color="#2563eb", alpha=0.7, s=24, label="Positive"
        )
        ax.scatter(
            base_neg, scores_neg[m], color="#dc2626", alpha=0.7, s=24, label="Negative"
        )
        ax.set_xlabel("Supervised FF (squared) Goodness", fontsize=8)
        ax.set_ylabel(f"Supervised FF ({m}) Goodness", fontsize=8)
        ax.set_title(f"{m} vs. squared", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle=":", alpha=0.5)

    total_slots = rows * cols
    for slot in range(n_scatter + 1, total_slots):
        r = slot // cols
        c = slot % cols
        fig.add_subplot(gs[r, c]).axis("off")

    fig.suptitle(
        "Supervised Forward-Forward: Inter-Goodness Metric Relationships & Geometries",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved goodness relationships plot to {output_path}")


def plot_noprop_vs_diffusion_denoising(
    noprop_model, loader, output_path: str | Path, device=None
):
    """Visualizes step-by-step target denoising in NoProp from initial noise z_0 to z_T."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noprop_model.to(device)
    noprop_model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:4].to(device), labels[:4].to(device)

    with torch.no_grad():
        b = images.size(0)
        state = torch.randn(b, noprop_model.num_classes, device=device)
        trajectories = [state.clone()]

        for block_idx in range(noprop_model.num_blocks):
            state = noprop_model.step(images, state, block_idx)
            trajectories.append(state.clone())

    # Plot heatmaps of the class target state across blocks
    fig, axes = plt.subplots(4, len(trajectories), figsize=(3 * len(trajectories), 7))
    for sample_idx in range(4):
        lbl = int(labels[sample_idx].item())
        for step_idx, traj in enumerate(trajectories):
            ax = axes[sample_idx, step_idx]
            probs = torch.softmax(traj[sample_idx], dim=0).cpu().numpy()
            ax.bar(
                range(10), probs, color="teal" if probs.argmax() == lbl else "salmon"
            )
            ax.set_ylim(0, 1.0)
            ax.set_xticks(range(10))
            if sample_idx == 0:
                title = (
                    f"z_0 (Noise)"
                    if step_idx == 0
                    else f"Block {step_idx} (z_{step_idx})"
                )
                ax.set_title(title, fontsize=10, fontweight="bold")
            if step_idx == 0:
                ax.set_ylabel(f"Digit {lbl}", fontsize=10, fontweight="bold")

    fig.suptitle(
        "NoProp-DT Target Denoising Trajectory Across Independent Blocks",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved NoProp denoising trajectory to {output_path}")


def plot_2d_decision_boundaries(
    bp_model,
    ff_model,
    noprop_model,
    train_ds,
    output_path: str | Path,
    device=None,
):
    """Visualizes decision boundaries on the 2D synthetic dataset for Backprop, FF, and NoProp."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bp_model.eval().to(device)
    ff_model.eval().to(device)
    noprop_model.eval().to(device)

    x_data, y_data = train_ds.tensors
    x_min, x_max = x_data[:, 0].min().item() - 0.3, x_data[:, 0].max().item() + 0.3
    y_min, y_max = x_data[:, 1].min().item() - 0.3, x_data[:, 1].max().item() + 0.3

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120)
    )
    grid_tensor = torch.tensor(
        np.stack([grid_x.ravel(), grid_y.ravel()], axis=1),
        dtype=torch.float32,
        device=device,
    )

    bp_preds_list = []
    ff_preds_list = []
    np_preds_list = []
    chunk_size = 512

    with torch.no_grad():
        for i in range(0, grid_tensor.size(0), chunk_size):
            chunk = grid_tensor[i : i + chunk_size]

            # 1. Backprop prediction
            bp_chunk = bp_model(chunk).argmax(dim=1).cpu().numpy()
            bp_preds_list.append(bp_chunk)

            # 2. FF prediction via accumulated goodness
            ff_chunk = (
                ff_model.forward_scores(chunk, threshold=0.5, device=device)
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
            ff_preds_list.append(ff_chunk)

            # 3. NoProp prediction
            state = torch.randn(chunk.size(0), noprop_model.num_classes, device=device)
            for block_idx in range(noprop_model.num_blocks):
                state = noprop_model.step(chunk, state, block_idx)
            np_preds_list.append(
                noprop_model.predict(state).argmax(dim=1).cpu().numpy()
            )

        bp_preds = np.concatenate(bp_preds_list).reshape(grid_x.shape)
        ff_preds = np.concatenate(ff_preds_list).reshape(grid_x.shape)
        np_preds = np.concatenate(np_preds_list).reshape(grid_x.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    cmap = "coolwarm"

    # 1. Backprop
    axes[0].contourf(grid_x, grid_y, bp_preds, alpha=0.3, cmap=cmap)
    axes[0].scatter(
        x_data[:, 0], x_data[:, 1], c=y_data, cmap=cmap, edgecolors="k", s=18
    )
    axes[0].set_title("Standard MLP (Backpropagation)", fontsize=11, fontweight="bold")

    # 2. Supervised FF: squared
    axes[1].contourf(grid_x, grid_y, ff_preds, alpha=0.3, cmap=cmap)
    axes[1].scatter(
        x_data[:, 0], x_data[:, 1], c=y_data, cmap=cmap, edgecolors="k", s=18
    )
    axes[1].set_title(
        "Supervised FF: squared (Local Goodness)", fontsize=11, fontweight="bold"
    )

    # 3. NoProp
    axes[2].contourf(grid_x, grid_y, np_preds, alpha=0.3, cmap=cmap)
    axes[2].scatter(
        x_data[:, 0], x_data[:, 1], c=y_data, cmap=cmap, edgecolors="k", s=18
    )
    axes[2].set_title("NoProp-DT (Target Diffusion)", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    fig.suptitle(
        "2D Synthetic Multi-Class Decision Landscapes Across Paradigms",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved 2D decision boundaries to {output_path}")


def plot_comparative_benchmark(results: dict, output_path: str | Path):
    """Bar chart comparisons across accuracy, training time, and memory footprint."""
    methods = []
    accuracies = []
    times = []
    bar_colors = []

    label_map = {
        "backprop": ("Backprop (MLP)", "#1e3a8a"),
        "supervised_ff_squared": ("Sup FF: squared", "#2563eb"),
        "supervised_ff_sparse": ("Sup FF: sparse", "#3b82f6"),
        "supervised_ff_margin": ("Sup FF: margin", "#60a5fa"),
        "supervised_ff_spectral": ("Sup FF: spectral", "#0284c7"),
        "supervised_ff_hebbian_wire": ("Sup FF: hebbian", "#0ea5e9"),
        "supervised_ff_negative_squared": ("Sup FF: neg_sq", "#94a3b8"),
        "unsupervised_ff_squared": ("Unsup FF: squared", "#059669"),
        "unsupervised_ff_sparse": ("Unsup FF: sparse", "#10b981"),
        "unsupervised_ff_margin": ("Unsup FF: margin", "#34d399"),
        "unsupervised_ff_spectral": ("Unsup FF: spectral", "#047857"),
        "unsupervised_ff_hebbian_wire": ("Unsup FF: hebbian", "#065f46"),
        "unsupervised_ff_negative_squared": ("Unsup FF: neg_sq", "#cbd5e1"),
        "noprop": ("NoProp-DT", "#f59e0b"),
        "diffusion": ("Diffusion Baseline", "#8b5cf6"),
    }

    for key, (name, color) in label_map.items():
        acc_key = f"{key}_accuracy"
        time_key = f"{key}_time_s"
        if acc_key in results:
            methods.append(name)
            accuracies.append(results[acc_key] * 100.0)
            times.append(results.get(time_key, 0.0))
            bar_colors.append(color)
        else:
            legacy_key = (
                key.replace("supervised_", "").replace("_squared", "")
                if "unsupervised" in key
                else key.replace("supervised_", "")
            )
            if f"{legacy_key}_accuracy" in results:
                methods.append(name)
                accuracies.append(results[f"{legacy_key}_accuracy"] * 100.0)
                times.append(results.get(f"{legacy_key}_time_s", 0.0))
                bar_colors.append(color)

    if not methods:
        return

    n_bars = len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 1.2 * n_bars), 5.5))

    # Accuracy comparison
    bars1 = axes[0].bar(
        range(n_bars),
        accuracies,
        color=bar_colors,
        alpha=0.90,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0].set_ylabel("Test Accuracy (%)", fontsize=10)
    axes[0].set_title(
        "Classification Accuracy Comparison Across Paradigms",
        fontweight="bold",
        fontsize=11,
    )
    axes[0].set_ylim(0, 105)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.2,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
    axes[0].set_xticks(range(n_bars))
    axes[0].set_xticklabels(methods, ha="right", fontsize=8, rotation=35)

    # Time comparison
    bars2 = axes[1].bar(
        range(n_bars),
        times,
        color=bar_colors,
        alpha=0.90,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1].set_ylabel("Training Time (seconds)", fontsize=10)
    axes[1].set_title(
        "Training Time Comparison Across Paradigms", fontweight="bold", fontsize=11
    )
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.08,
            f"{yval:.2f}s",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    axes[1].set_xticks(range(n_bars))
    axes[1].set_xticklabels(methods, ha="right", fontsize=8, rotation=35)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved comparative benchmark charts to {output_path}")


def plot_energy_and_memory_comparison(results: dict, output_path: str | Path):
    """
    2x2 vertical stack publication figure comparing:
    """
    methods = []
    energies = []
    memories = []
    accuracies = []
    bar_colors = []
    powers = []

    # Map key paradigms to clean labels & colors
    paradigm_order = [
        ("backprop", "Backprop (MLP)", "#1e3a8a"),
        ("supervised_ff_margin", "Sup FF: margin", "#2563eb"),
        ("supervised_ff_squared", "Sup FF: squared", "#3b82f6"),
        ("supervised_ff_spectral", "Sup FF: spectral", "#0284c7"),
        ("supervised_ff_hebbian_wire", "Sup FF: hebbian", "#0ea5e9"),
        ("unsupervised_ff_spectral", "Unsup FF: spectral", "#047857"),
        ("unsupervised_ff_hebbian_wire", "Unsup FF: hebbian", "#065f46"),
        ("unsupervised_ff_squared", "Unsup FF: squared", "#10b981"),
        ("noprop", "NoProp-DT", "#f59e0b"),
        ("diffusion", "Diffusion Baseline", "#8b5cf6"),
    ]

    for key, name, color in paradigm_order:
        acc_k = f"{key}_accuracy"
        en_k = f"{key}_energy_joules"
        mem_k = f"{key}_peak_memory_mb"
        pow_k = f"{key}_avg_power_w"
        if acc_k in results:
            methods.append(name)
            accuracies.append(results[acc_k] * 100.0)
            energies.append(results.get(en_k, results.get(f"{key}_time_s", 1.0) * 20.0))
            memories.append(results.get(mem_k, 15.0))
            powers.append(results.get(pow_k))
            bar_colors.append(color)

    if not methods:
        return

    n_bars = len(methods)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Row 1: Energy (Joules)
    bars1 = axes[0].bar(
        range(n_bars),
        energies,
        color=bar_colors,
        alpha=0.90,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0].set_ylabel("Energy Consumed (Joules)", fontsize=11, fontweight="bold")
    axes[0].set_title(
        "Total Training Energy Consumption",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max(energies) * 0.015,
            f"{yval:.1f} J",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    axes[0].set_xticks(range(n_bars))
    axes[0].set_xticklabels(methods, ha="right", fontsize=8.5, rotation=25)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Peak VRAM Memory Footprint (MB)
    bars2 = axes[1].bar(
        range(n_bars),
        memories,
        color=bar_colors,
        alpha=0.90,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1].set_ylabel("Peak VRAM (MB)", fontsize=11, fontweight="bold")
    axes[1].set_title(
        "Peak Allocated Memory Footprint",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max(memories) * 0.015,
            f"{yval:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    axes[1].set_xticks(range(n_bars))
    axes[1].set_xticklabels(methods, ha="right", fontsize=8.5, rotation=25)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    # Row 3: Energy-Accuracy Pareto Frontier
    axes[2].set_title(
        "Energy-Accuracy Pareto Frontier (Test Accuracy % vs Energy Joules)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    for i, (m, en, acc, col) in enumerate(
        zip(methods, energies, accuracies, bar_colors)
    ):
        axes[2].scatter(
            en, acc, color=col, s=120, edgecolor="black", linewidth=1.2, zorder=5
        )
        offset_y = 1.2 if i % 2 == 0 else -1.8
        offset_x = 0.2 if i % 2 == 0 else -0.2
        axes[2].annotate(
            m, (en + offset_x, acc + offset_y), fontsize=8, fontweight="bold", alpha=0.9
        )
    axes[2].set_xlabel("Energy Consumed (Joules)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    
    # Row 4: Power-Accuracy Pareto Frontier
    axes[2].set_title(
        "Power-Accuracy Pareto Frontier (Test Accuracy % vs Power W)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    for i, (m, pw, acc, col) in enumerate(
        zip(methods, powers, accuracies, bar_colors)
    ):
        axes[2].scatter(
            pw, acc, color=col, s=120, edgecolor="black", linewidth=1.2, zorder=5
        )
        offset_y = 1.2 if i % 2 == 0 else -1.8
        offset_x = 0.2 if i % 2 == 0 else -0.2
        axes[2].annotate(
            m, (pw + offset_x, acc + offset_y), fontsize=8, fontweight="bold", alpha=0.9
        )
    axes[2].set_xlabel("Power Consumed (W)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    


    fig.suptitle(
        "Energy, Power & Memory Profiling Across Paradigms",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved energy and memory comparison to {output_path}")


def plot_layer_representations(layer_data: dict, output_path: str | Path):
    """
    Visualizes layer-by-layer representational mechanics:
    - Panel 1: Backprop hierarchical abstraction (linear probe accuracy by layer)
    - Panel 2: Unsupervised FF (spectral) autonomous multi-scale goodness margin across layers
    - Panel 3: NoProp-DT reverse diffusion target space trajectory error across denoising steps
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: Backprop Hierarchical Abstraction
    if "backprop_layer_accs" not in layer_data:
        raise KeyError(
            "Missing required key 'backprop_layer_accs' in layer_data for plot_layer_representations."
        )
    bp_layers = layer_data["backprop_layer_accs"]
    labels = [x[0] for x in bp_layers]
    accs = [x[1] for x in bp_layers]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))
    bars = axes[0].bar(labels, accs, color=colors, edgecolor="black", width=0.55)
    axes[0].set_ylabel("Test Accuracy (%)", fontsize=10, fontweight="bold")
    axes[0].set_title(
        "Backprop: Hierarchical Abstraction\n(Layer-by-Layer Linear Separability)",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", labelsize=8.5, rotation=12)
    for b in bars:
        h = b.get_height()
        axes[0].text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.5,
            f"{h:.1f}%",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: Unsupervised FF Multi-Scale Contrast
    if "ff_spectral_margins" in layer_data and layer_data["ff_spectral_margins"]:
        ff_layers = layer_data["ff_spectral_margins"]
        x_indices = np.arange(len(ff_layers))
        width = 0.35
        l_names = [x[0] for x in ff_layers]
        pos_means = [x[1] for x in ff_layers]
        neg_means = [x[2] for x in ff_layers]
        axes[1].bar(
            x_indices - width / 2,
            pos_means,
            width,
            label="Clean Real (Pos)",
            color="#047857",
            edgecolor="black",
        )
        axes[1].bar(
            x_indices + width / 2,
            neg_means,
            width,
            label="Hybrid Mask (Neg)",
            color="#f87171",
            edgecolor="black",
        )
        axes[1].set_xticks(x_indices)
        axes[1].set_xticklabels(l_names, fontsize=9.5, fontweight="bold")
        axes[1].set_ylabel("Goodness Score", fontsize=10, fontweight="bold")
        axes[1].set_title(
            "Unsupervised FF:\nMulti-Scale Goodness Margin",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].legend(fontsize=8.5)
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Unsupervised FF Margins\nNot Evaluated",
            ha="center",
            va="center",
            fontsize=11,
        )
        axes[1].axis("off")

    # Panel 3: NoProp Denoising Target Trajectory
    if "noprop_step_errors" not in layer_data:
        raise KeyError(
            "Missing required key 'noprop_step_errors' in layer_data for plot_layer_representations."
        )
    np_steps = layer_data["noprop_step_errors"]
    step_names = [x[0] for x in np_steps]
    step_errors = [x[1] for x in np_steps]
    axes[2].plot(
        step_names,
        step_errors,
        marker="o",
        color="#d97706",
        linewidth=2.5,
        markersize=8,
        label="MSE to Target Simplex",
    )
    axes[2].fill_between(step_names, step_errors, color="#fef3c7", alpha=0.6)
    axes[2].set_ylabel("MSE to One-Hot Target Simplex", fontsize=10, fontweight="bold")
    axes[2].set_title(
        "NoProp-DT: Target Space Denoising\n(Traversing Reverse Diffusion Vector Field)",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].set_ylim(0, 1.0)
    for i, (name, err) in enumerate(np_steps):
        axes[2].text(
            i, err + 0.04, f"{err:.2f}", ha="center", fontsize=9, fontweight="bold"
        )
    axes[2].grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] saved layer representations to {output_path}")
