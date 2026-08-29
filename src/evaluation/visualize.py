import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import torch
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


def plot_input_and_output_rasters(input_spikes, output_spikes, title_prefix=""):
    """
    Plots side-by-side raster plots for input encoding spikes and output layer spikes.
    """
    in_spikes_sample = input_spikes[0].cpu().detach()
    out_spikes_sample = output_spikes[0].cpu().detach()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    splt.raster(in_spikes_sample, ax=ax1, s=15, c="black", marker="|")
    ax1.set_title(
        f"{title_prefix} Input Spike Train Raster", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("Time Step (t)", fontsize=11)
    ax1.set_ylabel("Input Neuron Index", fontsize=11)

    splt.raster(out_spikes_sample, ax=ax2, s=25, c="crimson", marker="|")
    ax2.set_title(
        f"{title_prefix} Output Layer Spike Raster", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Time Step (t)", fontsize=11)
    ax2.set_ylabel("Output Class Neuron Index", fontsize=11)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_metrics(loss_history, acc_history, task_name=""):
    """
    Plots the training loss and accuracy trajectories.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(loss_history, color="blue", marker="o")
    ax1.set_title(f"{task_name} Training Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(acc_history, color="green", marker="o")
    ax2.set_title(f"{task_name} Training Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_weight_trends(weight_history, task_name=""):
    """
    Plots the trend of 50 randomly selected active synapses over the epochs.
    weight_history is a list of 1D numpy arrays (the flattened effective weights).
    """
    if not weight_history:
        return

    import numpy as np

    history_matrix = np.stack(weight_history)  # Shape: (epochs, num_synapses)

    # Select 50 random active synapses to trace
    active_indices = np.where(history_matrix[0] > 0.01)[0]
    if len(active_indices) > 50:
        selected_indices = np.random.choice(active_indices, 50, replace=False)
    else:
        selected_indices = active_indices

    plt.figure(figsize=(10, 5))
    for idx in selected_indices:
        plt.plot(
            range(1, len(history_matrix) + 1),
            history_matrix[:, idx],
            alpha=0.5,
            linewidth=1,
        )

    plt.title(f"{task_name} Effective Synaptic Weights Evolution", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Effective Weight")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_mnist_shadows(W1_eff, W2_eff, filename=None):
    """
    Computes the effective receptive field for the 10 output classes by taking the dot product
    of the Hidden->Output weights and the Input->Hidden weights, then reshaping to 28x28.
    """
    # W1_eff: (128, 784)
    # W2_eff: (10, 128)
    # Total effective mapping: (10, 784)
    W_total = torch.matmul(W2_eff, W1_eff).cpu().detach().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        shadow = W_total[i].reshape(28, 28)
        im = axes[i].imshow(shadow, cmap="viridis")
        axes[i].set_title(f"Output Neuron {i}")
        axes[i].axis("off")

    fig.suptitle(
        "MNIST 'Shadows' - Output Neuron Receptive Fields (W2 * W1)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_hidden_raster(hidden_spikes, title="Hidden Layer Raster", filename=None):
    """
    Plots the raster for the hidden layer.
    hidden_spikes: (batch, time, features). We plot the first sample.
    """
    sample_spikes = hidden_spikes[0].cpu().detach()
    fig, ax = plt.subplots(figsize=(10, 5))
    splt.raster(sample_spikes, ax=ax, s=15, c="blue", marker="|")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step (t)")
    ax.set_ylabel("Hidden Neuron Index")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_membrane_potentials(
    potentials, thresholds, spikes, target_label=None, filename=None
):
    """
    Plots the membrane potential trajectories for the output neurons over time.
    Highlights when a spike occurs and shows the dynamic threshold.
    potentials: (batch, time, classes)
    """
    # Extract first sample
    pot_sample = potentials[0].cpu().detach().numpy()  # (time, 10)
    thresh_sample = thresholds[0].cpu().detach().numpy()  # (time, 10)
    spike_sample = spikes[0].cpu().detach().numpy()  # (time, 10)

    num_classes = pot_sample.shape[1]
    time_steps = pot_sample.shape[0]

    fig, axes = plt.subplots(
        num_classes, 1, figsize=(12, 1.5 * num_classes), sharex=True
    )
    if num_classes == 1:
        axes = [axes]

    for i in range(num_classes):
        ax = axes[i]
        # Plot potential
        color = "green" if (target_label is not None and i == target_label) else "gray"
        linewidth = 2 if (target_label is not None and i == target_label) else 1

        ax.plot(
            pot_sample[:, i],
            label="Membrane Potential",
            color=color,
            linewidth=linewidth,
        )
        ax.plot(
            thresh_sample[:, i],
            label="Dynamic Threshold",
            color="red",
            linestyle="--",
            alpha=0.6,
        )

        spike_times = [t for t in range(time_steps) if spike_sample[t, i] > 0]
        if spike_times:
            ax.vlines(
                x=spike_times,
                ymin=0,
                ymax=ax.get_ylim()[1],
                color="black",
                alpha=0.3,
                linestyle=":",
            )
            ax.scatter(
                spike_times,
                pot_sample[spike_times, i],
                color="red",
                marker="x",
                zorder=5,
            )

        ax.set_ylabel(f"Class {i}")
        if i == 0:
            ax.legend(loc="upper right", fontsize="small")

    axes[-1].set_xlabel("Time Step (t)")
    fig.suptitle(
        "Output Neurons: Membrane Potential vs Dynamic Threshold",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_stp_reserves(
    stp_history,
    title="Short-Term Plasticity: Neurotransmitter Depletion",
    filename=None,
):
    """
    Plots the neurotransmitter reserves (R) over time for a subset of synapses.
    stp_history: (batch, time, in_features)
    """
    import numpy as np

    # Extract first sample
    stp_sample = stp_history[0].cpu().detach().numpy()  # (time, in_features)

    # Select a few synapses that actually depleted (active synapses)
    min_reserves = stp_sample.min(axis=0)
    active_indices = np.argsort(min_reserves)[:5]  # Get 5 most depleted synapses

    plt.figure(figsize=(10, 4))
    for idx in active_indices:
        plt.plot(stp_sample[:, idx], linewidth=2, label=f"Synapse {idx}")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Neurotransmitter Reserve (R)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_dales_law_histogram(raw_weights, is_exc_mask, filename=None):
    """
    Plots a histogram of effective weights, separated by Excitatory (positive) and Inhibitory (negative).
    raw_weights: (out_features, in_features)
    is_exc_mask: (1, in_features) boolean mask
    """
    import numpy as np

    weights = raw_weights.cpu().detach().numpy().flatten()
    mask = is_exc_mask.cpu().detach().numpy().flatten()

    # Expand mask to match weights shape (out_features, in_features)
    out_features = raw_weights.shape[0]
    full_mask = np.tile(mask, (out_features, 1)).flatten()

    exc_weights = np.abs(weights[full_mask])
    inh_weights = -np.abs(weights[~full_mask])

    plt.figure(figsize=(10, 5))
    plt.hist(
        exc_weights,
        bins=50,
        color="crimson",
        alpha=0.7,
        label="Excitatory Synapses (+)",
    )
    plt.hist(
        inh_weights,
        bins=50,
        color="royalblue",
        alpha=0.7,
        label="Inhibitory Synapses (-)",
    )

    plt.title(
        "Dale's Principle: Biological Weight Bifurcation",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Effective Synaptic Weight")
    plt.ylabel("Frequency")
    plt.axvline(0, color="black", linestyle="-", linewidth=2)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_spatial_priors(attenuation_matrix, adjacency_matrix, filename=None):
    """
    Plots 2D heatmaps of the Synaptic Scaling Attenuation Matrix and the Structural Adjacency mask.
    """
    att = attenuation_matrix.cpu().detach().numpy()
    adj = adjacency_matrix.cpu().detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(att, cmap="viridis", aspect="auto")
    ax1.set_title("Synaptic Scaling Attenuation ($e^{-d/\lambda}$)", fontweight="bold")
    ax1.set_xlabel("Pre-Synaptic Neuron Index")
    ax1.set_ylabel("Post-Synaptic Neuron Index")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(adj, cmap="binary", aspect="auto")
    ax2.set_title(f"Structural Adjacency (Cutoff < $2\lambda$)", fontweight="bold")
    ax2.set_xlabel("Pre-Synaptic Neuron Index")
    ax2.set_ylabel("Post-Synaptic Neuron Index")
    fig.colorbar(im2, ax=ax2)

    fig.suptitle(
        "MICrONS Spatial Priors (1mm Cortical Column)", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(targets, preds, classes=10, filename=None):
    """
    Plots the confusion matrix for the predictions.
    """
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_sparsity_histogram(hidden_spike_counts, filename=None):
    """
    Plots a histogram of the total spike counts per hidden neuron across a dataset.
    hidden_spike_counts: 1D array or tensor of shape (hidden_dim,) containing total spikes per neuron.
    """
    plt.figure(figsize=(10, 5))
    counts = (
        hidden_spike_counts.cpu().numpy()
        if torch.is_tensor(hidden_spike_counts)
        else hidden_spike_counts
    )
    plt.hist(counts, bins=30, color="purple", alpha=0.7)
    plt.title(
        "Hidden Layer Sparsity (Spike Rate Distribution)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Total Spikes per Neuron")
    plt.ylabel("Number of Neurons")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_pca_representations(hidden_features, labels, filename=None):
    """
    Plots a 2D PCA projection of the hidden layer representations (e.g. spike counts).
    hidden_features: (num_samples, hidden_dim)
    labels: (num_samples,)
    """
    if isinstance(hidden_features, torch.Tensor):
        hidden_features = hidden_features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(hidden_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        pca_result[:, 0], pca_result[:, 1], c=labels, cmap="tab10", alpha=0.7, s=30
    )
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.title("PCA of Hidden Layer Spike Counts", fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_weight_distance_correlation(weights, attenuation, filename=None):
    """
    Scatter plot of biological attenuation distance vs learned absolute weights.
    """
    w = torch.abs(weights).cpu().detach().numpy().flatten()
    att = attenuation.cpu().detach().numpy().flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(att, w, alpha=0.1, color="teal", s=10)
    plt.title(
        "Biological Prior Correlation: Attenuation vs Learned Weight", fontweight="bold"
    )
    plt.xlabel("Synaptic Attenuation Factor (Higher = Closer)")
    plt.ylabel("Learned Absolute Weight")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
