import os
import torch
import argparse
import matplotlib.pyplot as plt
from src.core.model import ModifiedPerceptronClassifier
from src.connectomics.priors import get_microns_priors
from src.connectomics.dataloader import get_mnist_loaders
from src.evaluation.visualize import (
    plot_membrane_potentials,
    plot_hidden_raster,
    plot_input_and_output_rasters,
)


def evaluate_model(
    model_path="experiment_results/model.pth", sample_index=0, time_steps=30,seed=42
):
    print(f"Loading Model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first.")
        return

    print("Fetching Priors")
    priors_l1 = get_microns_priors(n_in=784, n_out=128,seed=seed)
    priors_l2 = get_microns_priors(n_in=128, n_out=10,seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedPerceptronClassifier(
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        priors_l1=priors_l1,
        priors_l2=priors_l2,
        weight_scale=10.0, 
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loading MNIST Test Dataset (Sample {sample_index})")
    _, test_loader = get_mnist_loaders(batch_size=1)

    for idx, (images, labels) in enumerate(test_loader):
        if idx == sample_index:
            target_image = images
            target_label = labels.item()
            break

    print(f"True Label: {target_label}")

    plt.figure(figsize=(3, 3))
    plt.imshow(target_image.squeeze().cpu().numpy(), cmap="gray")
    plt.title(f"Target Digit: {target_label}")
    plt.axis("off")
    plt.show()

    images = target_image.view(target_image.size(0), -1).to(device)
    img_expanded = images.unsqueeze(1).repeat(1, time_steps, 1)
    input_spikes = (torch.rand_like(img_expanded) < img_expanded).float()

    print("Running Forward Pass with Temporal State Tracking")
    with torch.no_grad():
        spike_counts, out_spikes, hidden_spikes, l2_states = model(
            input_spikes, return_states=True
        )

    prediction = torch.argmax(spike_counts, dim=1).item()
    print(f"Perceptron Prediction: {prediction}")
    print(f"Total Output Spikes: {spike_counts[0].cpu().numpy()}")

    print("\nGenerating Visualizations...")
    os.makedirs("experiment_results", exist_ok=True)

    plot_input_and_output_rasters(input_spikes, out_spikes, "Evaluation")

    plot_hidden_raster(hidden_spikes, "Layer 1 (Hidden) Sparse Activation Raster")

    potentials = l2_states["potentials"]
    thresholds = l2_states["thresholds"]

    plot_membrane_potentials(
        potentials, thresholds, out_spikes, target_label=target_label
    )

    print("Extracting STP States from Layer 1...")
    with torch.no_grad():
        _, l1_states = model.layer1(input_spikes, return_states=True)
        stp_hist = l1_states["stp_states"]

    from src.evaluation.visualize import plot_stp_reserves

    plot_stp_reserves(stp_hist, title="Layer 1 STP: Neurotransmitter Depletion")

    print("Evaluation Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained SNN")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Index of the MNIST test sample to evaluate.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=30,
        help="Integration timesteps for Poisson encoding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed"
    )
    args = parser.parse_args()

    evaluate_model(sample_index=args.sample, time_steps=args.time_steps)
