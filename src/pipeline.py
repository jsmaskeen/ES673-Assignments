import os
import torch
import torch.nn as nn
from src.connectomics.priors import get_microns_priors
from src.connectomics.dataloader import get_mnist_loaders
from src.core.model import ModifiedPerceptronClassifier
from src.evaluation.visualize import (
    plot_input_and_output_rasters,
    plot_metrics,
    plot_weight_trends,
    plot_mnist_shadows,
    plot_dales_law_histogram,
    plot_spatial_priors,
    plot_confusion_matrix,
    plot_sparsity_histogram,
    plot_pca_representations,
    plot_weight_distance_correlation,
)
from src.evaluation.performance_analyzer import PerformanceAnalyzer


class ModifiedPerceptronPipeline:
    def __init__(
        self,
        time_steps=20,
        epochs=10,
        weight_scale=10.0,
        output_neurons=10,
        output_dir="experiment_results",
        seed=42,
        lambda_constant=150.0,
        tau_m=20.0,
        U=0.2,
        gamma=0.02,
        surrogate_k=25.0,
        plot=True
    ):
        self.time_steps = time_steps
        self.epochs = epochs
        self.weight_scale = weight_scale
        self.output_neurons = output_neurons
        self.output_dir = output_dir
        self.analyzer = PerformanceAnalyzer(output_dir=output_dir)
        self.seed = seed
        self.lambda_constant = lambda_constant
        self.tau_m = tau_m
        self.U = U
        self.gamma = gamma
        self.surrogate_k = surrogate_k
        self.plot = plot

    def run_mnist_experiment(self):
        print("Fetching Biological Priors from MICrONS")
        priors_l1 = get_microns_priors(n_in=784, n_out=128,seed=self.seed, lambda_constant=self.lambda_constant)
        priors_l2 = get_microns_priors(n_in=128, n_out=self.output_neurons,seed=self.seed, lambda_constant=self.lambda_constant)

        print("Initializing Modified Perceptron Model")
        model = ModifiedPerceptronClassifier(
            input_dim=784,
            hidden_dim=128,
            output_dim=self.output_neurons,
            priors_l1=priors_l1,
            priors_l2=priors_l2,
            weight_scale=self.weight_scale,
            tau_m=self.tau_m,
            U=self.U,
            gamma=self.gamma,
            surrogate_k=self.surrogate_k,
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        train_loader, test_loader = get_mnist_loaders(batch_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        loss_hist, acc_hist = [], []
        weight_hist = []

        print("Training with BPTT and Short-Term Plasticity")

        for epoch in range(1, self.epochs + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            total_spikes = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.view(images.size(0), -1)

                # Poisson Encoding
                img_expanded = images.unsqueeze(1).repeat(1, self.time_steps, 1)
                input_spikes = (torch.rand_like(img_expanded) < img_expanded).float()

                optimizer.zero_grad()

                spike_counts, output_spikes = model(input_spikes)

                # Supervised BPTT logic
                loss = criterion(spike_counts, labels)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(spike_counts, dim=1)

                total_loss += loss.item() * images.size(0)
                correct += (preds == labels.cpu()).sum().item()
                total += labels.size(0)
                total_spikes += spike_counts.sum().item()

                if batch_idx >= 30:
                    break

            avg_loss = total_loss / total
            acc = 100.0 * correct / total
            avg_spikes_per_sample = total_spikes / total

            loss_hist.append(avg_loss)
            acc_hist.append(acc)

            self.analyzer.log_epoch(epoch, avg_loss, acc, avg_spikes_per_sample)

            with torch.no_grad():
                w_eff = (
                    model.layer2.synapse.get_effective_weights().cpu().flatten().numpy()
                )
                weight_hist.append(w_eff)

            print(
                f"Epoch {epoch:02d}/{self.epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | Spikes: {avg_spikes_per_sample:.1f}"
            )

            if (epoch == 1 or epoch == self.epochs) and self.plot:
                plot_input_and_output_rasters(
                    input_spikes, output_spikes, f"MNIST (Epoch {epoch})"
                )

            if epoch % 5 == 0 and self.output_neurons == 10:
                W1_eff = model.layer1.synapse.get_effective_weights()
                W2_eff = model.layer2.synapse.get_effective_weights()
                shadow_path = os.path.join(
                    self.output_dir, f"shadows_epoch_{epoch}.png"
                )
                if self.plot:
                    plot_mnist_shadows(W1_eff, W2_eff, filename=shadow_path)
        if self.plot:
            plot_metrics(loss_hist, acc_hist, "MNIST")
            plot_weight_trends(weight_hist, "MNIST L2")

        dales_law_path = os.path.join(self.output_dir, "dales_law_distribution.png")
        # Layer 1 raw weights
        if self.plot:
            plot_dales_law_histogram(
                model.layer1.synapse.raw_weights,
                priors_l1["pre_is_exc"],
                filename=dales_law_path,
            )

        spatial_priors_path = os.path.join(self.output_dir, "spatial_priors.png")
        if self.plot:
            plot_spatial_priors(
                model.layer1.synapse.attenuation,
                model.layer1.synapse.adjacency,
                filename=spatial_priors_path,
            )

        W1_eff = model.layer1.synapse.get_effective_weights()
        W2_eff = model.layer2.synapse.get_effective_weights()

        if self.output_neurons == 10:
            shadow_path = os.path.join(self.output_dir, "shadows_final.png")
            if self.plot:
                plot_mnist_shadows(W1_eff, W2_eff, filename=shadow_path)
        else:
            print(
                f"Skipping Receptive Field shadow plotting because output_neurons={self.output_neurons} (expects 10)."
            )

        model.eval()
        all_targets = []
        all_preds = []
        all_hidden_features = []

        total_input_spikes = 0
        total_hidden_spikes = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                labels = labels.to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )

                # Poisson Encoding
                img_expanded = images.unsqueeze(1).repeat(1, self.time_steps, 1)
                input_spikes = (torch.rand_like(img_expanded) < img_expanded).float()

                # Forward with state tracking
                spike_counts, output_spikes, hidden_spikes, l2_states = model(
                    input_spikes, return_states=True
                )
                preds = torch.argmax(spike_counts, dim=1)

                all_targets.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                # Hidden features: sum of spikes over time for each neuron
                hidden_spike_counts = hidden_spikes.sum(dim=1)  # (Batch, HiddenDim)
                all_hidden_features.append(hidden_spike_counts.cpu())

                total_input_spikes += input_spikes.sum().item()
                total_hidden_spikes += hidden_spikes.sum().item()
                total_samples += images.size(0)

        all_hidden_features = torch.cat(all_hidden_features, dim=0)
        if self.plot:
            plot_confusion_matrix(
                all_targets,
                all_preds,
                filename=os.path.join(self.output_dir, "confusion_matrix.png"),
            )

        total_spikes_per_neuron = all_hidden_features.sum(dim=0)
        if self.plot:
            plot_sparsity_histogram(
                total_spikes_per_neuron,
                filename=os.path.join(self.output_dir, "sparsity_histogram.png"),
            )

        subset_idx = torch.randperm(all_hidden_features.size(0))[:1000]
        if self.plot:
            plot_pca_representations(
                all_hidden_features[subset_idx],
                torch.tensor(all_targets)[subset_idx],
                filename=os.path.join(self.output_dir, "pca_representations.png"),
            )
        if self.plot:
            plot_weight_distance_correlation(
                model.layer1.synapse.raw_weights,
                model.layer1.synapse.attenuation,
                filename=os.path.join(self.output_dir, "weight_distance_correlation.png"),
            )

        model_path = os.path.join(self.output_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        params = {
            "time_steps": self.time_steps,
            "epochs": self.epochs,
            "weight_scale": self.weight_scale,
            "output_neurons": self.output_neurons,
            "biological_features": "Dale's Law, Cable Attenuation, STP, Dynamic Thresholds",
        }
        
        print("\nPipeline execution complete!")
        return acc
