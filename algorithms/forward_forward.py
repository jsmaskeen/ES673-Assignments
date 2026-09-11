import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from utils.masks import create_hybrid_negative_batch, inject_label_border
from utils.metrics import accuracy


class FFLayer(nn.Module):
    """A Forward-Forward layer."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (raw_activations, l2_normalized_orientation).

        Following Hinton (2022) Section 2.1 & Footnote 5:
        Goodness is computed on the unnormalized length of the ReLU activity vector.
        The activity vector is normalized to unit Euclidean length before passing to the next layer.
        """
        # Normalize input to unit Euclidean norm
        x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        h = torch.relu(self.linear(x_norm))
        # Unit orientation for next layer
        h_norm = h / h.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return h, h_norm


def _get_effective_threshold(threshold: float, dim: int, mode: str) -> float:
    """Calculates layer-level threshold theta for logistic goodness contrast p = sigma(G(y) - theta).

    In Hinton (2022), goodness is the sum of squared activations for unit-norm inputs,
    operating naturally in the [1.0, 4.0] range. The threshold is a fixed layer-level scalar
    (e.g., 2.0 per Hinton).
    """
    if mode == "negative_squared":
        return -abs(threshold)
    elif mode == "sparse":
        return threshold * 0.90
    elif mode == "margin":
        return threshold * 0.90
    elif mode == "spectral":
        return threshold * 1.10
    elif mode == "hebbian_wire":
        return threshold * 1.10
    else:
        return threshold


def goodness(activation: Tensor, mode: str) -> Tensor:
    """Computes layer goodness score G(y) for a batch of activations."""
    dim = activation.size(1)
    if mode == "squared":
        return activation.pow(2).sum(dim=1)
    elif mode == "negative_squared":
        return -activation.pow(2).sum(dim=1)
    elif mode == "sparse":
        return activation.pow(2).sum(dim=1) - 0.20 * (activation.abs().sum(dim=1) / math.sqrt(dim))
    elif mode == "margin":
        return activation.pow(2).sum(dim=1) - 0.40 * F.relu(activation - 1.0).sum(dim=1)
    elif mode == "spectral":
        # Principal Eigen-Energy:
        # Reshapes (B, D) into a 2D feature matrix Y in R^{m x k} (e.g. 16x16 for D=256).
        # The Gramian G = (1/m) Y^T Y in R^{k x k} captures feature covariance.
        # Power iteration extracts dominant eigenvalue lambda_max representing variance
        # concentrated along the principal axis, contrasting with diffuse noise.
        b, d = activation.size()
        k = int(math.isqrt(d))
        m = k if k * k == d else (16 if d % 16 == 0 else 8)
        k = d // m
        Y = activation.view(b, m, k)
        G_mat = torch.bmm(Y.transpose(1, 2), Y) / float(m)
        v = torch.ones(b, k, 1, device=activation.device, dtype=activation.dtype) * (1.0 / math.sqrt(k))
        for _ in range(3):
            v = torch.bmm(G_mat, v)
            v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-6)
        lambda_max = torch.bmm(torch.bmm(v.transpose(1, 2), G_mat), v).view(b)
        return activation.pow(2).sum(dim=1) + 0.30 * lambda_max * m
    elif mode == "hebbian_wire":
        # Biological 'Fire Together, Wire Together' Associative Energy:
        # Self energy: ||y||_2^2 = sum_j y_j^2
        # Synchronous co-activation: sum_{j != k} y_j * y_k = ||y||_1^2 - ||y||_2^2
        l2_sq = activation.pow(2).sum(dim=1)
        l1_sq = activation.sum(dim=1).pow(2)
        coop = (l1_sq - l2_sq).clamp_min(0.0)
        return l2_sq + 0.50 * (coop / dim)
    else:
        raise ValueError(f"Unknown goodness mode: {mode}. Choices: squared, negative_squared, sparse, margin, spectral, hebbian_wire")


class ForwardForward(nn.Module):
    """Supervised Forward-Forward network with dual passes and local layer-wise goodness learning."""
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        num_classes: int = 10,
        mode: str = "squared",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mode = mode

        if hidden_dims is not None:
            self.hidden_dims = list(hidden_dims) if isinstance(hidden_dims, (list, tuple)) else [hidden_dims]
        elif hidden_dim is not None:
            self.hidden_dims = [hidden_dim] * num_layers
        else:
            self.hidden_dims = [512, 256, 128]

        layers = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(FFLayer(in_dim, h_dim))
            in_dim = h_dim
        self.layers = nn.ModuleList(layers)

    def get_hard_negatives(self, images: Tensor, labels: Tensor, threshold: float, device: torch.device) -> Tensor:
        """Finds the most confusable incorrect class for each sample."""
        b = images.size(0)
        best_wrong_goodness = torch.full((b,), -1e9, device=device)
        hard_wrong_labels = (labels + 1) % self.num_classes

        with torch.no_grad():
            for c in range(self.num_classes):
                candidate_labels = torch.full((b,), c, device=device, dtype=torch.long)
                is_true = (candidate_labels == labels)
                cand_inp = inject_label_border(images, candidate_labels, num_classes=self.num_classes)
                total_goodness = torch.zeros(b, device=device)
                curr_feat = cand_inp
                for idx, layer in enumerate(self.layers):
                    act, curr_feat = layer(curr_feat)
                    #  accumulate goodness of all but the first hidden layer
                    if idx > 0 or len(self.layers) == 1:
                        eff_thresh = _get_effective_threshold(threshold, layer.hidden_dim, self.mode)
                        total_goodness += goodness(act, self.mode) - eff_thresh

                total_goodness[is_true] = -1e9
                better = total_goodness > best_wrong_goodness
                hard_wrong_labels[better] = c
                best_wrong_goodness[better] = total_goodness[better]

        return hard_wrong_labels

    def train_one_epoch(
        self,
        loader,
        epoch: int,
        epochs: int,
        lr: float,
        threshold: float,
        device: torch.device,
        hard_negatives: bool = False,
        optimizers: list[torch.optim.Optimizer] | None = None,
    ):
        if optimizers is None:
            optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in self.layers]
        criterion = nn.BCEWithLogitsLoss()

        for layer_idx, layer in enumerate(self.layers):
            layer.train()
            running_loss = 0.0
            pos_scores_all = []
            neg_scores_all = []
            eff_thresh = _get_effective_threshold(threshold, layer.hidden_dim, self.mode)

            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                b = images.size(0)

                # Positive pass: real input with correct label
                pos_input = inject_label_border(images, labels, wrong_labels=None, num_classes=self.num_classes)

                # Negative pass: real input with wrong label
                if hard_negatives:
                    wrong_labels = self.get_hard_negatives(images, labels, threshold, device)
                else:
                    rand_offset = torch.randint(1, self.num_classes, labels.shape, device=device)
                    wrong_labels = (labels + rand_offset) % self.num_classes

                neg_input = inject_label_border(images, labels, wrong_labels=wrong_labels, num_classes=self.num_classes)

                # Forward through preceding layers with detached activations
                with torch.no_grad():
                    pos_feat = pos_input
                    neg_feat = neg_input
                    for prev_layer in self.layers[:layer_idx]:
                        _, pos_feat = prev_layer(pos_feat)
                        _, neg_feat = prev_layer(neg_feat)

                pos_act, _ = layer(pos_feat)
                neg_act, _ = layer(neg_feat)

                pos_score = goodness(pos_act, self.mode) - eff_thresh
                neg_score = goodness(neg_act, self.mode) - eff_thresh

                pos_scores_all.append(pos_score.detach().mean().item())
                neg_scores_all.append(neg_score.detach().mean().item())

                scores = torch.cat([pos_score, neg_score])
                targets = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])

                loss = criterion(scores, targets)
                optimizers[layer_idx].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[layer_idx].step()
                running_loss += loss.item() * b

            avg_loss = running_loss / len(loader.dataset)
            pos_mean = sum(pos_scores_all) / max(len(pos_scores_all), 1)
            neg_mean = sum(neg_scores_all) / max(len(neg_scores_all), 1)
            print(f"[Supervised FF:{self.mode}] layer {layer_idx + 1}/{len(self.layers)} epoch {epoch}/{epochs} loss={avg_loss:.4f} pos_G={pos_mean:.3f} neg_G={neg_mean:.3f} margin={pos_mean - neg_mean:.3f}")

    def forward_scores(
        self,
        images: Tensor,
        threshold: float,
        device: torch.device,
        skip_first: bool = True,
    ) -> Tensor:
        """Computes accumulated goodness across layers for each candidate class."""
        b = images.size(0)
        class_scores = []
        for c in range(self.num_classes):
            cand_labels = torch.full((b,), c, device=device, dtype=torch.long)
            cand_input = inject_label_border(images, cand_labels, num_classes=self.num_classes)
            total = torch.zeros(b, device=device)
            feat = cand_input
            for idx, layer in enumerate(self.layers):
                act, feat = layer(feat)
                # accumulate goodness of all but the first hidden layer when multi-layer
                if not skip_first or idx > 0 or len(self.layers) == 1:
                    eff_thresh = _get_effective_threshold(threshold, layer.hidden_dim, self.mode)
                    total += goodness(act, self.mode) - eff_thresh
            class_scores.append(total)
        return torch.stack(class_scores, dim=1)


class UnsupervisedForwardForward(nn.Module):
    """Unsupervised Forward-Forward using Hinton's smooth-blurred mask hybrid images."""
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        num_classes: int = 10,
        mode: str = "squared",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mode = mode

        if hidden_dims is not None:
            self.hidden_dims = list(hidden_dims) if isinstance(hidden_dims, (list, tuple)) else [hidden_dims]
        elif hidden_dim is not None:
            self.hidden_dims = [hidden_dim] * num_layers
        else:
            self.hidden_dims = [512, 256, 128]

        layers = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(FFLayer(in_dim, h_dim))
            in_dim = h_dim
        self.layers = nn.ModuleList(layers)
        self.linear_probe = None
        self.epoch_snapshots = []

    def make_positive_negative(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """positive = clean images; negative = hybrid images via blurred masks."""
        flat_pos = images.view(images.size(0), -1).to(dtype=torch.float32)
        hybrid_neg, _, _ = create_hybrid_negative_batch(images, blur_iterations=4, threshold=0.5)
        flat_neg = hybrid_neg.view(images.size(0), -1).to(dtype=torch.float32)
        return flat_pos, flat_neg

    def train_one_epoch(
        self,
        loader,
        epoch: int,
        epochs: int,
        lr: float,
        threshold: float,
        device: torch.device,
        optimizers: list[torch.optim.Optimizer] | None = None,
    ):
        if optimizers is None:
            optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in self.layers]
        criterion = nn.BCEWithLogitsLoss()
        latest_pos = None
        latest_neg = None

        for layer_idx, layer in enumerate(self.layers):
            layer.train()
            running_loss = 0.0
            pos_scores_all = []
            neg_scores_all = []
            eff_thresh = _get_effective_threshold(threshold, layer.hidden_dim, self.mode)

            for images, _ in loader:
                images = images.to(device)
                b = images.size(0)
                pos_input, neg_input = self.make_positive_negative(images)
                latest_pos = pos_input[:4].detach().cpu()
                latest_neg = neg_input[:4].detach().cpu()

                with torch.no_grad():
                    pos_feat = pos_input
                    neg_feat = neg_input
                    for prev_layer in self.layers[:layer_idx]:
                        _, pos_feat = prev_layer(pos_feat)
                        _, neg_feat = prev_layer(neg_feat)

                pos_act, _ = layer(pos_feat)
                neg_act, _ = layer(neg_feat)

                pos_score = goodness(pos_act, self.mode) - eff_thresh
                neg_score = goodness(neg_act, self.mode) - eff_thresh

                pos_scores_all.append(pos_score.detach().mean().item())
                neg_scores_all.append(neg_score.detach().mean().item())

                scores = torch.cat([pos_score, neg_score])
                targets = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])

                loss = criterion(scores, targets)
                optimizers[layer_idx].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[layer_idx].step()
                running_loss += loss.item() * b

            avg_loss = running_loss / len(loader.dataset)
            pos_mean = sum(pos_scores_all) / max(len(pos_scores_all), 1)
            neg_mean = sum(neg_scores_all) / max(len(neg_scores_all), 1)
            print(f"[Unsupervised FF:{self.mode}] layer {layer_idx + 1}/{len(self.layers)} epoch {epoch}/{epochs} loss={avg_loss:.4f} pos_G={pos_mean:.3f} neg_G={neg_mean:.3f} margin={pos_mean - neg_mean:.3f}")

        if latest_pos is not None and latest_neg is not None:
            self.epoch_snapshots.append({
                "epoch": epoch,
                "positive": latest_pos,
                "negative": latest_neg,
                "pos_goodness": pos_mean,
                "neg_goodness": neg_mean,
                "margin": pos_mean - neg_mean,
            })

    def extract_features(self, images: Tensor, device: torch.device) -> Tensor:
        """Concatenates normalized orientation representations from all layers."""
        feat = images.view(images.size(0), -1).to(device=device, dtype=torch.float32)
        layer_reps = []
        for layer in self.layers:
            _, feat = layer(feat)
            layer_reps.append(feat)
        return torch.cat(layer_reps, dim=1)

    def fit_linear_probe(self, loader, device: torch.device, epochs: int = 8, lr: float = 0.01):
        """Fits a linear classifier / linear probe on frozen unsupervised representations without backprop into layers."""
        self.eval()
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                feats = self.extract_features(images, device).cpu()
                all_feats.append(feats)
                all_labels.append(labels.cpu())

        x_rep = torch.cat(all_feats, dim=0)
        y_true = torch.cat(all_labels, dim=0)

        rep_dim = x_rep.size(1)
        self.linear_probe = nn.Linear(rep_dim, self.num_classes).to(device)
        probe_opt = torch.optim.Adam(self.linear_probe.parameters(), lr=lr)
        ce_loss = nn.CrossEntropyLoss()

        probe_ds = torch.utils.data.TensorDataset(x_rep, y_true)
        probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=64, shuffle=True)

        for ep in range(epochs):
            for bx, by in probe_loader:
                bx = bx.to(device)
                by = by.to(device)
                out = self.linear_probe(bx)
                loss = ce_loss(out, by)
                probe_opt.zero_grad(set_to_none=True)
                loss.backward()
                probe_opt.step()

    def predict(self, images: Tensor, device: torch.device) -> Tensor:
        if self.linear_probe is None:
            raise RuntimeError("Linear probe is not trained. Call fit_linear_probe first.")
        self.eval()
        with torch.no_grad():
            feats = self.extract_features(images, device)
            logits = self.linear_probe(feats)
            return logits.argmax(dim=1)


def train_forward_forward(
    model: ForwardForward,
    loader,
    epochs: int,
    lr: float,
    threshold: float,
    device: torch.device,
    hard_negatives: bool = False,
):
    model.to(device)
    optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in model.layers]
    for epoch in range(1, epochs + 1):
        model.train_one_epoch(
            loader,
            epoch,
            epochs,
            lr,
            threshold,
            device,
            hard_negatives=hard_negatives,
            optimizers=optimizers,
        )


def train_unsupervised_forward_forward(
    model: UnsupervisedForwardForward,
    loader,
    epochs: int,
    lr: float,
    threshold: float,
    device: torch.device,
):
    model.to(device)
    optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in model.layers]
    for epoch in range(1, epochs + 1):
        model.train_one_epoch(
            loader,
            epoch,
            epochs,
            lr,
            threshold,
            device,
            optimizers=optimizers,
        )


def evaluate_forward_forward(
    model: ForwardForward,
    loader,
    threshold: float,
    device: torch.device,
    skip_first: bool = True,
) -> float:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            scores = model.forward_scores(images, threshold, device, skip_first=skip_first)
            predictions.append(scores.argmax(dim=1))
            targets.append(labels)
    return accuracy(torch.cat(predictions), torch.cat(targets))


def evaluate_unsupervised_forward_forward(model: UnsupervisedForwardForward, train_loader, test_loader, device: torch.device) -> float:
    model.eval()
    model.fit_linear_probe(train_loader, device, epochs=8)
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model.predict(images, device)
            predictions.append(preds)
            targets.append(labels)
    return accuracy(torch.cat(predictions), torch.cat(targets))
