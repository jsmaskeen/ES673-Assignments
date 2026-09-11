import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from utils.metrics import accuracy


class NoPropBlock(nn.Module):
    """Independent block for NoProp.

    Takes the broadcast input image x and noisy target state z_{t-1},
    and predicts the clean target label representation u_y via class simplex projection.
    """
    def __init__(self, input_dim: int, hidden_dim: int, target_dim: int = 10):
        super().__init__()
        self.image_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.state_net = nn.Sequential(
            nn.Linear(target_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.fuse_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, images: Tensor, noisy_target: Tensor, W_embed: Tensor | None = None) -> Tensor:
        images_flat = images.view(images.size(0), -1).to(dtype=torch.float32)
        h_img = self.image_net(images_flat)
        h_state = self.state_net(noisy_target.to(dtype=torch.float32))
        fused = torch.cat([h_img, h_state], dim=1)
        logits = self.fuse_net(fused)
        probs = F.softmax(logits, dim=1)
        if W_embed is not None:
            return torch.matmul(probs, W_embed)
        return probs


class NoProp(nn.Module):
    """NoProp model with T discrete diffusion blocks and output classification head."""
    def __init__(
        self,
        blocks: int = 3,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = blocks
        self.blocks = nn.ModuleList([
            NoPropBlock(input_dim, hidden_dim, num_classes) for _ in range(blocks)
        ])
        # Class embedding matrix W_Embed (fixed one-hot identity matrix)
        self.register_buffer("W_embed", torch.eye(num_classes, dtype=torch.float32))

        # Output classification head p_theta_out(y | z_T)
        self.out_head = nn.Linear(num_classes, num_classes)

        # Cosine noise schedule for discrete alpha_bar from step 0 to T
        t = torch.linspace(0, 1, blocks + 1)
        s = 0.008
        f_t = torch.cos(((t + s) / (1 + s)) * math.pi / 2).pow(2)
        alpha_bar = 1.0 - (f_t / f_t[0])
        alpha_bar = alpha_bar.clamp(0.01, 0.99)
        self.register_buffer("alpha_bar", alpha_bar)

    def get_step_coefficients(self, t: int) -> tuple[Tensor, Tensor, Tensor]:
        """Calculates Ornstein-Uhlenbeck reverse step coefficients a_t, b_t, c_t.

        formulas derived from variational diffusion ELBO:
          alpha_{t-1} = alpha_bar_{t-1} / alpha_bar_t
          a_t = sqrt(alpha_bar_t) * (1 - alpha_{t-1}) / (1 - alpha_bar_{t-1})
          b_t = sqrt(alpha_{t-1}) * (1 - alpha_bar_t) / (1 - alpha_bar_{t-1})
          c_t = (1 - alpha_bar_t) * (1 - alpha_{t-1}) / (1 - alpha_bar_{t-1})
        """
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_prev = self.alpha_bar[t - 1]
        alpha_t_minus_1 = (alpha_bar_prev / alpha_bar_t).clamp(0.001, 0.999)

        denom = (1.0 - alpha_bar_prev).clamp(min=1e-5)
        a_t = torch.sqrt(alpha_bar_t) * (1.0 - alpha_t_minus_1) / denom
        b_t = torch.sqrt(alpha_t_minus_1) * (1.0 - alpha_bar_t) / denom
        c_t = (1.0 - alpha_bar_t) * (1.0 - alpha_t_minus_1) / denom
        return a_t, b_t, c_t

    def sample_noisy_target(self, targets: Tensor, step: int) -> Tensor:
        """Forward diffusion: z_{step} ~ N(sqrt(alpha_bar_{step}) * u_y, (1 - alpha_bar_{step}) * I)."""
        alpha_bar_s = self.alpha_bar[step]
        mean = torch.sqrt(alpha_bar_s) * targets
        std = torch.sqrt(1.0 - alpha_bar_s)
        eps = torch.randn_like(targets)
        return mean + std * eps

    def step(self, x: Tensor, z_prev: Tensor, block_idx: int, add_noise: bool = False) -> Tensor:
        """Reverse diffusion step for block_idx (t = block_idx + 1) following Li et al. (2025 Eq 3)."""
        t = block_idx + 1
        a_t, b_t, c_t = self.get_step_coefficients(t)
        pred_u = self.blocks[block_idx](x, z_prev, self.W_embed)
        z_t = a_t * pred_u + b_t * z_prev
        if add_noise:
            z_t = z_t + torch.sqrt(c_t) * torch.randn_like(z_prev)
        return z_t

    def predict(self, state_T: Tensor) -> Tensor:
        """Maps final denoised state z_T to classification logits via output head p_theta_out(y | z_T)."""
        return self.out_head(state_T)


def train_noprop(
    model: NoProp,
    loader,
    epochs: int,
    lr: float,
    device: torch.device,
):
    """Trains each NoProp block independently with SNR-weighted MSE loss on target reconstruction,
    and trains the output head with cross-entropy loss on z_T.
    """
    model.to(device)
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    # Train independent diffusion dynamic blocks u_t
    for block_idx, block in enumerate(model.blocks):
        optimizer = torch.optim.AdamW(block.parameters(), lr=lr)
        step_prev = block_idx
        t = block_idx + 1

        # SNR weighting: w_t = (T / (2 * eta)) * (SNR(t) - SNR(t-1))
        snr_t = model.alpha_bar[t] / (1.0 - model.alpha_bar[t]).clamp(min=1e-5)
        snr_prev = model.alpha_bar[step_prev] / (1.0 - model.alpha_bar[step_prev]).clamp(min=1e-5)
        snr_weight = (snr_t - snr_prev).clamp(min=0.1, max=10.0).item()

        for epoch in range(1, epochs + 1):
            block.train()
            running_loss = 0.0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                b = images.size(0)

                # Clean target label representation u_y (one-hot)
                target = F.one_hot(labels, model.num_classes).float().to(device)

                # Sample noisy target z_{t-1} ~ q(z_{t-1} | y)
                noisy_target = model.sample_noisy_target(target, step_prev)

                # Independent local block prediction
                pred_target = block(images, noisy_target, model.W_embed)
                loss = snr_weight * mse_criterion(pred_target, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * b

            avg_loss = running_loss / len(loader.dataset)
            print(f"[NoProp] block {block_idx + 1}/{model.num_blocks} epoch {epoch}/{epochs} loss={avg_loss:.4f}")

    # Train output classification head p_theta_out(y | z_T) on final target state z_T ~ q(z_T | y)
    out_optimizer = torch.optim.AdamW(model.out_head.parameters(), lr=lr)
    step_T = model.num_blocks
    for epoch in range(1, epochs + 1):
        model.out_head.train()
        running_ce_loss = 0.0
        for images, labels in loader:
            labels = labels.to(device)
            b = labels.size(0)

            target = F.one_hot(labels, model.num_classes).float().to(device)
            z_T = model.sample_noisy_target(target, step_T)

            logits = model.out_head(z_T)
            loss = ce_criterion(logits, labels)

            out_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            out_optimizer.step()
            running_ce_loss += loss.item() * b

        avg_ce_loss = running_ce_loss / len(loader.dataset)
        print(f"[NoProp] output head epoch {epoch}/{epochs} cross_entropy={avg_ce_loss:.4f}")


def evaluate_noprop(model: NoProp, loader, device: torch.device, stochastic: bool = False) -> float:
    """Inference for NoProp: start from z_0 ~ N(0, I), sequentially denoise through blocks, and classify with out_head.

    Supports stochastic reverse steps with noise sqrt(c_t) * eps
    or deterministic mode (posterior mean).
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            b = images.size(0)

            # Start from pure Gaussian noise z_0 ~ N(0, I)
            state = torch.randn(b, model.num_classes, device=device, dtype=torch.float32)

            for block_idx in range(model.num_blocks):
                # Apply stochastic noise on intermediate steps
                add_noise = stochastic and (block_idx < model.num_blocks - 1)
                state = model.step(images, state, block_idx, add_noise=add_noise)

            # Classify using the learned output classification head p_theta_out(y | z_T)
            logits = model.predict(state)
            predictions.append(logits.argmax(dim=1))
            targets.append(labels)

    return accuracy(torch.cat(predictions), torch.cat(targets))
