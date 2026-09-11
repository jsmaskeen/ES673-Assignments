import os

import torch
from torch import Tensor, nn
from torchvision.utils import save_image

from utils.metrics import accuracy


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    x = torch.linspace(0, 1, timesteps + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x + s) / (1 + s)) * torch.pi / 2).pow(2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999)


class DenoisingDiffusionClassifier(nn.Module):
    """Conditional Denoising Diffusion Classifier.

    Learns conditional score field eps_theta(x_t, t, c). Classification is performed
    via Bayes rule by comparing denoising reconstruction error across candidate classes.
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.time_emb = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.label_emb = nn.Embedding(num_classes, 64)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.noise_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, noisy: Tensor, timesteps: Tensor, labels: Tensor) -> Tensor:
        noisy = noisy.to(dtype=torch.float32)
        time_emb = self.time_emb(timesteps.to(dtype=torch.float32).unsqueeze(1) / 100.0)
        label_emb = self.label_emb(labels.to(dtype=torch.long))
        cond = torch.cat([time_emb, label_emb], dim=1).to(dtype=torch.float32)
        features = torch.cat([noisy, cond], dim=1)
        latent = self.encoder(features)
        return self.noise_head(latent)


def train_diffusion(model: DenoisingDiffusionClassifier, loader, epochs: int, lr: float, device: torch.device, total_steps: int = 30):
    """Trains conditional diffusion model via standard score-matching MSE objective."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    betas = cosine_beta_schedule(total_steps).to(device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            clean = images.view(images.size(0), -1).to(device=device, dtype=torch.float32)
            t = torch.randint(0, total_steps, (clean.size(0),), device=device)
            noise = torch.randn_like(clean)
            alpha_t = alphas_cumprod[t].view(-1, 1).to(device=device, dtype=torch.float32)
            noisy = torch.sqrt(alpha_t) * clean + torch.sqrt(1.0 - alpha_t) * noise
            noise_pred = model(noisy, t, labels)
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f"[Diffusion] epoch {epoch}/{epochs} denoise_loss={avg_loss:.4f}")


def evaluate_diffusion(
    model: DenoisingDiffusionClassifier,
    loader,
    device: torch.device,
    total_steps: int = 30,
    eval_timesteps: list[int] | None = None,
) -> float:
    """
    Classifies images by evaluating conditional score error ||eps_theta(x_t, t, c) - eps||^2 across all classes.
    The class yielding the lowest error (highest likelihood under conditional diffusion prior) is chosen.
    """
    model.eval()
    predictions = []
    targets = []
    betas = cosine_beta_schedule(total_steps).to(device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    if eval_timesteps is None:
        eval_timesteps = [max(0, int(total_steps * r)) for r in [0.20, 0.50, 0.80]]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            b = images.size(0)
            clean = images.view(b, -1).to(dtype=torch.float32)

            class_errors = torch.zeros((b, model.num_classes), device=device)

            for step in eval_timesteps:
                t = torch.full((b,), step, device=device, dtype=torch.long)
                alpha_t = alphas_cumprod[step].view(1, 1).to(device=device)
                noise = torch.randn_like(clean)
                noisy = torch.sqrt(alpha_t) * clean + torch.sqrt(1.0 - alpha_t) * noise

                for c in range(model.num_classes):
                    cand_c = torch.full((b,), c, device=device, dtype=torch.long)
                    noise_pred = model(noisy, t, cand_c)
                    err = (noise_pred - noise).pow(2).mean(dim=1)
                    class_errors[:, c] += err

            preds = class_errors.argmin(dim=1)
            predictions.append(preds)
            targets.append(labels)

    return accuracy(torch.cat(predictions), torch.cat(targets))


def sample_diffusion(model: DenoisingDiffusionClassifier, class_label: int, count: int, device: torch.device, output_path: str, total_steps: int = 30):
    model.eval()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    samples = torch.randn(count, 784, device=device, dtype=torch.float32)
    labels = torch.full((count,), class_label, device=device, dtype=torch.long)
    betas = cosine_beta_schedule(total_steps).to(device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for step in range(total_steps - 1, -1, -1):
        t = torch.full((count,), step, device=device, dtype=torch.long)
        beta_s = betas[step]
        alpha_s = alphas[step]
        alpha_bar_s = alphas_cumprod[step]

        with torch.no_grad():
            noise_pred = model(samples, t, labels)
            # Canonical DDPM reverse step
            samples = (samples - (beta_s / torch.sqrt(1.0 - alpha_bar_s)) * noise_pred) / torch.sqrt(alpha_s)
        if step > 0:
            alpha_bar_prev = alphas_cumprod[step - 1]
            sigma_t = torch.sqrt(beta_s * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_s))
            samples += sigma_t * torch.randn_like(samples)

    grid = samples.view(-1, 1, 28, 28)
    save_image(grid.clamp(0, 1), output_path, nrow=4)
    print(f"[Diffusion] saved samples to {output_path}")
