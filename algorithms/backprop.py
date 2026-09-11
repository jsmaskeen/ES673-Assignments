import torch
from torch import Tensor, nn

from utils.metrics import accuracy


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_size: int | None = None,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is not None:
            self.hidden_dims = list(hidden_dims) if isinstance(hidden_dims, (list, tuple)) else [hidden_dims]
        elif hidden_size is not None:
            self.hidden_dims = [hidden_size]
        else:
            self.hidden_dims = [512, 256, 128]

        self.input_size = input_size
        self.num_classes = num_classes

        self.layer_blocks = nn.ModuleList()
        in_dim = input_size
        for h_dim in self.hidden_dims:
            block = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            self.layer_blocks.append(block)
            in_dim = h_dim
        self.out_linear = nn.Linear(in_dim, num_classes)

    def get_layer_features(self, x: Tensor) -> list[Tensor]:
        """Returns intermediate representations after each hidden block."""
        x_flat = x.view(x.size(0), -1)
        feats = []
        curr = x_flat
        for block in self.layer_blocks:
            curr = block(curr)
            feats.append(curr)
        return feats

    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.view(x.size(0), -1)
        curr = x_flat
        for block in self.layer_blocks:
            curr = block(curr)
        return self.out_linear(curr)


def train_backprop(
    model: MLPClassifier,
    loader,
    epochs: int,
    lr: float,
    device: torch.device,
    val_loader=None,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(loader.dataset)
        if val_loader is not None:
            acc = evaluate_backprop(model, val_loader, device)
            print(f"[Backprop] epoch {epoch}/{epochs} loss={avg_loss:.4f} val_acc={acc:.4f}")
        else:
            acc = evaluate_backprop(model, loader, device)
            print(f"[Backprop] epoch {epoch}/{epochs} loss={avg_loss:.4f} train_acc={acc:.4f}")


def evaluate_backprop(model: MLPClassifier, loader, device: torch.device) -> float:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions.append(logits.argmax(1))
            targets.append(labels)

    all_pred = torch.cat(predictions)
    all_target = torch.cat(targets)
    return accuracy(all_pred, all_target)
