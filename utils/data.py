import math
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


def make_loaders(data_dir: str, batch_size: int, train_subset: int, test_subset: int):
    """Loads MNIST with standard normalization. Also returns unnormalize utility."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    if train_subset > 0:
        train_dataset = Subset(
            train_dataset, range(min(train_subset, len(train_dataset)))
        )
    if test_subset > 0:
        test_dataset = Subset(test_dataset, range(min(test_subset, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


def unnormalize_mnist(tensor: torch.Tensor) -> torch.Tensor:
    """Inverts MNIST normalization (mean=0.1307, std=0.3081) and clamps to [0, 1]."""
    return (tensor * 0.3081 + 0.1307).clamp(0.0, 1.0)


def make_synthetic_2d_loader(
    n_samples: int = 1200,
    n_classes: int = 3,
    noise: float = 0.2,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, TensorDataset, TensorDataset]:
    """Generates a synthetic multi-class 2D spiral/pinwheel dataset.

    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    points_per_class = n_samples // n_classes
    x_list = []
    y_list = []

    for c in range(n_classes):
        r = torch.linspace(0.1, 1.0, points_per_class)
        t = torch.linspace(
            c * (2 * math.pi / n_classes),
            (c + 1.8) * (2 * math.pi / n_classes),
            points_per_class,
        )
        t = t + torch.randn(points_per_class) * noise

        x1 = r * torch.sin(t)
        x2 = r * torch.cos(t)
        points = torch.stack([x1, x2], dim=1)
        labels = torch.full((points_per_class,), c, dtype=torch.long)

        x_list.append(points)
        y_list.append(labels)

    x_all = torch.cat(x_list, dim=0).float()
    y_all = torch.cat(y_list, dim=0).long()

    # Shuffle
    perm = torch.randperm(x_all.size(0))
    x_all = x_all[perm]
    y_all = y_all[perm]

    split = int(0.8 * len(x_all))
    train_ds = TensorDataset(x_all[:split], y_all[:split])
    test_ds = TensorDataset(x_all[split:], y_all[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ds, test_ds
