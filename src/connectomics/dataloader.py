from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_mnist_loaders(batch_size=64, num_train=1000, num_test=200):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_subset = Subset(train_dataset, range(num_train))
    test_subset = Subset(test_dataset, range(num_test))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
