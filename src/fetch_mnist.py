"""
Script to download and cache the MNIST dataset to data/mnist.npz.
Normalises pixel values to [0.0, 1.0] as float32.
"""

import os
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml


def prepare_mnist(data_dir: Path = Path("data")) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    target_file = data_dir / "mnist.npz"

    if target_file.exists():
        print(f"MNIST cache already exists at {target_file.resolve()}")
        data = np.load(target_file)
        print(f"Loaded: X_train {data['X_train'].shape}, y_train {data['y_train'].shape}, "
              f"X_test {data['X_test'].shape}, y_test {data['y_test'].shape}")
        return target_file

    print("Fetching MNIST (mnist_784) via scikit-learn OpenML...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target

    # Convert to appropriate types
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.uint8)

    # Standard split: first 60k train, remaining 10k test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"Saving compressed data to {target_file}...")
    np.savez_compressed(
        target_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    print(f"Done! Saved {target_file.stat().st_size / (1024*1024):.2f} MB")
    return target_file


if __name__ == "__main__":
    prepare_mnist()
