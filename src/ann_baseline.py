import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.connectomics.dataloader import get_mnist_loaders

class SimpleANN(nn.Module):
    """
    Standard Artificial Neural Network baseline.
    Same dimensions as the SNN (784 -> 128 -> 10).
    """
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def run_ann_baseline(epochs=15, output_dir="experiment_results"):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleANN().to(device)
    
    train_loader, _ = get_mnist_loaders(batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    loss_hist, acc_hist = [], []
    
    print("Training ANN with Backpropagation")
    print("="*50)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            
            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx >= 30: # Keeping the same artificial batch limit as the modified perceptron for a fair comparison
                break
                
        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        
        loss_hist.append(avg_loss)
        acc_hist.append(acc)
        
        print(f"ANN Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
        
    print("Saving Baseline Analytics...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(loss_hist, color='blue', marker='o', label='ANN Loss')
    ax1.set_title("ANN Training Loss", fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(acc_hist, color='green', marker='o', label='ANN Accuracy')
    ax2.set_title("ANN Training Accuracy", fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ann_baseline_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"ANN metrics plot saved to {plot_path}")
    print("\nBaseline execution complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Standard ANN Baseline")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    args = parser.parse_args()
    
    run_ann_baseline(epochs=args.epochs)
