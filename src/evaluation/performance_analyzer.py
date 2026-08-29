import os
import json
import matplotlib.pyplot as plt

class PerformanceAnalyzer:
    def __init__(self, output_dir="experiment_results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        
    def log_epoch(self, epoch, loss, accuracy, avg_spikes):
        self.results.append({
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "avg_spikes": avg_spikes
        })
