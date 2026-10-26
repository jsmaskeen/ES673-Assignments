import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from src.snn_common import load_mnist, encode_poisson, simulate_snn, train_and_evaluate_readout

data = load_mnist(n_train=2000, n_val=500, n_test=1000, seed=42)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
X_val = data["X_val"]

N_in, N_hid = 784, 400
p_conn = 0.05
rng = np.random.RandomState(42)

# Sample sparse weights
mask = rng.rand(N_in, N_hid) < p_conn
raw_w = np.full((N_in, N_hid), np.nan, dtype=np.float32)
# Using normal distribution
raw_w[mask] = np.clip(rng.normal(0.5, 0.1, size=mask.sum()), 0.0, None)

spk_idx_val, spk_t_val = encode_poisson(X_val[:200], seed=42)

# Sweep scales around 1.0 to 4.0 (20x compensation)
scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
best_s, best_diff = 1.0, 999.0
for s in scales:
    w_mat = raw_w * s
    sim = simulate_snn(spk_idx_val, spk_t_val, 200, w_matrix=w_mat, build_dir="build_sparse_calib")
    rate = sim["mean_rate_hz"]
    print(f"Sparse scale {s:.2f} -> Mean Rate: {rate:.1f} Hz")
    if abs(rate - 25.0) < best_diff:
        best_diff = abs(rate - 25.0)
        best_s = s

print(f"Optimal compensated sparse scale: {best_s}")
w_sparse_opt = raw_w * best_s

spk_idx_tr, spk_t_tr = encode_poisson(X_train, seed=42)
spk_idx_te, spk_t_te = encode_poisson(X_test, seed=100)

sim_tr = simulate_snn(spk_idx_tr, spk_t_tr, len(X_train), w_matrix=w_sparse_opt, build_dir="build_sp_tr_opt")
sim_te = simulate_snn(spk_idx_te, spk_t_te, len(X_test), w_matrix=w_sparse_opt, build_dir="build_sp_te_opt")

res = train_and_evaluate_readout(sim_tr["spike_counts"], y_train, sim_te["spike_counts"], y_test)
acc = res["accuracy"]
rate = sim_te["mean_rate_hz"]
print(f"Compensated Sparse SNN (p=0.05, Normal) Accuracy: {acc*100:.2f}% | Test Rate: {rate:.1f} Hz")
