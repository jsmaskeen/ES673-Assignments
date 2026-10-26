# ES 673 Assignment 4: Spiking Neural Networks

This repository contains the complete implementation, experimental suites, unit tests, and interactive demonstration notebook for Assignment 4 on Spiking Neural Networks (SNNs).

---

## Repository Structure

```
.
├── Take Home Assignment 4.md     # Assignment prompt and specifications
├── q2_snn_mnist.ipynb            # Fully executed demonstration notebook (Question 2)
├── pyproject.toml                # Project metadata and dependencies (managed via uv)
├── uv.lock                       # Deterministic dependency lockfile
├── .python-version               # Python version pinning (3.12)
├── src/                          # Reusable library and experiment scripts
│   ├── snn_common.py             # Data loading, Poisson/TTFS encoders, Brian 2 simulation, 
│   │                             # parallel calibration, diagnostics, and 5-fold CV readout
│   ├── run_q2_experiments.py     # Multi-seed experimental suite (baseline, tau sweep, sparsity)
│   ├── exp_encoding_check.py     # Encoding check: noise, jitter, spike drop, rare-event benchmark
│   ├── fetch_mnist.py            # Local MNIST download and caching utility
│   ├── bench_backends.py         # Brian 2 backend benchmarking (Cython vs cpp_standalone)
│   ├── create_notebook.py        # Synchronized notebook generator
│   └── test_sparse_compensation.py # Rapid verification of fan-in compensation scaling
└── tests/                        # Automated test suite
    └── test_snn.py               # 10 unit tests covering encoders, refractoriness, 
                                  # silence, chunk invariance, calibration, and diagnostics
```

---

## Environment Setup

The project uses Python 3.12 and is configured for [`uv`](https://github.com/astral-sh/uv). A C++ compiler (such as Microsoft Visual C++ Build Tools on Windows or GCC/Clang on Linux/macOS) is required for Brian 2's `cpp_standalone` code generation.

### 1. Using `uv` (Recommended)

Clone the repository and install all dependencies:
```bash
# Sync all dependencies exactly as pinned
uv sync
```

### 2. Using standard `pip`

Alternatively, install dependencies via `pip`:
```bash
pip install brian2 numpy scipy scikit-learn matplotlib pandas pytest jupyter nbconvert seaborn
```

---

## Running the Code

### 1. Run the Unit Test Suite (10 Automated Tests)
Verifies spike encoders, refractoriness bounds, zero-input silence, vectorized chunk invariance, analytical calibration bracketing, and diagnostic metrics:
```bash
uv run pytest tests/test_snn.py
```

### 2. Execute the Interactive Demonstration Notebook
Launch or execute the full demonstration notebook end-to-end:
```bash
# Launch interactive Jupyter server
uv run jupyter notebook q2_snn_mnist.ipynb
```

### 3. Run Question 2 Multi-Seed Experiment Suite
Evaluates baseline dense SNN, uniform weights, signed control, membrane time constant ($\tau_m \in \{5, 10, 20, 50, 100\}\text{ ms}$) sweeps, and sparse connectivity ($p=0.05$) across 3 independent random seeds (`[42, 43, 44]`):
```bash
uv run python src/run_q2_experiments.py
```
*Results, summary CSVs, and diagnostic figures will be generated inside the `results/` folder.*

### 4. Run Phase 2 Encoding Check
Compares Poisson Rate vs. Latency (TTFS) coding under pixel noise, spike deletion, timing jitter, and streaming rare-event detection (PR-AUC and Target Recall @ 1% FAR):
```bash
uv run python src/exp_encoding_check.py
```

---

## Key Experimental Results Summary

* **Raw Pixel Reference (Linear Upper Bound):** $90.27 \pm 0.00\%$
* **Baseline SNN (Dense $N(0.5, 0.1)$, $\tau_m=20$ ms):** $48.80 \pm 0.72\%$
  * Neurons collapse onto a single common-mode total-ink detector ($R_{\text{eff}} = 1.0$, pairwise response correlation $0.9831$).
* **Dense Uniform Weights ($U(0, 1)$):** $79.29 \pm 0.67\%$ (+30.5 pts over baseline)
* **Dense Signed Control ($N(0, 0.25)$):** $77.27 \pm 0.77\%$ (+28.5 pts over baseline)
* **Extra Credit 1 ($\tau_m$ Sweep):**
  * Fast membrane ($\tau_m = 5\text{ ms}$): **$57.02 \pm 1.48\%$** (coincidence detection)
  * Slow membrane ($\tau_m = 100\text{ ms}$): **$35.84 \pm 0.34\%$** (temporal low-pass blurring)
* **Extra Credit 2 (Sparse Connectivity, $p=0.05$):** **$87.40 \pm 0.48\%$** (+38.6 pts over baseline)
  * Breaks common-mode collapse ($R_{\text{eff}} = 4.1$, response correlation $0.46$) with **20.3× fewer synapses** (15,452 vs 313,600).
