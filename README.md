# NeuroAI Assignment 2: Learning Paradigms Benchmarking

The following paradigms are implemented and tested:
1. **Backpropagation** 
2. **Supervised Forward-Forward (FF)** 
3. **Unsupervised Forward-Forward** 
4. **Discrete-Time NoProp (NoProp-DT)** 
5. **Denoising Diffusion Classifier**

Backpropagation and Diffusion are kept as baseline.

---

## Environment Setup

Create a virtual environment and run: `python main.py --help`

---

## Benchmark Commands

### Default Iso-Parameter Benchmark (Single Invocation — Baseline)
Runs all 5 learning paradigms across all 6 goodness functions using architecture that keeps parameters almost constant across all paradigms (567k parameters appex) for 10 epochs with GPU acceleration.

```python
python main.py
```

Under this default single invocation:
- **Normal MLP (Backprop):** 567,434 params (`--mlp-hidden-dims 512 256 128`)
- **Supervised FF:** 566,144 params (`--ff-hidden-dims 512 256 128`) 
- **Unsupervised FF:** 566,144 params + linear probe 
- **NoProp-DT:** 568,270 params (4 blocks, `--noprop-hidden-dim 140`) 
- **Diffusion Classifier:** 577,680 params (`--diffusion-hidden-dim 288`) 

---

### Optimised Benchmark
We reduce the parameter capacity of each model down to ~220k–235k parameters while preserving classification accuracy within $\pm 1.5\%$ error tolerance

```bash
python main.py --preset optimised --output results_optimized.json --plot-dir plots_optimized
```

---

### Exploring Forward-Forward Variants

#### A. Compare All 6 Goodness Functions

```bash
python main.py \
  --ff-goodness-modes squared negative_squared sparse margin spectral hebbian_wire \
  --epochs 5 \
  --train-subset 2048 \
  --test-subset 512
```

#### B. Supervised FF with Hard Negative Mining
```bash
python main.py \
  --ff-goodness-modes squared \
  --ff-hard-negatives \
  --epochs 5 \
  --train-subset 2048 \
  --test-subset 512
```

#### C. Supervised FF Accumulating All Layers
```bash
python main.py \
  --ff-goodness-modes squared \
  --ff-include-first-layer \
  --epochs 5 \
  --train-subset 2048 \
  --test-subset 512
```