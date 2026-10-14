# Gemma-2 Architectural Ablation Framework

A standalone, reproducible ablation study suite for **Google Gemma-2-2B** investigating Grouped Query Attention (GQA) redundancy, depth-position sensitivity, and syntactic vs. semantic functional dissociation.

---

## 1. Quickstart & Environment Setup

### 1.1 Install Dependencies
From within the `q2_ablation_gemma` directory, install required packages:
```powershell
pip install -r requirements.txt
```

### 1.2 Configure Hugging Face Authentication Token
Gemma-2 is a gated model. You must accept the model terms on Hugging Face ([huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)).

Set your access token in your terminal environment:
```powershell
# PowerShell (Windows)
$env:HF_TOKEN = "blah blah"   # Replace 'blah blah' with your actual HF token

# Command Prompt (cmd.exe)
set HF_TOKEN=blah blah

# Bash / Linux
export HF_TOKEN="blah blah"
```
> **Note:** The code has a built-in fallback: if no token is found in the environment, it defaults to `"blah blah"`. You can also pass the token directly via `--token "your_token"`.

---

## 2. Running Experiments

All commands are executed from the `q2_ablation_gemma` directory using Python module syntax (`-m`).

### 2.1 Step 1: Run the Sanity Check
Verify model loading, eager attention mode, forward hook installation, and clean hook detachment:
```powershell
python -m gemma.orchestrator --sanity-check
```

### 2.2 Step 2: Run Baseline Evaluations (Zero Ablation)
Evaluates unablated Gemma-2-2B across all three downstream tasks (Fluency, IMDB Valence, and Garden-Path Syntax):
```powershell
python -m gemma.orchestrator --experiment experiments.baselines --max-samples 100 --outdir outputs
```

### 2.3 Step 3: Run GQA Head Ablation Sweeps
Ablates Query heads (`q_only`), Key/Value groups (`kv_group`), and matched full heads (`full_head`) across severities $[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]$:
```powershell
python -m gemma.orchestrator --experiment experiments.head_sweeps --max-samples 100 --outdir outputs
```

### 2.4 Step 4: Run Position-Aware Layer Ablation Sweeps
Ablates early layers ($0..k$), middle layers ($13 \pm k/2$), late layers ($26-k..26$), and random layers across Attention, MLP, and full Decoder Blocks:
```powershell
python -m gemma.orchestrator --experiment experiments.layer_sweeps --max-samples 100 --outdir outputs
```

### 2.5 Step 5: Run Embedding Masking Sweeps
Masks input embedding dimensions ($0\%$ to $100\%$):
```powershell
python -m gemma.orchestrator --experiment experiments.emb_sweeps --max-samples 100 --outdir outputs
```

### 2.6 Step 6 (New Parameter): Run Perturbation Modality Sweeps
Evaluates the effect of **structural silence (`zero`)** vs. **synaptic jitter (`gaussian_noise`)** vs. **tonic baseline excitation (`mean_clamp`)** across attention heads and layers:
```powershell
uv run python -m gemma.orchestrator --experiment experiments.perturbation_sweeps --max-samples 100 --outdir outputs
```

### 2.7 Step 7 (New Parameter): Run Spectral PCA Subspace Sweeps
Projects out the top-$k$ principal geometric variance directions from the embedding space:
```powershell
uv run python -m gemma.orchestrator --experiment experiments.spectral_sweeps --max-samples 100 --outdir outputs
```

---

## 3. Results Collation & Creative Visualizations

### 3.1 Collate Outputs into CSV & Markdown
Aggregates all Parquet and JSON files in `outputs/` into unified summary tables:
```powershell
python -m analysis.collate --input-dir outputs/ --output-csv outputs/summary_all_runs.csv --output-agg outputs/summary_aggregated.csv --output-md outputs/summary_table.md
```

### 3.2 Generate Publication-Grade Visualizations
Generates 5 publication-ready figures in `plots/`:
```powershell
python -m analysis.visualize --input-dir outputs/ --output-dir plots/
```

### Figures Generated:
1. **`fig1_waterfall_degradation.png`**: Cross-task performance waterfall curves (WikiText-2 PPL, IMDB Valence Accuracy, Garden-Path Penalty Ratio vs. Severity).
2. **`fig2_gqa_sensitivity.png`**: GQA architectural vulnerability: Query-only vs. KV-group vs. Full-head degradation.
3. **`fig3_position_resilience.png`**: Position-aware depth resilience profile (Early vs. Middle vs. Late vs. Random layers).
4. **`fig4_functional_dissociation.png`**: 2D scatter plot mapping Semantic Accuracy vs. Syntactic Parsing Accuracy across ablated layers.
5. **`fig5_failure_horizons.png`**: Critical failure thresholds ($50\%$ drop) across architectural components.

---

## 4. Package Architecture

```
q2_ablation_gemma/
├── gemma/
│   ├── backbone.py           # Model loader, eager attention mode, HF token handler
│   ├── surgery.py            # GQA head surgery, position layer surgery, embedding masking
│   ├── orchestrator.py       # Experiment runner, context managers, Parquet exporter
│   └── tasks/
│       ├── fluency.py        # WikiText-2: Perplexity & Bits-Per-Character
│       ├── valence.py        # IMDB: Generative forced-choice sentiment accuracy
│       └── structural.py     # Garden-path & long-range agreement syntactic probes
├── experiments/
│   ├── baselines.py          # Zero-ablation baseline configurations
│   ├── head_sweeps.py        # GQA Q-only vs KV-group vs Full-head sweeps
│   ├── layer_sweeps.py       # Position-aware layer sweeps (Attn / MLP / Block)
│   └── emb_sweeps.py         # Embedding dimension masking sweeps
├── analysis/
│   ├── collate.py            # Parquet aggregation into summary CSV and Markdown
│   └── visualize.py          # Creative visualization suite (6 figures)
├── outputs/                  # Auto-generated Parquet and JSON results
└── plots/                    # Auto-generated PNG figures (300 DPI)
```
