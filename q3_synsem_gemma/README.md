# Gemma-2 Syntax vs. Semantics Analysis Framework

A multi-task empirical probing suite for **Google Gemma-2-2B** evaluating syntactic structure vs. semantic meaning, layer-wise representational crystallization across all 27 layers, and causal dissociation under ablation surgery.

---

## 1. Quickstart & Setup

### 1.1 Install Dependencies
From within the `q3_synsem_gemma` directory:
```powershell
pip install -r requirements.txt
```

### 1.2 Configure Hugging Face Authentication Token
Gemma-2 is a gated model. Set your token in your shell environment:
```powershell
# PowerShell (Windows)
$env:HF_TOKEN = "blah blah"   # Replace 'blah blah' with your actual HF token

# Command Prompt (cmd.exe)
set HF_TOKEN=blah blah

# Bash / Linux
export HF_TOKEN="blah blah"
```

---

## 2. Running Probing Experiments

All commands are executed from the `q3_synsem_gemma` directory using Python module syntax (`-m`).

### 2.1 Run the Full Probing Battery Across 27 Layers
Evaluates all datasets (BLiMP 10 subsets, STS-B, WiC, HellaSwag, Colorless-Green, and RSA matrix):
```powershell
python -m synsem.pipeline --experiment layer_probes --max-samples 50 --outdir outputs
```

### 2.2 Run Individual Probes
```powershell
# 1. BLiMP Advanced Syntactic Minimal Pairs (10 Subsets)
python -m synsem.pipeline --experiment blimp --max-samples 50 --outdir outputs

# 2. STS-B Semantic Textual Similarity (Layer-by-Layer Spearman rho)
python -m synsem.pipeline --experiment stsb --max-samples 50 --outdir outputs

# 3. SuperGLUE WiC Contextual Polysemy Disambiguation
python -m synsem.pipeline --experiment wic --max-samples 50 --outdir outputs

# 4. Colorless-Green Chomskyan Dissociation (Type A / B / C / D)
python -m synsem.pipeline --experiment chomsky --max-samples 50 --outdir outputs

# 5. 27x27 Inter-Layer Representational Similarity Analysis (RSA)
python -m synsem.pipeline --experiment rsa --max-samples 24 --outdir outputs
```

### 2.3 Run Causal Dissociation Under Ablation Surgery (Connecting Q2 & Q3)
Applies ablation surgery (Early vs. Late layers, Q vs. KV heads) from `common_gemma` to test how syntactic and semantic capabilities causally break:
```powershell
python -m synsem.pipeline --experiment causal_dissociation --max-samples 50 --outdir outputs
```

---

## 3. Results Collation & Creative Visualizations

### 3.1 Collate Outputs into Markdown Summary
```powershell
python -m analysis.collate --input-dir outputs/ --output-md outputs/q3_summary_table.md
```

### 3.2 Generate Publication-Grade Visualizations
Generates creative, high-resolution figures in `plots/`:
```powershell
python -m analysis.visualize --input-dir outputs/ --output-dir plots/
```

### Figures Generated:
1. **`fig1_blimp_accuracy.png`**: BLiMP accuracy across all 10 advanced subsets with linguistic categorization (Islands, Binding, Agreement, etc.).
2. **`fig2_stsb_layer_evolution.png`**: Spearman rank correlation $\rho$ plotted across all 27 layers showing where semantic similarity peaks.
3. **`fig3_wic_polysemy_resolution.png`**: Token cosine curves for Same-Sense vs. Different-Sense contexts across layers, isolating polysemy resolution.
4. **`fig4_colorless_green_dissociation.png`**: Perplexity distribution across Chomskyan quadrants (Type A, B, C, D).
5. **`fig5_word_surprisal_profiles.png`**: Token-level surprisal step plot highlighting exact syntactic violation spikes.
6. **`fig7_layer_rsa_matrix.png`**: $27 \times 27$ Inter-layer representational similarity matrix heatmap.
7. **`fig9_causal_dissociation.png`**: Causal surgery impact: Early vs. Late layer ablation on BLiMP Syntax vs. STS-B Semantics.

---

## 4. Architecture & Shared Core

Both Question 2 and Question 3 leverage the root `common_gemma` package:
```
assn3/
├── common_gemma/             # Unified Gemma-2 core
│   ├── backbone.py           # Model loader, eager mode, bfloat16, HF token
│   ├── surgery.py            # GQA head surgery, position-aware layer surgery
│   └── prober.py             # 27-layer hidden state & token extraction
├── q2_ablation_gemma/        # Question 2: Architectural Ablation Study
└── q3_synsem_gemma/          # Question 3: Syntax vs. Semantics Analysis
```
