# Bio-Inspired Modified Perceptron 
## ES 673 Assignment 1

**Author:** Jaskirat Singh Maskeen

## Setup and Installation
It is recommended to use `uv` or `venv` to create a virtual environment for the project.

1. **Create and activate the environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   # source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy matplotlib pandas scipy scikit-learn snntorch
   ```

## Running the Code

### 1. The Bio-Inspired Perceptron
To run the main experiment with default biological parameters:
```bash
python -m src.main --epochs 50 --time-steps 20
```
*You can customize biological parameters via CLI arguments (e.g., `--lambda-constant`, `--tau-m`, `--U`, `--gamma`, `--surrogate-k`).*

### 2. The Standard ANN Baseline
To run the standard feed-forward ANN for comparison:
```bash
python -m src.ann_baseline --epochs 15
```

### 3. Biological Parameter Sweeps
To run automated 1D sweeps across the various biological constants to observe their impact on performance:
```bash
python sweep_experiments.py --epochs 10 --time-steps 20
```

### 4. Test Sample
```bash
python python -m src.test_model --sample 4
```

## Results
All experimental outputs, diagnostic plots, and saved models are automatically generated and stored in the **`experiment_results/`** directory.
