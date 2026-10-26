"""
Main execution script for Question 2: SNN on MNIST in Brian2.
Performs comprehensive multi-seed (seeds 42, 43, 44) evaluation:
1. Baseline simulation with dense N(0.5, 0.1) weights and linear readout (Logistic + Ridge with CV).
2. Weight initialization comparison: Normal vs Uniform random weights, plus zero-mean signed control.
3. Extra Credit 1: Systematic tau_m sweep (5, 10, 20, 50, 100 ms) with spike-timing metrics (first-spike latency, ISI CV) and dual voltage traces.
4. Extra Credit 2: Sparse connectivity (p = 0.05) with fan-in compensation, receptive fields, side-by-side digit templates, and Spike-Triggered Averages (STAs).
5. Non-spiking controls: Raw pixels reference and analytic infinite-time LIF rate representation.
6. Mechanistic diagnostics: Hidden neuron pairwise response correlation, effective rank, and ink correlation.
Saves all figures, spike features, and tabular summaries to results/.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.snn_common import (
    load_mnist,
    encode_poisson,
    simulate_snn,
    calibrate_weight_scale,
    train_and_evaluate_readout,
    compute_response_diagnostics,
    compute_analytic_lif_rate
)


def run_all_q2_experiments(seeds=(42, 43, 44)):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 68)
    print("=== Question 2: Brian2 SNN Feedforward MNIST Multi-Seed Suite ===")
    print(f"=== Running Seeds: {seeds} ===")
    print("=" * 68)

    # 1. Load Data
    print("\n[Step 1] Loading MNIST Dataset...")
    splits = load_mnist(n_train=5000, n_val=500, n_test=1500, seed=42)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val = splits["X_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    N_in = 784
    N_hid = 400
    T_enc_ms = 100.0
    dt_ms = 0.5
    tau_m_baseline = 20.0
    t_ref_ms = 5.0
    tau_values = [5.0, 10.0, 20.0, 50.0, 100.0]

    # Pre-encode Poisson spikes for reproducibility across seeds
    print("\n[Step 2] Encoding Train and Test images to Poisson Spikes...")
    spk_idx_tr, spk_t_tr = encode_poisson(X_train, T_enc_ms=T_enc_ms, dt_ms=dt_ms, seed=42)
    spk_idx_te, spk_t_te = encode_poisson(X_test, T_enc_ms=T_enc_ms, dt_ms=dt_ms, seed=100)

    # Data structures to accumulate metrics across seeds
    all_metrics = {
        "raw_pixels": [],
        "baseline_logistic": [],
        "baseline_ridge": [],
        "analytic_lif": [],
        "uniform_logistic": [],
        "signed_logistic": [],
        "sparse_logistic": [],
    }
    for tau in tau_values:
        all_metrics[f"tau_fixed_{int(tau)}ms"] = []
        all_metrics[f"tau_matched_{int(tau)}ms"] = []

    # Store representative artifacts from seed 42 for figures
    rep_artifacts = {}

    for seed_idx, seed in enumerate(seeds):
        print("\n" + "#" * 60)
        print(f"### RUNNING SEED {seed} ({seed_idx + 1}/{len(seeds)}) ###")
        print("#" * 60)

        # ---------------------------------------------------------
        # Non-spiking Control 0: Raw Pixels + Logistic Regression CV
        # ---------------------------------------------------------
        res_raw = train_and_evaluate_readout(X_train, y_train, X_test, y_test, model_type="logistic", use_cv=True, seed=seed)
        diag_raw = compute_response_diagnostics(X_test, X_test)
        all_metrics["raw_pixels"].append({
            "acc": res_raw["accuracy"],
            "f1": res_raw["macro_f1"],
            "rate": np.nan,
            "resp_corr": diag_raw["mean_response_corr"],
            "eff_rank": diag_raw["effective_rank"],
            "ink_corr": diag_raw["mean_ink_corr"],
            "synapses": N_in
        })
        print(f"Raw Pixels (Logistic CV): Acc = {res_raw['accuracy']*100:.2f}%, F1 = {res_raw['macro_f1']:.4f}")

        # ---------------------------------------------------------
        # Experiment 1: Baseline SNN (Dense, N(0.5, 0.1), tau_m = 20ms)
        # ---------------------------------------------------------
        print(f"\n--- [Seed {seed}] Baseline SNN (Dense Normal) ---")
        scale_base, w_base = calibrate_weight_scale(
            X_val, target_rate_hz=25.0, N_hid=N_hid, tau_m_ms=tau_m_baseline,
            connectivity="dense", weight_dist="normal", seed=seed,
            build_dir=f"build_calib_base_s{seed}"
        )

        sim_base_tr = simulate_snn(
            spk_idx_tr, spk_t_tr, len(X_train),
            N_in=N_in, N_hid=N_hid, tau_m_ms=tau_m_baseline, t_ref_ms=t_ref_ms,
            w_matrix=w_base, build_dir=f"build_base_tr_s{seed}"
        )
        sim_base_te = simulate_snn(
            spk_idx_te, spk_t_te, len(X_test),
            N_in=N_in, N_hid=N_hid, tau_m_ms=tau_m_baseline, t_ref_ms=t_ref_ms,
            w_matrix=w_base, build_dir=f"build_base_te_s{seed}",
            record_traces=(seed == seeds[0]), return_spike_timing=True
        )

        X_tr_base = sim_base_tr["spike_counts"]
        X_te_base = sim_base_te["spike_counts"]

        res_lr_base = train_and_evaluate_readout(X_tr_base, y_train, X_te_base, y_test, model_type="logistic", use_cv=True, seed=seed)
        res_rg_base = train_and_evaluate_readout(X_tr_base, y_train, X_te_base, y_test, model_type="ridge", use_cv=True, seed=seed)
        diag_base = compute_response_diagnostics(X_te_base, X_test)

        all_metrics["baseline_logistic"].append({
            "acc": res_lr_base["accuracy"], "f1": res_lr_base["macro_f1"],
            "rate": sim_base_te["mean_rate_hz"], "resp_corr": diag_base["mean_response_corr"],
            "eff_rank": diag_base["effective_rank"], "ink_corr": diag_base["mean_ink_corr"],
            "synapses": N_in * N_hid, "scale": scale_base
        })
        all_metrics["baseline_ridge"].append({
            "acc": res_rg_base["accuracy"], "f1": res_rg_base["macro_f1"],
            "rate": sim_base_te["mean_rate_hz"], "resp_corr": diag_base["mean_response_corr"],
            "eff_rank": diag_base["effective_rank"], "ink_corr": diag_base["mean_ink_corr"],
            "synapses": N_in * N_hid, "scale": scale_base
        })

        print(f"Baseline Logistic: Acc = {res_lr_base['accuracy']*100:.2f}%, Rate = {sim_base_te['mean_rate_hz']:.2f} Hz, EffRank = {diag_base['effective_rank']:.1f}, RespCorr = {diag_base['mean_response_corr']:.4f}")
        print(f"Baseline Ridge:    Acc = {res_rg_base['accuracy']*100:.2f}%")

        if seed == seeds[0]:
            rep_artifacts["w_base"] = w_base
            rep_artifacts["sim_base_te"] = sim_base_te
            rep_artifacts["res_lr_base"] = res_lr_base
            rep_artifacts["diag_base"] = diag_base
            # Cache features
            np.savez_compressed(
                results_dir / "baseline_features.npz",
                X_train=X_tr_base, y_train=y_train,
                X_test=X_te_base, y_test=y_test,
                w_matrix=w_base
            )

        # ---------------------------------------------------------
        # Analytic Infinite-Time LIF Rate Representation
        # ---------------------------------------------------------
        rates_tr_ana = compute_analytic_lif_rate(X_train, w_base, tau_m_ms=tau_m_baseline)
        rates_te_ana = compute_analytic_lif_rate(X_test, w_base, tau_m_ms=tau_m_baseline)
        res_ana = train_and_evaluate_readout(rates_tr_ana, y_train, rates_te_ana, y_test, model_type="logistic", use_cv=True, seed=seed)
        diag_ana = compute_response_diagnostics(rates_te_ana, X_test)
        all_metrics["analytic_lif"].append({
            "acc": res_ana["accuracy"], "f1": res_ana["macro_f1"],
            "rate": float(np.mean(rates_te_ana)), "resp_corr": diag_ana["mean_response_corr"],
            "eff_rank": diag_ana["effective_rank"], "ink_corr": diag_ana["mean_ink_corr"],
            "synapses": N_in * N_hid
        })
        print(f"Analytic LIF Rate Baseline: Acc = {res_ana['accuracy']*100:.2f}%, EffRank = {diag_ana['effective_rank']:.1f}")

        # ---------------------------------------------------------
        # Experiment 2: Weight Initialisation (Uniform U(0, 1) & Signed Control)
        # ---------------------------------------------------------
        print(f"\n--- [Seed {seed}] Weight Init Comparison (Uniform U(0, 1)) ---")
        scale_uni, w_uni = calibrate_weight_scale(
            X_val, target_rate_hz=25.0, N_hid=N_hid, tau_m_ms=tau_m_baseline,
            connectivity="dense", weight_dist="uniform", seed=seed,
            build_dir=f"build_calib_uni_s{seed}"
        )
        sim_uni_tr = simulate_snn(spk_idx_tr, spk_t_tr, len(X_train), w_matrix=w_uni, build_dir=f"build_uni_tr_s{seed}")
        sim_uni_te = simulate_snn(spk_idx_te, spk_t_te, len(X_test), w_matrix=w_uni, build_dir=f"build_uni_te_s{seed}")
        res_uni = train_and_evaluate_readout(sim_uni_tr["spike_counts"], y_train, sim_uni_te["spike_counts"], y_test, model_type="logistic", use_cv=True, seed=seed)
        diag_uni = compute_response_diagnostics(sim_uni_te["spike_counts"], X_test)

        all_metrics["uniform_logistic"].append({
            "acc": res_uni["accuracy"], "f1": res_uni["macro_f1"],
            "rate": sim_uni_te["mean_rate_hz"], "resp_corr": diag_uni["mean_response_corr"],
            "eff_rank": diag_uni["effective_rank"], "ink_corr": diag_uni["mean_ink_corr"],
            "synapses": N_in * N_hid, "scale": scale_uni
        })
        print(f"Uniform U(0,1): Acc = {res_uni['accuracy']*100:.2f}%, EffRank = {diag_uni['effective_rank']:.1f}, RespCorr = {diag_uni['mean_response_corr']:.4f}")

        if seed == seeds[0]:
            rep_artifacts["w_uni"] = w_uni
            rep_artifacts["sim_uni_te"] = sim_uni_te
            rep_artifacts["diag_uni"] = diag_uni

        # Signed-weight common-mode control: N(0, 0.25) with positive bias so mean rate ~25 Hz
        rng_signed = np.random.RandomState(seed + 500)
        w_signed_raw = rng_signed.normal(0.0, 0.25, size=(N_in, N_hid)).astype(np.float32)
        # Shift slightly so neurons are driven above threshold
        w_signed_shifted = w_signed_raw + 0.15
        sim_sgn_tr = simulate_snn(spk_idx_tr, spk_t_tr, len(X_train), w_matrix=w_signed_shifted * 0.40, build_dir=f"build_sgn_tr_s{seed}")
        sim_sgn_te = simulate_snn(spk_idx_te, spk_t_te, len(X_test), w_matrix=w_signed_shifted * 0.40, build_dir=f"build_sgn_te_s{seed}")
        res_sgn = train_and_evaluate_readout(sim_sgn_tr["spike_counts"], y_train, sim_sgn_te["spike_counts"], y_test, model_type="logistic", use_cv=True, seed=seed)
        diag_sgn = compute_response_diagnostics(sim_sgn_te["spike_counts"], X_test)
        all_metrics["signed_logistic"].append({
            "acc": res_sgn["accuracy"], "f1": res_sgn["macro_f1"],
            "rate": sim_sgn_te["mean_rate_hz"], "resp_corr": diag_sgn["mean_response_corr"],
            "eff_rank": diag_sgn["effective_rank"], "ink_corr": diag_sgn["mean_ink_corr"],
            "synapses": N_in * N_hid
        })

        # ---------------------------------------------------------
        # Experiment 3: Extra Credit 1 — tau_m Sweep (5, 10, 20, 50, 100 ms)
        # ---------------------------------------------------------
        print(f"\n--- [Seed {seed}] Extra Credit 1: Systematic tau_m Sweep ---")
        if seed == seeds[0]:
            rep_artifacts["traces_fixed"] = {}
            rep_artifacts["traces_matched"] = {}
            rep_artifacts["timing_metrics"] = {}

        for tau in tau_values:
            # Protocol A: Fixed weights (w_base)
            sim_tf_tr = simulate_snn(spk_idx_tr, spk_t_tr, len(X_train), tau_m_ms=tau, w_matrix=w_base, build_dir=f"build_tf_tr_t{int(tau)}_s{seed}")
            sim_tf_te = simulate_snn(
                spk_idx_te, spk_t_te, len(X_test), tau_m_ms=tau, w_matrix=w_base,
                build_dir=f"build_tf_te_t{int(tau)}_s{seed}",
                record_traces=(seed == seeds[0] and tau in [5.0, 20.0, 50.0]),
                return_spike_timing=True
            )
            res_tf = train_and_evaluate_readout(sim_tf_tr["spike_counts"], y_train, sim_tf_te["spike_counts"], y_test, model_type="logistic", use_cv=False, seed=seed)
            diag_tf = compute_response_diagnostics(sim_tf_te["spike_counts"], X_test)

            all_metrics[f"tau_fixed_{int(tau)}ms"].append({
                "acc": res_tf["accuracy"], "f1": res_tf["macro_f1"],
                "rate": sim_tf_te["mean_rate_hz"], "eff_rank": diag_tf["effective_rank"],
                "resp_corr": diag_tf["mean_response_corr"], "ink_corr": diag_tf["mean_ink_corr"],
                "cv_isi": sim_tf_te.get("cv_isi", np.nan),
                "first_spike": np.nanmean(sim_tf_te["first_spike_ms"]),
                "synapses": N_in * N_hid
            })

            if seed == seeds[0] and tau in [5.0, 20.0, 50.0] and "trace_t" in sim_tf_te:
                rep_artifacts["traces_fixed"][tau] = (sim_tf_te["trace_t"], sim_tf_te["trace_v"])

            # Protocol B: Rate-matched weights (calibrated per tau_m)
            scale_tm, w_tm = calibrate_weight_scale(
                X_val, target_rate_hz=25.0, N_hid=N_hid, tau_m_ms=tau,
                connectivity="dense", weight_dist="normal", seed=seed,
                build_dir=f"build_calib_tm_t{int(tau)}_s{seed}"
            )
            sim_tm_tr = simulate_snn(spk_idx_tr, spk_t_tr, len(X_train), tau_m_ms=tau, w_matrix=w_tm, build_dir=f"build_tm_tr_t{int(tau)}_s{seed}")
            sim_tm_te = simulate_snn(
                spk_idx_te, spk_t_te, len(X_test), tau_m_ms=tau, w_matrix=w_tm,
                build_dir=f"build_tm_te_t{int(tau)}_s{seed}",
                record_traces=(seed == seeds[0] and tau in [5.0, 20.0, 50.0]),
                return_spike_timing=True
            )
            res_tm = train_and_evaluate_readout(sim_tm_tr["spike_counts"], y_train, sim_tm_te["spike_counts"], y_test, model_type="logistic", use_cv=False, seed=seed)
            diag_tm = compute_response_diagnostics(sim_tm_te["spike_counts"], X_test)

            all_metrics[f"tau_matched_{int(tau)}ms"].append({
                "acc": res_tm["accuracy"], "f1": res_tm["macro_f1"],
                "rate": sim_tm_te["mean_rate_hz"], "eff_rank": diag_tm["effective_rank"],
                "resp_corr": diag_tm["mean_response_corr"], "ink_corr": diag_tm["mean_ink_corr"],
                "cv_isi": sim_tm_te.get("cv_isi", np.nan),
                "first_spike": np.nanmean(sim_tm_te["first_spike_ms"]),
                "scale": scale_tm,
                "synapses": N_in * N_hid
            })

            print(f"  tau={tau:3.0f}ms | Fixed: Acc={res_tf['accuracy']*100:5.2f}%, Rate={sim_tf_te['mean_rate_hz']:5.2f}Hz | Matched: Acc={res_tm['accuracy']*100:5.2f}%, Rate={sim_tm_te['mean_rate_hz']:5.2f}Hz (scale={scale_tm:.3f})")

            if seed == seeds[0]:
                if tau in [5.0, 20.0, 50.0] and "trace_t" in sim_tm_te:
                    rep_artifacts["traces_matched"][tau] = (sim_tm_te["trace_t"], sim_tm_te["trace_v"])
                rep_artifacts["timing_metrics"][tau] = {
                    "first_spike_ms": sim_tm_te["first_spike_ms"],
                    "isi_ms": sim_tm_te["isi_ms"],
                    "cv_isi": sim_tm_te["cv_isi"]
                }

        # ---------------------------------------------------------
        # Experiment 4: Extra Credit 2 — Sparse Connectivity (p = 0.05)
        # ---------------------------------------------------------
        print(f"\n--- [Seed {seed}] Extra Credit 2: Sparse Connectivity (p = 0.05) ---")
        scale_sp, w_sp = calibrate_weight_scale(
            X_val, target_rate_hz=25.0, N_hid=N_hid, tau_m_ms=tau_m_baseline,
            connectivity="sparse", p_conn=0.05, seed=seed,
            build_dir=f"build_calib_sp_s{seed}"
        )
        sim_sp_tr = simulate_snn(spk_idx_tr, spk_t_tr, len(X_train), w_matrix=w_sp, build_dir=f"build_sp_tr_s{seed}")
        sim_sp_te = simulate_snn(spk_idx_te, spk_t_te, len(X_test), w_matrix=w_sp, build_dir=f"build_sp_te_s{seed}")
        res_sp = train_and_evaluate_readout(sim_sp_tr["spike_counts"], y_train, sim_sp_te["spike_counts"], y_test, model_type="logistic", use_cv=True, seed=seed)
        diag_sp = compute_response_diagnostics(sim_sp_te["spike_counts"], X_test)

        synapses_sp = int((~np.isnan(w_sp)).sum())
        all_metrics["sparse_logistic"].append({
            "acc": res_sp["accuracy"], "f1": res_sp["macro_f1"],
            "rate": sim_sp_te["mean_rate_hz"], "resp_corr": diag_sp["mean_response_corr"],
            "eff_rank": diag_sp["effective_rank"], "ink_corr": diag_sp["mean_ink_corr"],
            "synapses": synapses_sp, "scale": scale_sp
        })
        print(f"Sparse p=0.05: Acc = {res_sp['accuracy']*100:.2f}%, Rate = {sim_sp_te['mean_rate_hz']:.2f} Hz, EffRank = {diag_sp['effective_rank']:.1f}, RespCorr = {diag_sp['mean_response_corr']:.4f}, Synapses = {synapses_sp:,}")

        if seed == seeds[0]:
            rep_artifacts["w_sp"] = w_sp
            rep_artifacts["sim_sp_te"] = sim_sp_te
            rep_artifacts["res_sp"] = res_sp
            rep_artifacts["diag_sp"] = diag_sp

    # -------------------------------------------------------------
    # [Step 3] Statistical Analysis across Seeds
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== SUMMARY STATISTICS ACROSS SEEDS ===")
    print("=" * 60)

    summary_rows = []

    def get_stats(metric_list, key):
        vals = [m.get(key, np.nan) for m in metric_list if not np.isnan(m.get(key, np.nan))]
        if len(vals) == 0:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals, ddof=1 if len(vals) > 1 else 0))

    configs_to_summarize = [
        ("Raw Pixels (Reference Logistic)", "raw_pixels"),
        ("Baseline SNN: Dense N(0.5, 0.1), tau_m=20ms (Logistic CV)", "baseline_logistic"),
        ("Baseline SNN: Dense N(0.5, 0.1), tau_m=20ms (Ridge CV)", "baseline_ridge"),
        ("Analytic LIF Rate Representation (tau_m=20ms, Logistic)", "analytic_lif"),
        ("Weight Init: Dense Uniform U(0, 1), tau_m=20ms", "uniform_logistic"),
        ("Weight Init: Signed Zero-Mean Control, tau_m=20ms", "signed_logistic"),
        ("Extra Credit 1: tau_m = 5 ms (Rate-Matched ~25Hz)", "tau_matched_5ms"),
        ("Extra Credit 1: tau_m = 10 ms (Rate-Matched ~25Hz)", "tau_matched_10ms"),
        ("Extra Credit 1: tau_m = 20 ms (Rate-Matched ~25Hz)", "tau_matched_20ms"),
        ("Extra Credit 1: tau_m = 50 ms (Rate-Matched ~25Hz)", "tau_matched_50ms"),
        ("Extra Credit 1: tau_m = 100 ms (Rate-Matched ~25Hz)", "tau_matched_100ms"),
        ("Extra Credit 1: tau_m = 5 ms (Fixed Weights)", "tau_fixed_5ms"),
        ("Extra Credit 1: tau_m = 10 ms (Fixed Weights)", "tau_fixed_10ms"),
        ("Extra Credit 1: tau_m = 50 ms (Fixed Weights)", "tau_fixed_50ms"),
        ("Extra Credit 1: tau_m = 100 ms (Fixed Weights)", "tau_fixed_100ms"),
        ("Extra Credit 2: Sparse Connectivity (p = 0.05, ~39 syn/neuron)", "sparse_logistic"),
    ]

    for label, group_key in configs_to_summarize:
        acc_m, acc_s = get_stats(all_metrics[group_key], "acc")
        f1_m, f1_s = get_stats(all_metrics[group_key], "f1")
        rate_m, rate_s = get_stats(all_metrics[group_key], "rate")
        eff_m, eff_s = get_stats(all_metrics[group_key], "eff_rank")
        corr_m, corr_s = get_stats(all_metrics[group_key], "resp_corr")
        syn = all_metrics[group_key][0].get("synapses", 313600) if len(all_metrics[group_key]) > 0 else 313600

        summary_rows.append({
            "Model Configuration": label,
            "Test Accuracy (%)": f"{acc_m*100:.2f} ± {acc_s*100:.2f}%" if not np.isnan(acc_s) else f"{acc_m*100:.2f}%",
            "Macro F1": f"{f1_m:.4f} ± {f1_s:.4f}",
            "Mean Hidden Rate (Hz)": f"{rate_m:.2f} ± {rate_s:.2f}" if not np.isnan(rate_m) else "N/A",
            "Effective Rank": f"{eff_m:.1f} ± {eff_s:.1f}" if not np.isnan(eff_m) else "N/A",
            "Response Corr": f"{corr_m:.4f} ± {corr_s:.4f}" if not np.isnan(corr_m) else "N/A",
            "Synapses": f"{syn:,}"
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / "q2_experiments_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved Question 2 Multi-Seed Summary Table to {summary_path}")
    print("\n" + df_summary.to_string(index=False))

    # -------------------------------------------------------------
    # [Step 4] Generate All Diagnostic Figures
    # -------------------------------------------------------------
    print("\n[Step 4] Generating Diagnostic Figures...")

    # Plot 1: Baseline Spike Raster Plot (0, 1, 8)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, dpi=300)
    target_digits = [0, 1, 8]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    spk_t_rep = rep_artifacts["sim_base_te"]["spike_times"]
    spk_i_rep = rep_artifacts["sim_base_te"]["spike_neurons"]

    for ax_idx, digit in enumerate(target_digits):
        img_idx = np.where(y_test == digit)[0][0]
        t_start = img_idx * T_enc_ms
        t_end = t_start + T_enc_ms
        mask = (spk_t_rep >= t_start) & (spk_t_rep < t_end)
        axes[ax_idx].scatter(spk_t_rep[mask] - t_start, spk_i_rep[mask], s=6, color=colors[ax_idx], alpha=0.85)
        axes[ax_idx].set_ylabel(f"Digit '{digit}'\nNeuron ID", fontsize=10)
        axes[ax_idx].set_ylim(-5, N_hid + 5)
        axes[ax_idx].grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time within Image Window (ms)", fontsize=11)
    fig.suptitle("Hidden Layer Spike Raster for Distinct Digits (0, 1, 8) [tau_m = 20ms]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_baseline_raster.png")
    plt.close()

    # Plot 1b: Baseline Membrane Potential Traces (Neurons 0 & 1)
    if 20.0 in rep_artifacts["traces_fixed"]:
        t_b, v_b = rep_artifacts["traces_fixed"][20.0]
        mask_b = t_b <= T_enc_ms
        plt.figure(figsize=(8, 3.8), dpi=300)
        plt.plot(t_b[mask_b], v_b[0, mask_b], color="#1f77b4", lw=1.5, label="Neuron 0")
        if v_b.shape[0] > 1:
            plt.plot(t_b[mask_b], v_b[1, mask_b], color="#d62728", lw=1.2, linestyle="--", label="Neuron 1")
        plt.axhline(-52.0, color="black", linestyle="--", label="Threshold (-52 mV)")
        plt.axhline(-65.0, color="gray", linestyle=":", label="Reset (-65 mV)")
        plt.xlabel("Time within Window (ms)", fontsize=11)
        plt.ylabel("Membrane Potential (mV)", fontsize=11)
        plt.title("Baseline SNN Membrane Potential Traces: Neurons 0 & 1 [tau_m = 20ms]", fontsize=12, fontweight="bold")
        plt.legend(fontsize=9, loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(results_dir / "fig_q2_baseline_voltage_trace.png")
        plt.close()

    # Plot 2: Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(rep_artifacts["res_lr_base"]["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    base_acc_mean = np.mean([m["acc"] for m in all_metrics["baseline_logistic"]])
    plt.title(f"Baseline SNN Confusion Matrix (Mean Acc = {base_acc_mean*100:.2f}%)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_baseline_confusion_matrix.png")
    plt.close()

    # Plot 3: Hidden Layer Rate Distribution
    hidden_counts = rep_artifacts["sim_base_te"]["spike_counts"]
    hidden_rates = hidden_counts.sum(axis=0) / (len(X_test) * (T_enc_ms / 1000.0))
    plt.figure(figsize=(7, 4), dpi=300)
    plt.hist(hidden_rates, bins=25, color="#2ca02c", edgecolor="black", alpha=0.8)
    plt.axvline(hidden_rates.mean(), color="red", linestyle="--", label=f"Mean = {hidden_rates.mean():.1f} Hz")
    plt.xlabel("Neuron Firing Rate (Hz)", fontsize=11)
    plt.ylabel("Number of Neurons", fontsize=11)
    plt.title("Distribution of Hidden Neuron Firing Rates [tau_m = 20ms]", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_baseline_rate_dist.png")
    plt.close()

    # Plot 4: tau_m Sweep Accuracy with Error Bars (Fixed vs Rate-Matched)
    fixed_accs = [np.mean([m["acc"]*100 for m in all_metrics[f"tau_fixed_{int(tau)}ms"] if not np.isnan(m["acc"])]) for tau in tau_values]
    fixed_errs = [np.std([m["acc"]*100 for m in all_metrics[f"tau_fixed_{int(tau)}ms"] if not np.isnan(m["acc"])], ddof=1) if len(all_metrics[f"tau_fixed_{int(tau)}ms"]) > 1 else 0.0 for tau in tau_values]
    matched_accs = [np.mean([m["acc"]*100 for m in all_metrics[f"tau_matched_{int(tau)}ms"] if not np.isnan(m["acc"])]) for tau in tau_values]
    matched_errs = [np.std([m["acc"]*100 for m in all_metrics[f"tau_matched_{int(tau)}ms"] if not np.isnan(m["acc"])], ddof=1) if len(all_metrics[f"tau_matched_{int(tau)}ms"]) > 1 else 0.0 for tau in tau_values]

    plt.figure(figsize=(7.5, 4.8), dpi=300)
    plt.errorbar(tau_values, fixed_accs, yerr=fixed_errs, fmt="o-", capsize=4, label="Protocol A: Fixed Weights (Uncalibrated Excitability)", color="#d62728", lw=2)
    plt.errorbar(tau_values, matched_accs, yerr=matched_errs, fmt="s--", capsize=4, label="Protocol B: Rate-Matched (~25 Hz Target Drive)", color="#1f77b4", lw=2)
    plt.xlabel("Membrane Time Constant tau_m (ms)", fontsize=11)
    plt.ylabel("Classification Accuracy (%)", fontsize=11)
    plt.title("Effect of Membrane Time Constant tau_m (Mean ± 1 SD across 3 Seeds)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_tau_sweep_accuracy.png")
    plt.close()

    # Plot 5: Dual Voltage Traces (Fixed Scale top row, Rate-Matched bottom row)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True, dpi=300)
    tau_sub = [5.0, 20.0, 50.0]
    tau_colors = {5.0: "#2ca02c", 20.0: "#1f77b4", 50.0: "#ff7f0e"}

    for col_idx, tau_k in enumerate(tau_sub):
        # Top: Fixed scale
        t_f, v_f = rep_artifacts["traces_fixed"][tau_k]
        mask_f = t_f <= 100.0
        axes[0, col_idx].plot(t_f[mask_f], v_f[0, mask_f], color=tau_colors[tau_k], lw=1.5)
        axes[0, col_idx].axhline(-52.0, color="black", linestyle="--", alpha=0.7)
        axes[0, col_idx].axhline(-65.0, color="gray", linestyle=":", alpha=0.7)
        axes[0, col_idx].set_title(f"Fixed: tau_m = {tau_k:.0f} ms", fontsize=11, fontweight="bold")
        axes[0, col_idx].grid(True, linestyle="--", alpha=0.5)

        # Bottom: Rate-matched scale
        t_m, v_m = rep_artifacts["traces_matched"][tau_k]
        mask_m = t_m <= 100.0
        axes[1, col_idx].plot(t_m[mask_m], v_m[0, mask_m], color=tau_colors[tau_k], lw=1.5)
        axes[1, col_idx].axhline(-52.0, color="black", linestyle="--", alpha=0.7)
        axes[1, col_idx].axhline(-65.0, color="gray", linestyle=":", alpha=0.7)
        axes[1, col_idx].set_title(f"Rate-Matched: tau_m = {tau_k:.0f} ms", fontsize=11, fontweight="bold")
        axes[1, col_idx].grid(True, linestyle="--", alpha=0.5)
        axes[1, col_idx].set_xlabel("Time (ms)", fontsize=10)

    axes[0, 0].set_ylabel("Membrane V (mV)\n[Fixed Scale s=0.18]", fontsize=10)
    axes[1, 0].set_ylabel("Membrane V (mV)\n[Rate-Matched Scale]", fontsize=10)
    fig.suptitle("LIF Membrane Potential Traces: Fixed vs Rate-Matched Scale", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_tau_voltage_traces.png")
    plt.close()

    # Plot 6: Spike Timing Distributions across tau_m (First-Spike Latency & ISI)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=300)
    # Latencies
    mean_latencies = [np.mean([m["first_spike"] for m in all_metrics[f"tau_matched_{int(t)}ms"]]) for t in tau_values]
    axes[0].plot(tau_values, mean_latencies, "o-", color="#1f77b4", lw=2)
    axes[0].set_xlabel("tau_m (ms)", fontsize=10)
    axes[0].set_ylabel("Mean First-Spike Latency (ms)", fontsize=10)
    axes[0].set_title("First-Spike Latency vs tau_m", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # CV_ISI vs tau_m
    cv_isis = [np.nanmean([m["cv_isi"] for m in all_metrics[f"tau_matched_{int(t)}ms"]]) for t in tau_values]
    axes[1].plot(tau_values, cv_isis, "s-", color="#d62728", lw=2)
    axes[1].set_xlabel("tau_m (ms)", fontsize=10)
    axes[1].set_ylabel("Coefficient of Variation of ISI", fontsize=10)
    axes[1].set_title("ISI Variability (CV_ISI) vs tau_m", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # ISI Histograms for tau=10ms and tau=50ms
    isi_10 = rep_artifacts["timing_metrics"].get(10.0, {}).get("isi_ms", np.array([]))
    isi_50 = rep_artifacts["timing_metrics"].get(50.0, {}).get("isi_ms", np.array([]))
    if len(isi_10) > 0:
        axes[2].hist(isi_10[isi_10 < 40], bins=20, alpha=0.6, label="tau_m = 10ms", density=True, color="#2ca02c")
    if len(isi_50) > 0:
        axes[2].hist(isi_50[isi_50 < 40], bins=20, alpha=0.6, label="tau_m = 50ms", density=True, color="#ff7f0e")
    axes[2].set_xlabel("Inter-Spike Interval (ms)", fontsize=10)
    axes[2].set_ylabel("Probability Density", fontsize=10)
    axes[2].set_title("ISI Distribution (Matched Drive)", fontsize=11, fontweight="bold")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Spike Timing & Temporal Statistics across tau_m Sweep", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_tau_spike_timing.png")
    plt.close()

    # Plot 7: Feature Redundancy Diagnostics (Dense Normal vs Uniform vs Sparse)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    # A: Scree Plot (Eigenvalue spectrum)
    for model_name, counts_mat, col in [
        ("Dense N(0.5,0.1)", rep_artifacts["sim_base_te"]["spike_counts"], "#d62728"),
        ("Dense U(0,1)", rep_artifacts["sim_uni_te"]["spike_counts"], "#ff7f0e"),
        ("Sparse p=0.05", rep_artifacts["sim_sp_te"]["spike_counts"], "#2ca02c")
    ]:
        Xc = counts_mat.astype(np.float64) - np.mean(counts_mat, axis=0)
        s_vals = np.linalg.svd(Xc, compute_uv=False)
        eigs = (s_vals ** 2) / max(1, len(Xc) - 1)
        axes[0].plot(np.arange(1, 51), eigs[:50] / np.sum(eigs), label=model_name, color=col, lw=2)

    axes[0].set_xlabel("Principal Component Index", fontsize=10)
    axes[0].set_ylabel("Fraction of Variance", fontsize=10)
    axes[0].set_title("Eigenvalue Spectrum (First 50 PCs)", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # B: Pairwise Response Correlation Heatmap Comparison (100x100 subset)
    c_dense = np.corrcoef(rep_artifacts["sim_base_te"]["spike_counts"][:, :80].T)
    c_sparse = np.corrcoef(rep_artifacts["sim_sp_te"]["spike_counts"][:, :80].T)
    c_diff = c_dense - c_sparse

    sns.heatmap(c_dense[:50, :50], ax=axes[1], cmap="coolwarm", vmin=-0.1, vmax=0.9, cbar=True)
    axes[1].set_title("Dense Response Correlation (50x50)\n[High Common-Mode Redundancy]", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    sns.heatmap(c_sparse[:50, :50], ax=axes[2], cmap="coolwarm", vmin=-0.1, vmax=0.9, cbar=True)
    axes[2].set_title("Sparse Response Correlation (50x50)\n[Diverse Decoupled Subspaces]", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle("Mechanistic Diagnostics of Feature Diversity & Common-Mode Redundancy", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_feature_redundancy.png")
    plt.close()

    # Plot 8: Weight Filters (Dense vs Sparse)
    fig, axes = plt.subplots(4, 4, figsize=(6.5, 6.5), dpi=300)
    w_base_rep = rep_artifacts["w_base"]
    for k in range(16):
        ax = axes[k // 4, k % 4]
        ax.imshow(w_base_rep[:, k].reshape(28, 28), cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Neuron {k}", fontsize=8)
    fig.suptitle("Initial Dense Weight Receptive Fields (N(0.5, 0.1))", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_dense_receptive_fields.png")
    plt.close()

    fig, axes = plt.subplots(4, 4, figsize=(6.5, 6.5), dpi=300)
    w_sp_rep = rep_artifacts["w_sp"]
    for k in range(16):
        ax = axes[k // 4, k % 4]
        ax.imshow(np.nan_to_num(w_sp_rep[:, k], nan=0.0).reshape(28, 28), cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Neuron {k}", fontsize=8)
    fig.suptitle("Sparse Weight Receptive Fields (p = 0.05, ~39 Synapses/Neuron)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_sparse_receptive_fields.png")
    plt.close()

    # Plot 9: Side-by-Side Back-Projected Digit Templates (Dense vs Sparse)
    # Formula: W_in @ (W_readout / scale_X).T
    clf_base = rep_artifacts["res_lr_base"]["classifier"]
    scaler_base = rep_artifacts["res_lr_base"]["scaler"]
    W_eff_base = clf_base.coef_ / (scaler_base.scale_ + 1e-8) # (10, 400)
    tmpl_base = W_eff_base @ w_base_rep.T # (10, 784)

    clf_sp = rep_artifacts["res_sp"]["classifier"]
    scaler_sp = rep_artifacts["res_sp"]["scaler"]
    W_eff_sp = clf_sp.coef_ / (scaler_sp.scale_ + 1e-8)
    tmpl_sp = W_eff_sp @ np.nan_to_num(w_sp_rep, nan=0.0).T

    fig, axes = plt.subplots(2, 10, figsize=(14, 3.4), dpi=300)
    for d in range(10):
        # Dense (top)
        axes[0, d].imshow(tmpl_base[d].reshape(28, 28), cmap="coolwarm")
        axes[0, d].axis("off")
        axes[0, d].set_title(f"Digit {d}", fontsize=9)
        # Sparse (bottom)
        axes[1, d].imshow(tmpl_sp[d].reshape(28, 28), cmap="coolwarm")
        axes[1, d].axis("off")

    axes[0, 0].text(-10, 14, "Dense\nBaseline\n(48.8%)", va="center", ha="right", fontsize=9, fontweight="bold")
    axes[1, 0].text(-10, 14, "Sparse\np=0.05\n(87.4%)", va="center", ha="right", fontsize=9, fontweight="bold")
    fig.suptitle("Scale-Corrected Back-Projected Class Readout Templates: Dense vs Sparse", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_readout_digit_templates.png")
    plt.close()

    # Plot 10: Mean-Centered Spike-Triggered Averages (STAs) of Top 16 Active Sparse Neurons
    spk_sp_counts = rep_artifacts["sim_sp_te"]["spike_counts"]  # (1500, 400)
    top_neurons = np.argsort(spk_sp_counts.sum(axis=0))[-16:][::-1]
    X_mean = np.mean(X_test, axis=0)

    fig, axes = plt.subplots(4, 4, figsize=(7.0, 7.0), dpi=300)
    for idx, neu_id in enumerate(top_neurons):
        ax = axes[idx // 4, idx % 4]
        weights_img = spk_sp_counts[:, neu_id].astype(np.float32)
        if np.sum(weights_img) > 0:
            sta_diff = np.average(X_test - X_mean, axis=0, weights=weights_img).reshape(28, 28)
            vmax = max(abs(sta_diff.min()), abs(sta_diff.max()), 1e-4)
            ax.imshow(sta_diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(np.zeros((28, 28)), cmap="coolwarm", vmin=-1, vmax=1)
        ax.axis("off")
        ax.set_title(f"N{neu_id} ({int(weights_img.sum())} spk)", fontsize=8)
    fig.suptitle("Mean-Centered Spike-Triggered Averages (STAs) of Top Sparse Neurons", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_q2_sparse_sta.png")
    plt.close()

    print("\nAll Question 2 multi-seed simulations and figures completed successfully!")


if __name__ == "__main__":
    run_all_q2_experiments(seeds=(42, 43, 44))
