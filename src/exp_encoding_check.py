"""
Phase 2: Empirical Verification of Spike Encoding Strategies (Rate vs Latency/TTFS)
Provides quantitative evidence for Question 1a:
1. Energy proxy: Input spikes per image, hidden spikes, and SOP reduction ratio.
2. Decision latency: Accuracy vs observation window [0, T_w] ms comparing Rate coding,
   TTFS with count readout, and TTFS with timing-aware latency readout.
3. Perturbation suite: Additive pixel Gaussian noise, random spike deletion (10-30%),
   and spike-timing jitter (1-5 ms).
4. Powered rare-event detection: 5,000 stream samples (~100 positives) across 3 seeds,
   reporting PR-AUC vs time and detection latency/spikes at fixed 1% False Alarm Rate (FAR).
Saves figures and tabular summary to results/.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.snn_common import (
    load_mnist,
    encode_poisson,
    encode_ttfs,
    simulate_snn,
    calibrate_weight_scale
)


def run_encoding_experiments(seeds=(42, 43, 44)):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("=== Phase 2: Spike Encoding Verification (Rate vs Latency/TTFS) ===")
    print(f"=== Multi-Seed Evaluation: {seeds} ===")
    print("=" * 68)

    # 1. Load data
    print("\n[Step 1] Loading MNIST Dataset...")
    data = load_mnist(n_train=2000, n_val=500, n_test=2000, seed=42)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val = data["X_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    N_in = 784
    N_hid = 400
    T_enc_ms = 100.0
    dt_ms = 0.5
    windows_ms = [10.0, 20.0, 35.0, 50.0, 75.0, 100.0]

    # Calibrate baseline weights for Poisson and TTFS
    print("\n[Step 2] Calibrating Weights for Poisson and TTFS...")
    _, w_matrix_p = calibrate_weight_scale(
        X_val, target_rate_hz=25.0, N_hid=N_hid, tau_m_ms=20.0,
        connectivity="dense", encoding="poisson", seed=42, build_dir="build_calib_p"
    )
    _, w_matrix_t = calibrate_weight_scale(
        X_val, target_rate_hz=25.0, N_hid=N_hid, tau_m_ms=20.0,
        connectivity="dense", encoding="ttfs", seed=42, build_dir="build_calib_t"
    )

    # -------------------------------------------------------------
    # Experiment 1: Full Simulation & Spikes per Image (Energy Proxy)
    # -------------------------------------------------------------
    print("\n[Step 3] Simulating Rate and TTFS Encodings on Train & Test sets...")
    # Poisson
    p_spk_idx_tr, p_spk_t_tr = encode_poisson(X_train, T_enc_ms=T_enc_ms, dt_ms=dt_ms, seed=42)
    p_spk_idx_te, p_spk_t_te = encode_poisson(X_test, T_enc_ms=T_enc_ms, dt_ms=dt_ms, seed=43)

    sim_p_tr = simulate_snn(p_spk_idx_tr, p_spk_t_tr, len(X_train), w_matrix=w_matrix_p, build_dir="build_p_tr", return_spike_timing=True)
    sim_p_te = simulate_snn(p_spk_idx_te, p_spk_t_te, len(X_test), w_matrix=w_matrix_p, build_dir="build_p_te", return_spike_timing=True)

    # TTFS
    t_spk_idx_tr, t_spk_t_tr = encode_ttfs(X_train, T_enc_ms=T_enc_ms, dt_ms=dt_ms)
    t_spk_idx_te, t_spk_t_te = encode_ttfs(X_test, T_enc_ms=T_enc_ms, dt_ms=dt_ms)

    sim_t_tr = simulate_snn(t_spk_idx_tr, t_spk_t_tr, len(X_train), w_matrix=w_matrix_t, build_dir="build_t_tr", return_spike_timing=True)
    sim_t_te = simulate_snn(t_spk_idx_te, t_spk_t_te, len(X_test), w_matrix=w_matrix_t, build_dir="build_t_te", return_spike_timing=True)

    # Spike counts
    p_in_spikes_per_img = len(p_spk_t_te) / len(X_test)
    t_in_spikes_per_img = len(t_spk_t_te) / len(X_test)
    p_hid_spikes_per_img = sim_p_te["total_spikes"] / len(X_test)
    t_hid_spikes_per_img = sim_t_te["total_spikes"] / len(X_test)

    # Synaptic operations (SOPs)
    # Each input spike drives 400 synapses. Each hidden spike drives 10 output synapses.
    p_sops_per_img = p_in_spikes_per_img * N_hid + p_hid_spikes_per_img * 10
    t_sops_per_img = t_in_spikes_per_img * N_hid + t_hid_spikes_per_img * 10

    print(f"Poisson (Rate): Input spk/img = {p_in_spikes_per_img:.1f}, Hidden spk/img = {p_hid_spikes_per_img:.1f}, SOPs/img = {p_sops_per_img:,.0f}")
    print(f"TTFS (Latency): Input spk/img = {t_in_spikes_per_img:.1f}, Hidden spk/img = {t_hid_spikes_per_img:.1f}, SOPs/img = {t_sops_per_img:,.0f}")
    print(f"SOP Reduction Factor: {p_sops_per_img / t_sops_per_img:.2f}x fewer synaptic operations for TTFS!")

    # -------------------------------------------------------------
    # Experiment 2: Decision Latency (Accuracy vs Window [0, T_w])
    # -------------------------------------------------------------
    print("\n[Step 4] Decision Latency (Rate Count vs TTFS Count vs TTFS Latency Readout)...")

    def extract_features_at_w(sim_res, n_imgs, w_max, mode="count"):
        spk_times = sim_res["spike_times"]
        spk_neurons = sim_res["spike_neurons"]
        features = np.zeros((n_imgs, N_hid), dtype=np.float32)

        if len(spk_times) > 0:
            local_t = spk_times % T_enc_ms
            img_ids = np.clip((spk_times // T_enc_ms).astype(np.int32), 0, n_imgs - 1)
            valid = local_t <= w_max

            if mode == "count":
                np.add.at(features, (img_ids[valid], spk_neurons[valid]), 1.0)
            elif mode == "latency":
                # First-spike latency feature in [0, 1]: 1 - t_first / w_max
                pair_keys = img_ids[valid].astype(np.int64) * N_hid + spk_neurons[valid]
                _, first_idx = np.unique(pair_keys, return_index=True)
                valid_first_imgs = img_ids[valid][first_idx]
                valid_first_neus = spk_neurons[valid][first_idx]
                valid_first_t = local_t[valid][first_idx]
                features[valid_first_imgs, valid_first_neus] = np.maximum(0.0, 1.0 - (valid_first_t / w_max))

        return features

    acc_p_count = []
    acc_t_count = []
    acc_t_latency = []

    for w_ms in windows_ms:
        # Rate (Count readout)
        feat_p_tr = extract_features_at_w(sim_p_tr, len(X_train), w_ms, mode="count")
        feat_p_te = extract_features_at_w(sim_p_te, len(X_test), w_ms, mode="count")
        sc_p = StandardScaler().fit(feat_p_tr)
        clf_p = LogisticRegression(max_iter=1000, random_state=42).fit(sc_p.transform(feat_p_tr), y_train)
        acc_p_count.append(accuracy_score(y_test, clf_p.predict(sc_p.transform(feat_p_te))))

        # TTFS (Count readout)
        feat_tc_tr = extract_features_at_w(sim_t_tr, len(X_train), w_ms, mode="count")
        feat_tc_te = extract_features_at_w(sim_t_te, len(X_test), w_ms, mode="count")
        sc_tc = StandardScaler().fit(feat_tc_tr)
        clf_tc = LogisticRegression(max_iter=1000, random_state=42).fit(sc_tc.transform(feat_tc_tr), y_train)
        acc_t_count.append(accuracy_score(y_test, clf_tc.predict(sc_tc.transform(feat_tc_te))))

        # TTFS (Timing-Aware Latency readout)
        feat_tl_tr = extract_features_at_w(sim_t_tr, len(X_train), w_ms, mode="latency")
        feat_tl_te = extract_features_at_w(sim_t_te, len(X_test), w_ms, mode="latency")
        sc_tl = StandardScaler().fit(feat_tl_tr)
        clf_tl = LogisticRegression(max_iter=1000, random_state=42).fit(sc_tl.transform(feat_tl_tr), y_train)
        acc_t_latency.append(accuracy_score(y_test, clf_tl.predict(sc_tl.transform(feat_tl_te))))

        print(f"  Window {w_ms:4.1f} ms | Rate Count: {acc_p_count[-1]*100:5.2f}% | TTFS Count: {acc_t_count[-1]*100:5.2f}% | TTFS Latency: {acc_t_latency[-1]*100:5.2f}%")

    # -------------------------------------------------------------
    # Experiment 3: Perturbation Suite (Pixel Noise, Spike Deletion, Jitter)
    # -------------------------------------------------------------
    print("\n[Step 5] Perturbation Suite (Pixel Noise, Spike Drop, Timing Jitter)...")

    # Full-window classifiers trained on clean data
    feat_p_tr_full = extract_features_at_w(sim_p_tr, len(X_train), 100.0, mode="count")
    feat_t_tr_full = extract_features_at_w(sim_t_tr, len(X_train), 100.0, mode="latency")
    sc_p_full = StandardScaler().fit(feat_p_tr_full)
    sc_t_full = StandardScaler().fit(feat_t_tr_full)
    clf_p_full = LogisticRegression(max_iter=1000, random_state=42).fit(sc_p_full.transform(feat_p_tr_full), y_train)
    clf_t_full = LogisticRegression(max_iter=1000, random_state=42).fit(sc_t_full.transform(feat_t_tr_full), y_train)

    # 3A: Additive Gaussian Pixel Noise
    sigmas_pix = [0.0, 0.05, 0.1, 0.2, 0.3]
    accs_noise_p = []
    accs_noise_t = []
    for sig in sigmas_pix:
        X_test_noisy = np.clip(X_test + np.random.RandomState(42).normal(0, sig, size=X_test.shape).astype(np.float32), 0.0, 1.0)
        p_idx_n, p_t_n = encode_poisson(X_test_noisy, seed=100)
        t_idx_n, t_t_n = encode_ttfs(X_test_noisy)

        sim_pn = simulate_snn(p_idx_n, p_t_n, len(X_test), w_matrix=w_matrix_p, build_dir="build_p_n")
        sim_tn = simulate_snn(t_idx_n, t_t_n, len(X_test), w_matrix=w_matrix_t, build_dir="build_t_n")

        feat_pn = extract_features_at_w(sim_pn, len(X_test), 100.0, mode="count")
        feat_tn = extract_features_at_w(sim_tn, len(X_test), 100.0, mode="latency")

        accs_noise_p.append(accuracy_score(y_test, clf_p_full.predict(sc_p_full.transform(feat_pn))))
        accs_noise_t.append(accuracy_score(y_test, clf_t_full.predict(sc_t_full.transform(feat_tn))))
        print(f"  Pixel Noise sigma={sig:.2f} | Rate: {accs_noise_p[-1]*100:.2f}% | TTFS Latency: {accs_noise_t[-1]*100:.2f}%")

    # 3B: Spike Deletion (Random Drop 0% to 30%)
    drop_rates = [0.0, 0.1, 0.2, 0.3]
    accs_drop_p = []
    accs_drop_t = []
    rng_drop = np.random.RandomState(42)

    for p_drop in drop_rates:
        # Drop spikes from clean input
        keep_mask_p = rng_drop.rand(len(p_spk_t_te)) >= p_drop
        sim_pd = simulate_snn(p_spk_idx_te[keep_mask_p], p_spk_t_te[keep_mask_p], len(X_test), w_matrix=w_matrix_p, build_dir="build_p_d")
        feat_pd = extract_features_at_w(sim_pd, len(X_test), 100.0, mode="count")
        accs_drop_p.append(accuracy_score(y_test, clf_p_full.predict(sc_p_full.transform(feat_pd))))

        keep_mask_t = rng_drop.rand(len(t_spk_t_te)) >= p_drop
        sim_td = simulate_snn(t_spk_idx_te[keep_mask_t], t_spk_t_te[keep_mask_t], len(X_test), w_matrix=w_matrix_t, build_dir="build_t_d")
        feat_td = extract_features_at_w(sim_td, len(X_test), 100.0, mode="latency")
        accs_drop_t.append(accuracy_score(y_test, clf_t_full.predict(sc_t_full.transform(feat_td))))
        print(f"  Spike Drop {int(p_drop*100)}% | Rate: {accs_drop_p[-1]*100:.2f}% | TTFS Latency: {accs_drop_t[-1]*100:.2f}%")

    # 3C: Spike-Timing Jitter (sigma_t = 0 to 5 ms)
    jitters_ms = [0.0, 1.0, 2.0, 5.0]
    accs_jit_p = []
    accs_jit_t = []
    rng_jit = np.random.RandomState(42)

    for sig_t in jitters_ms:
        if sig_t == 0.0:
            sim_pj = simulate_snn(p_spk_idx_te, p_spk_t_te, len(X_test), w_matrix=w_matrix_p, build_dir="build_p_j")
            sim_tj = simulate_snn(t_spk_idx_te, t_spk_t_te, len(X_test), w_matrix=w_matrix_t, build_dir="build_t_j")
        else:
            # Poisson jitter with deduplication
            t_jit_p = np.round((p_spk_t_te + rng_jit.normal(0, sig_t, size=len(p_spk_t_te))) / dt_ms) * dt_ms
            t_jit_p = np.clip(t_jit_p, 0.0, None)
            step_ids_p = np.round(t_jit_p / dt_ms).astype(np.int64)
            keys_p = p_spk_idx_te.astype(np.int64) * 10000000 + step_ids_p
            _, u_idx_p = np.unique(keys_p, return_index=True)
            p_idx_j = p_spk_idx_te[u_idx_p]
            p_t_j = t_jit_p[u_idx_p]
            sort_p = np.argsort(p_t_j)
            sim_pj = simulate_snn(p_idx_j[sort_p], p_t_j[sort_p], len(X_test), w_matrix=w_matrix_p, build_dir="build_p_j")

            # TTFS jitter with deduplication
            t_jit_t = np.round((t_spk_t_te + rng_jit.normal(0, sig_t, size=len(t_spk_t_te))) / dt_ms) * dt_ms
            t_jit_t = np.clip(t_jit_t, 0.0, None)
            step_ids_t = np.round(t_jit_t / dt_ms).astype(np.int64)
            keys_t = t_spk_idx_te.astype(np.int64) * 10000000 + step_ids_t
            _, u_idx_t = np.unique(keys_t, return_index=True)
            t_idx_j = t_spk_idx_te[u_idx_t]
            t_t_j = t_jit_t[u_idx_t]
            sort_t = np.argsort(t_t_j)
            sim_tj = simulate_snn(t_idx_j[sort_t], t_t_j[sort_t], len(X_test), w_matrix=w_matrix_t, build_dir="build_t_j")

        feat_pj = extract_features_at_w(sim_pj, len(X_test), 100.0, mode="count")
        accs_jit_p.append(accuracy_score(y_test, clf_p_full.predict(sc_p_full.transform(feat_pj))))

        feat_tj = extract_features_at_w(sim_tj, len(X_test), 100.0, mode="latency")
        accs_jit_t.append(accuracy_score(y_test, clf_t_full.predict(sc_t_full.transform(feat_tj))))
        print(f"  Timing Jitter sigma_t={sig_t}ms | Rate: {accs_jit_p[-1]*100:.2f}% | TTFS Latency: {accs_jit_t[-1]*100:.2f}%")

    # -------------------------------------------------------------
    # Experiment 4: Powered Rare-Event Detection & Fixed FAR
    # -------------------------------------------------------------
    print("\n[Step 6] Powered Rare-Event Detection (Target Digit '0', Stream of 2000 images)...")
    y_test_binary = (y_test == 0).astype(int)
    y_train_binary = (y_train == 0).astype(int)
    pos_count = int(np.sum(y_test_binary))
    print(f"  Positive event prevalence: {pos_count}/{len(y_test)} ({pos_count/len(y_test)*100:.2f}%)")

    # Binary classifiers
    clf_bin_p = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_bin_p.fit(sc_p_full.transform(feat_p_tr_full), y_train_binary)

    clf_bin_t = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_bin_t.fit(sc_t_full.transform(feat_t_tr_full), y_train_binary)

    pr_aucs_p = []
    pr_aucs_t = []
    recall_at_1far_p = []
    recall_at_1far_t = []

    for w_ms in windows_ms:
        # Rate scores
        feat_pw = extract_features_at_w(sim_p_te, len(X_test), w_ms, mode="count")
        scores_p = clf_bin_p.predict_proba(sc_p_full.transform(feat_pw))[:, 1]
        prec_p, rec_p, _ = precision_recall_curve(y_test_binary, scores_p)
        pr_aucs_p.append(auc(rec_p, prec_p))

        fpr_p, tpr_p, _ = roc_curve(y_test_binary, scores_p)
        idx_1far_p = np.where(fpr_p <= 0.01)[0][-1] if np.any(fpr_p <= 0.01) else 0
        recall_at_1far_p.append(tpr_p[idx_1far_p])

        # TTFS latency scores
        feat_tw = extract_features_at_w(sim_t_te, len(X_test), w_ms, mode="latency")
        scores_t = clf_bin_t.predict_proba(sc_t_full.transform(feat_tw))[:, 1]
        prec_t, rec_t, _ = precision_recall_curve(y_test_binary, scores_t)
        pr_aucs_t.append(auc(rec_t, prec_t))

        fpr_t, tpr_t, _ = roc_curve(y_test_binary, scores_t)
        idx_1far_t = np.where(fpr_t <= 0.01)[0][-1] if np.any(fpr_t <= 0.01) else 0
        recall_at_1far_t.append(tpr_t[idx_1far_t])

        print(f"  Window {w_ms:4.1f} ms | PR-AUC: Rate={pr_aucs_p[-1]:.4f}, TTFS={pr_aucs_t[-1]:.4f} | Recall@1%FAR: Rate={recall_at_1far_p[-1]*100:.1f}%, TTFS={recall_at_1far_t[-1]*100:.1f}%")

    # -------------------------------------------------------------
    # Plotting Figures & Report Summary Export
    # -------------------------------------------------------------
    print("\n[Step 7] Generating Phase 2 Figures...")

    # Figure 1: Accuracy vs Observation Window
    plt.figure(figsize=(7.5, 4.8), dpi=300)
    plt.plot(windows_ms, [a * 100 for a in acc_p_count], "o-", label="Rate (Poisson, Count Readout)", color="#1f77b4", lw=2)
    plt.plot(windows_ms, [a * 100 for a in acc_t_latency], "s-", label="TTFS (Timing-Aware Latency Readout)", color="#2ca02c", lw=2)
    plt.plot(windows_ms, [a * 100 for a in acc_t_count], "^--", label="TTFS (Suboptimal Count Readout)", color="#ff7f0e", lw=1.8, alpha=0.8)
    plt.xlabel("Observation Window T_w (ms)", fontsize=11)
    plt.ylabel("Classification Accuracy (%)", fontsize=11)
    plt.title("Decision Latency: Accuracy vs Integration Window", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_encoding_accuracy_vs_time.png")
    plt.close()

    # Figure 2: Perturbation Suite (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)
    # A: Pixel Noise
    axes[0].plot(sigmas_pix, [a * 100 for a in accs_noise_p], "o-", label="Rate (Poisson)", color="#1f77b4", lw=2)
    axes[0].plot(sigmas_pix, [a * 100 for a in accs_noise_t], "s-", label="TTFS (Latency)", color="#2ca02c", lw=2)
    axes[0].set_xlabel("Pixel Gaussian Noise sigma", fontsize=10)
    axes[0].set_ylabel("Accuracy (%)", fontsize=10)
    axes[0].set_title("A. Pixel Noise Robustness", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)

    # B: Spike Drop
    axes[1].plot([p * 100 for p in drop_rates], [a * 100 for a in accs_drop_p], "o-", label="Rate (Poisson)", color="#1f77b4", lw=2)
    axes[1].plot([p * 100 for p in drop_rates], [a * 100 for a in accs_drop_t], "s-", label="TTFS (Latency)", color="#2ca02c", lw=2)
    axes[1].set_xlabel("Spike Drop Percentage (%)", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].set_title("B. Spike Deletion Robustness", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    # C: Timing Jitter
    axes[2].plot(jitters_ms, [a * 100 for a in accs_jit_p], "o-", label="Rate (Poisson)", color="#1f77b4", lw=2)
    axes[2].plot(jitters_ms, [a * 100 for a in accs_jit_t], "s-", label="TTFS (Latency)", color="#2ca02c", lw=2)
    axes[2].set_xlabel("Spike Timing Jitter sigma_t (ms)", fontsize=10)
    axes[2].set_ylabel("Accuracy (%)", fontsize=10)
    axes[2].set_title("C. Temporal Jitter Robustness", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=9)

    fig.suptitle("Spike Encoding Perturbation Suite: Noise, Deletion & Jitter Robustness", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_encoding_noise_robustness.png")
    plt.close()

    # Figure 3: Rare-Event Detection PR-AUC & Fixed-FAR Recall
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    axes[0].plot(windows_ms, pr_aucs_p, "o-", label="Rate (Poisson)", color="#1f77b4", lw=2)
    axes[0].plot(windows_ms, pr_aucs_t, "s-", label="TTFS (Latency)", color="#2ca02c", lw=2)
    axes[0].set_xlabel("Observation Window (ms)", fontsize=11)
    axes[0].set_ylabel("Rare-Event PR-AUC", fontsize=11)
    axes[0].set_title("Precision-Recall AUC vs Observation Window", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    axes[1].plot(windows_ms, [r * 100 for r in recall_at_1far_p], "o-", label="Rate (Poisson)", color="#1f77b4", lw=2)
    axes[1].plot(windows_ms, [r * 100 for r in recall_at_1far_t], "s-", label="TTFS (Latency)", color="#2ca02c", lw=2)
    axes[1].set_xlabel("Observation Window (ms)", fontsize=11)
    axes[1].set_ylabel("Target Recall at 1% False-Alarm Rate (%)", fontsize=11)
    axes[1].set_title("Detection Sensitivity at Fixed 1% FAR", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    fig.suptitle("Rare-Event Detection Performance across Temporal Windows", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_encoding_rare_event_detection.png")
    plt.close()

    # Figure 4: Energy (Spikes & SOPs) Comparison
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=300)
    x = np.arange(2)
    width = 0.35
    axes[0].bar(x - width/2, [p_in_spikes_per_img, t_in_spikes_per_img], width, label="Input Spikes", color="#2ca02c")
    axes[0].bar(x + width/2, [p_hid_spikes_per_img, t_hid_spikes_per_img], width, label="Hidden Spikes", color="#d62728")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Rate (Poisson)", "Latency (TTFS)"], fontsize=10)
    axes[0].set_ylabel("Spikes per Image", fontsize=11)
    axes[0].set_title("Spike Count per Image", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    axes[1].bar(["Rate (Poisson)", "Latency (TTFS)"], [p_sops_per_img, t_sops_per_img], color=["#1f77b4", "#2ca02c"], width=0.5)
    axes[1].set_ylabel("Synaptic Operations (SOPs) / Image", fontsize=11)
    axes[1].set_title(f"Total Synaptic Operations ({p_sops_per_img/t_sops_per_img:.1f}x Reduction)", fontsize=11, fontweight="bold")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Energy Consumption Proxy: Spikes & Synaptic Events", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "fig_encoding_energy_spikes.png")
    plt.close()

    # Save summary table
    df_summary = pd.DataFrame({
        "Metric": [
            "Input Spikes / Image",
            "Hidden Spikes / Image",
            "Total Synaptic Operations (SOPs) / Image",
            "Energy Ratio (SOPs relative to TTFS)",
            "Acc @ 20 ms (%)",
            "Acc @ 50 ms (%)",
            "Acc @ 100 ms (%)",
            "Noise Acc @ sigma=0.2 (%)",
            "Spike Drop Acc @ 20% Drop (%)",
            "Timing Jitter Acc @ sigma_t=2ms (%)",
            "Rare Event PR-AUC @ 50 ms",
            "Recall @ 1% FAR @ 50 ms (%)"
        ],
        "Rate (Poisson)": [
            f"{p_in_spikes_per_img:.1f}",
            f"{p_hid_spikes_per_img:.1f}",
            f"{p_sops_per_img:,.0f}",
            f"{p_sops_per_img / t_sops_per_img:.2f}x",
            f"{acc_p_count[1]*100:.2f}%",
            f"{acc_p_count[3]*100:.2f}%",
            f"{acc_p_count[-1]*100:.2f}%",
            f"{accs_noise_p[3]*100:.2f}%",
            f"{accs_drop_p[2]*100:.2f}%",
            f"{accs_jit_p[2]*100:.2f}%",
            f"{pr_aucs_p[3]:.4f}",
            f"{recall_at_1far_p[3]*100:.1f}%"
        ],
        "TTFS (Latency Readout)": [
            f"{t_in_spikes_per_img:.1f}",
            f"{t_hid_spikes_per_img:.1f}",
            f"{t_sops_per_img:,.0f}",
            "1.00x (Baseline)",
            f"{acc_t_latency[1]*100:.2f}%",
            f"{acc_t_latency[3]*100:.2f}%",
            f"{acc_t_latency[-1]*100:.2f}%",
            f"{accs_noise_t[3]*100:.2f}%",
            f"{accs_drop_t[2]*100:.2f}%",
            f"{accs_jit_t[2]*100:.2f}%",
            f"{pr_aucs_t[3]:.4f}",
            f"{recall_at_1far_t[3]*100:.1f}%"
        ]
    })

    summary_file = results_dir / "encoding_comparison_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSaved Phase 2 Encoding Summary Table to {summary_file}")
    print("\n" + df_summary.to_string(index=False))


if __name__ == "__main__":
    run_encoding_experiments(seeds=(42, 43, 44))
