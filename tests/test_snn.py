"""
Unit tests for SNN encoders, model simulation, refractoriness, and readout.
Run via: uv run pytest tests/test_snn.py
"""

import numpy as np
import pytest
from src.snn_common import (
    load_mnist,
    encode_poisson,
    encode_ttfs,
    simulate_snn,
    calibrate_weight_scale,
    train_and_evaluate_readout,
    compute_response_diagnostics,
    compute_analytic_lif_rate
)


def test_mnist_loader():
    splits = load_mnist(n_train=100, n_val=20, n_test=20, seed=42)
    assert splits["X_train"].shape == (100, 784)
    assert splits["y_train"].shape == (100,)
    assert splits["X_val"].shape == (20, 784)
    assert splits["X_test"].shape == (20, 784)
    # Normalized [0, 1]
    assert splits["X_train"].min() >= 0.0
    assert splits["X_train"].max() <= 1.0


def test_poisson_encoder():
    # Constant image with known rate
    n_images = 10
    T_enc_ms = 100.0
    r_max_hz = 100.0
    images = np.full((n_images, 784), 0.5, dtype=np.float32)
    
    indices, times = encode_poisson(images, T_enc_ms=T_enc_ms, r_max_hz=r_max_hz, seed=42)
    assert len(indices) == len(times)
    assert np.all(indices >= 0) and np.all(indices < 784)
    # Check monotonicity of spike times
    assert np.all(np.diff(times) >= 0.0)

    # Theoretical expectation: rate = 0.5 * 100 = 50 Hz.
    # Total time = 10 * 0.1 s = 1.0 s.
    # Expected spikes per neuron = 50. Total expected spikes = 50 * 784 = 39,200.
    empirical_spikes = len(times)
    expected_spikes = 50.0 * 784
    rel_error = abs(empirical_spikes - expected_spikes) / expected_spikes
    assert rel_error < 0.05, f"Poisson rate relative error too large: {rel_error:.3f}"


def test_ttfs_encoder():
    # 2 pixels: 0.8 and 0.2
    images = np.array([[0.8, 0.2]], dtype=np.float32)
    indices, times = encode_ttfs(images, T_enc_ms=100.0, dt_ms=0.5, threshold_pixel=0.1)
    
    assert len(indices) == 2
    # Pixel 0 (brighter) must spike earlier than Pixel 1
    idx_0 = np.where(indices == 0)[0][0]
    idx_1 = np.where(indices == 1)[0][0]
    assert times[idx_0] < times[idx_1]


def test_refractoriness_and_simulation():
    # Run 5 images on a 10-neuron hidden layer
    n_images = 5
    images = np.random.uniform(0.2, 0.8, size=(n_images, 784)).astype(np.float32)
    spk_idx, spk_t = encode_poisson(images, T_enc_ms=100.0, dt_ms=0.5, seed=42)

    # Fixed weights to ensure firing
    w_mat = np.full((784, 10), 0.04, dtype=np.float32)
    t_ref_ms = 5.0

    res = simulate_snn(
        spk_idx, spk_t, n_images=n_images,
        T_enc_ms=100.0, dt_ms=0.5,
        N_in=784, N_hid=10,
        tau_m_ms=20.0, t_ref_ms=t_ref_ms,
        w_matrix=w_mat,
        backend="cpp_standalone",
        build_dir="build_test_ref"
    )

    counts = res["spike_counts"]
    assert counts.shape == (n_images, 10)
    assert counts.sum() > 0

    # Verify refractoriness: for any single neuron, inter-spike intervals must be >= t_ref_ms
    spk_times = res["spike_times"]
    spk_neurons = res["spike_neurons"]
    for neuron_id in range(10):
        t_n = spk_times[spk_neurons == neuron_id]
        if len(t_n) > 1:
            isis = np.diff(t_n)
            # Allow tiny float precision margin
            assert np.all(isis >= (t_ref_ms - 1e-4)), f"Refractory violation found for neuron {neuron_id}: min ISI = {isis.min()} ms"


def test_readout_training():
    # Synthetic spike counts and labels
    rng = np.random.RandomState(42)
    X_train = rng.poisson(lam=5.0, size=(100, 50))
    y_train = rng.randint(0, 10, size=100)
    X_test = rng.poisson(lam=5.0, size=(20, 50))
    y_test = rng.randint(0, 10, size=20)

    res = train_and_evaluate_readout(X_train, y_train, X_test, y_test, model_type="logistic", seed=42)
    assert "accuracy" in res
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["confusion_matrix"].shape == (10, 10)


def test_zero_input_silence():
    # 5 all-zero images must emit 0 spikes and have NaN/0 latency
    images = np.zeros((5, 784), dtype=np.float32)
    spk_idx, spk_t = encode_poisson(images, T_enc_ms=100.0, dt_ms=0.5, seed=42)
    assert len(spk_idx) == 0

    res = simulate_snn(
        spk_idx, spk_t, n_images=5,
        T_enc_ms=100.0, dt_ms=0.5,
        N_in=784, N_hid=10,
        tau_m_ms=20.0,
        build_dir="build_test_zero",
        return_spike_timing=True
    )
    assert res["total_spikes"] == 0
    assert np.all(res["spike_counts"] == 0)
    assert np.all(np.isnan(res["first_spike_ms"]))


def test_chunk_size_invariance():
    # Verify that simulating 4 images in two 2-image segments vs all 4 at once gives identical counts
    images = np.random.uniform(0.1, 0.7, size=(4, 784)).astype(np.float32)
    spk_idx, spk_t = encode_poisson(images, T_enc_ms=100.0, dt_ms=0.5, seed=42)
    w_mat = np.full((784, 10), 0.05, dtype=np.float32)

    # All 4 images
    res_all = simulate_snn(
        spk_idx, spk_t, n_images=4,
        N_in=784, N_hid=10,
        w_matrix=w_mat,
        build_dir="build_test_chunk_all",
        return_spike_timing=True
    )

    # First 2 images
    mask_2 = spk_t < 200.0
    res_part1 = simulate_snn(
        spk_idx[mask_2], spk_t[mask_2], n_images=2,
        N_in=784, N_hid=10,
        w_matrix=w_mat,
        build_dir="build_test_chunk_p1",
        return_spike_timing=True
    )

    np.testing.assert_array_equal(res_all["spike_counts"][:2], res_part1["spike_counts"])


def test_calibration_10ms_brackets_target():
    # Verify that calibrate_weight_scale with tau_m=10ms achieves 20-30 Hz
    # Directly prevents regression of Finding F1
    splits = load_mnist(n_train=100, n_val=50, n_test=50, seed=42)
    scale, calib_w = calibrate_weight_scale(
        splits["X_val"][:50],
        target_rate_hz=25.0,
        N_hid=50,
        tau_m_ms=10.0,
        connectivity="dense",
        seed=42,
        build_dir="build_test_calib_10ms"
    )
    assert scale > 0.16, f"Scale {scale} too low for tau_m=10ms; must exceed old broken grid maximum (0.16)"

    # Verify on validation set
    spk_idx, spk_t = encode_poisson(splits["X_val"][:50], seed=42)
    sim = simulate_snn(
        spk_idx, spk_t, n_images=50,
        N_in=784, N_hid=50,
        tau_m_ms=10.0,
        w_matrix=calib_w,
        build_dir="build_test_calib_verify"
    )
    rate = sim["mean_rate_hz"]
    assert 18.0 <= rate <= 32.0, f"Calibrated rate {rate:.2f} Hz not in expected [18, 32] Hz target window"


def test_response_diagnostics():
    rng = np.random.RandomState(42)
    # 50 images, 20 neurons
    counts = rng.poisson(lam=4.0, size=(50, 20))
    images = rng.uniform(0.0, 1.0, size=(50, 784))
    diag = compute_response_diagnostics(counts, images)

    assert "mean_response_corr" in diag
    assert "effective_rank" in diag
    assert "mean_ink_corr" in diag
    assert 1.0 <= diag["effective_rank"] <= 20.0
    assert -1.0 <= diag["mean_response_corr"] <= 1.0


def test_analytic_lif_rate():
    images = np.ones((5, 784), dtype=np.float32)
    w = np.full((784, 10), 0.05, dtype=np.float32)
    rates = compute_analytic_lif_rate(images, w, tau_m_ms=20.0)
    assert rates.shape == (5, 10)
    assert np.all(rates > 0.0)
    # Higher input intensity -> higher rate
    rates_low = compute_analytic_lif_rate(images * 0.5, w, tau_m_ms=20.0)
    assert np.all(rates >= rates_low)

