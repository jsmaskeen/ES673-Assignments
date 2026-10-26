"""
Core module for Spiking Neural Network (SNN) simulation and analysis in Brian2.
Provides data loaders, spike encoders (Poisson and TTFS), model builder,
simulation harness using cpp_standalone/cython, feature extraction, and linear readouts.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV, RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

import brian2 as b2
from brian2 import ms, mV, Hz


def load_mnist(
    data_path: Union[str, Path] = "data/mnist.npz",
    n_train: int = 10000,
    n_val: int = 2000,
    n_test: int = 2000,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Load MNIST dataset, with train, validation, and test splits.
    Returns normalized pixel arrays in [0, 1] and labels.
    """
    path = Path(data_path)
    if not path.exists():
        from src.fetch_mnist import prepare_mnist
        prepare_mnist(path.parent)

    data = np.load(path)
    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_test_full = data["X_test"]
    y_test_full = data["y_test"]

    rng = np.random.RandomState(seed)
    
    # Stratified or random permutation for train/val
    train_indices = rng.permutation(len(X_train_full))
    X_train = X_train_full[train_indices[:n_train]]
    y_train = y_train_full[train_indices[:n_train]]
    
    X_val = X_train_full[train_indices[n_train:n_train + n_val]]
    y_val = y_train_full[train_indices[n_train:n_train + n_val]]

    test_indices = rng.permutation(len(X_test_full))
    X_test = X_test_full[test_indices[:n_test]]
    y_test = y_test_full[test_indices[:n_test]]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }


def encode_poisson(
    images: np.ndarray,
    T_enc_ms: float = 100.0,
    dt_ms: float = 0.5,
    r_max_hz: float = 100.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Poisson spike encoding for a batch of images.
    Returns:
        spike_indices: int array of input neuron IDs
        spike_times_ms: float array of spike times in ms
    """
    rng = np.random.RandomState(seed)
    n_images, N_in = images.shape
    steps_per_image = int(round(T_enc_ms / dt_ms))

    # Probability of spike per time bin
    p_spike = np.clip(images * (r_max_hz * (dt_ms / 1000.0)), 0.0, 1.0)
    
    # Generate random uniform values in chunks to keep memory bounded
    chunk_size = 500
    all_indices = []
    all_times = []

    for c_start in range(0, n_images, chunk_size):
        c_end = min(c_start + chunk_size, n_images)
        c_len = c_end - c_start
        
        rand_vals = rng.rand(c_len, steps_per_image, N_in).astype(np.float32)
        spike_mask = rand_vals < p_spike[c_start:c_end, None, :]
        
        c_idx, step_idx, neuron_idx = np.where(spike_mask)
        img_idx = c_idx + c_start
        times = img_idx * T_enc_ms + step_idx * dt_ms
        
        all_indices.append(neuron_idx)
        all_times.append(times)

    if len(all_indices) > 0:
        spike_indices = np.concatenate(all_indices).astype(np.int32)
        spike_times_ms = np.concatenate(all_times).astype(np.float64)
    else:
        spike_indices = np.empty(0, dtype=np.int32)
        spike_times_ms = np.empty(0, dtype=np.float64)

    # Sort spikes by time (required by SpikeGeneratorGroup)
    sort_order = np.argsort(spike_times_ms)
    return spike_indices[sort_order], spike_times_ms[sort_order]


def encode_ttfs(
    images: np.ndarray,
    T_enc_ms: float = 100.0,
    dt_ms: float = 0.5,
    threshold_pixel: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-To-First-Spike (TTFS) latency encoding for static camera frames.
    Brighter pixels fire earlier: t_spike = T_enc * (1 - pixel_value).
    Pixels below threshold_pixel do not spike.
    """
    n_images, N_in = images.shape
    all_indices = []
    all_times = []

    for i in range(n_images):
        img = images[i]
        valid_mask = img > threshold_pixel
        neuron_ids = np.where(valid_mask)[0]
        
        # Latency in [0, T_enc_ms)
        t_local = T_enc_ms * (1.0 - img[valid_mask])
        # Quantize to dt
        t_local = np.round(t_local / dt_ms) * dt_ms
        t_abs = i * T_enc_ms + t_local

        all_indices.append(neuron_ids)
        all_times.append(t_abs)

    if len(all_indices) > 0:
        spike_indices = np.concatenate(all_indices).astype(np.int32)
        spike_times_ms = np.concatenate(all_times).astype(np.float64)
    else:
        spike_indices = np.empty(0, dtype=np.int32)
        spike_times_ms = np.empty(0, dtype=np.float64)

    sort_order = np.argsort(spike_times_ms)
    return spike_indices[sort_order], spike_times_ms[sort_order]


def simulate_snn(
    spike_indices: np.ndarray,
    spike_times_ms: np.ndarray,
    n_images: int,
    T_enc_ms: float = 100.0,
    dt_ms: float = 0.5,
    N_in: int = 784,
    N_hid: int = 400,
    tau_m_ms: float = 20.0,
    v_rest_mv: float = -65.0,
    v_reset_mv: float = -65.0,
    v_th_mv: float = -52.0,
    t_ref_ms: float = 5.0,
    w_matrix: Optional[np.ndarray] = None,
    backend: str = "cpp_standalone",
    build_dir: str = "build_sim",
    record_traces: bool = False,
    return_spike_timing: bool = False,
    openmp_threads: int = 0
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Run Brian2 SNN simulation over n_images and extract hidden spike counts per image.
    Uses cpp_standalone for fast C++ execution with reproducible state resets.
    """
    b2.device.reinit()
    b2.device.activate(build_on_run=False)

    build_path = Path(build_dir)
    if build_path.exists():
        shutil.rmtree(build_path, ignore_errors=True)

    if backend == "cpp_standalone":
        b2.set_device("cpp_standalone", directory=str(build_path), build_on_run=False)
        b2.prefs.devices.cpp_standalone.openmp_threads = openmp_threads
    else:
        b2.prefs.codegen.target = "cython"

    dt = dt_ms * ms
    b2.defaultclock.dt = dt

    # Units
    T_enc = T_enc_ms * ms
    tau_m = tau_m_ms * ms
    v_rest = v_rest_mv * mV
    v_reset = v_reset_mv * mV
    v_th = v_th_mv * mV
    t_ref = t_ref_ms * ms

    # Input group
    P = b2.SpikeGeneratorGroup(N_in, spike_indices, spike_times_ms * ms)

    # Hidden group
    eqs = """
    dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
    """
    G = b2.NeuronGroup(
        N_hid, eqs,
        threshold="v > v_th", reset="v = v_reset",
        refractory=t_ref, method="exact"
    )
    G.v = v_rest

    # Reset membrane potential at start of each image
    reset_op = G.run_regularly("v = v_rest", dt=T_enc, order=-1)

    # Synapses
    S = b2.Synapses(P, G, "w : volt", on_pre="v_post += w")
    
    if w_matrix is not None:
        # Explicit weight matrix (N_in, N_hid)
        i_idx, j_idx = np.where(~np.isnan(w_matrix))
        S.connect(i=i_idx, j=j_idx)
        S.w = w_matrix[i_idx, j_idx] * mV
    else:
        # Default dense connection with 0.05 mV
        S.connect(p=1.0)
        S.w = 0.05 * mV

    # Spike monitor
    M_spikes = b2.SpikeMonitor(G)
    
    monitors = [M_spikes]
    M_state = None
    if record_traces:
        M_state = b2.StateMonitor(G, "v", record=range(min(5, N_hid)))
        monitors.append(M_state)

    net = b2.Network(P, G, S, reset_op, *monitors)
    net.run(n_images * T_enc)

    t_start = time.perf_counter()
    if backend == "cpp_standalone":
        b2.device.build(directory=str(build_path), clean=True)
    wall_clock = time.perf_counter() - t_start

    # Extract spike counts per image: matrix of shape (n_images, N_hid)
    spike_i = np.array(M_spikes.i)
    spike_t = np.array(M_spikes.t / ms)

    spike_counts = np.zeros((n_images, N_hid), dtype=np.int32)
    if len(spike_i) > 0:
        image_ids = np.clip((spike_t // T_enc_ms).astype(np.int32), 0, n_images - 1)
        np.add.at(spike_counts, (image_ids, spike_i), 1)

    results: Dict[str, Any] = {
        "spike_counts": spike_counts,
        "total_spikes": len(spike_i),
        "mean_rate_hz": float(len(spike_i)) / (n_images * (T_enc_ms / 1000.0) * N_hid),
        "wall_clock_s": wall_clock,
        "spike_times": spike_t,
        "spike_neurons": spike_i
    }

    if return_spike_timing:
        # First spike latency relative to image onset (in ms)
        first_spike_ms = np.full((n_images, N_hid), np.nan, dtype=np.float32)
        all_isis: List[float] = []

        if len(spike_i) > 0:
            image_ids = np.clip((spike_t // T_enc_ms).astype(np.int32), 0, n_images - 1)
            rel_t = spike_t - image_ids * T_enc_ms
            
            # First spike latency: first occurrence of (image_id, neuron_id) since spike_t is sorted
            pair_keys = image_ids.astype(np.int64) * N_hid + spike_i
            _, first_idx = np.unique(pair_keys, return_index=True)
            first_spike_ms[image_ids[first_idx], spike_i[first_idx]] = rel_t[first_idx]

            # Inter-spike intervals within same image window and neuron
            sort_keys = np.lexsort((rel_t, spike_i, image_ids))
            sorted_img = image_ids[sort_keys]
            sorted_neu = spike_i[sort_keys]
            sorted_t = rel_t[sort_keys]
            same_burst = (sorted_img[:-1] == sorted_img[1:]) & (sorted_neu[:-1] == sorted_neu[1:])
            if np.any(same_burst):
                all_isis = list(sorted_t[1:][same_burst] - sorted_t[:-1][same_burst])

        results["first_spike_ms"] = first_spike_ms
        results["isi_ms"] = np.array(all_isis, dtype=np.float32)
        if len(all_isis) > 1 and np.mean(all_isis) > 0:
            results["cv_isi"] = float(np.std(all_isis) / np.mean(all_isis))
        else:
            results["cv_isi"] = float("nan")

    if record_traces and M_state is not None:
        results["trace_t"] = np.array(M_state.t / ms)
        results["trace_v"] = np.array(M_state.v / mV)

    # Clean up build directory to save disk space
    if backend == "cpp_standalone" and build_path.exists():
        shutil.rmtree(build_path, ignore_errors=True)

    return results


def calibrate_weight_scale(
    images_val: np.ndarray,
    target_rate_hz: float = 25.0,
    N_hid: int = 400,
    tau_m_ms: float = 20.0,
    connectivity: str = "dense",
    p_conn: float = 1.0,
    weight_dist: str = "normal",
    encoding: str = "poisson",
    seed: int = 42,
    backend: str = "cpp_standalone",
    build_dir: str = "build_calib"
) -> Tuple[float, np.ndarray]:
    """
    Search for the optimal weight scale multiplier so the mean hidden firing
    rate matches target_rate_hz (~20-30 Hz) on validation images.
    Uses mean-field centering and a single-build parallel subpopulation sweep
    to ensure rapid, non-edge-clipping calibration for all tau_m regimes.
    """
    n_sample = min(200, len(images_val))
    val_subset = images_val[:n_sample]
    
    # Generate reference unscaled weights
    rng = np.random.RandomState(seed)
    N_in = 784
    
    if connectivity == "dense":
        if weight_dist == "normal":
            raw_w = np.clip(rng.normal(0.5, 0.1, size=(N_in, N_hid)), 0.0, None)
        else: # uniform
            raw_w = rng.uniform(0.0, 1.0, size=(N_in, N_hid))
        mean_w0 = float(np.mean(raw_w))
    else: # sparse
        mask = rng.rand(N_in, N_hid) < p_conn
        raw_w = np.full((N_in, N_hid), np.nan, dtype=np.float32)
        if weight_dist == "normal":
            raw_w[mask] = np.clip(rng.normal(0.5, 0.1, size=mask.sum()), 0.0, None)
        else:
            raw_w[mask] = rng.uniform(0.0, 1.0, size=mask.sum())
        mean_w0 = float(np.mean(raw_w[mask])) if mask.sum() > 0 else 0.5

    # Encode spikes
    if encoding == "ttfs":
        spk_idx, spk_t = encode_ttfs(val_subset)
        s_est = 0.85 * (20.0 / max(1.0, tau_m_ms)) * (1.0 / max(0.01, p_conn)) * (0.5 / max(0.01, mean_w0))
    else:
        spk_idx, spk_t = encode_poisson(val_subset, seed=seed)
        s_est = 0.18 * (20.0 / max(1.0, tau_m_ms)) * (1.0 / max(0.01, p_conn)) * (0.5 / max(0.01, mean_w0))

    multipliers = [0.35, 0.55, 0.75, 1.0, 1.3, 1.7, 2.2, 2.8]
    candidate_scales = np.array([s_est * m for m in multipliers], dtype=np.float64)
    K = len(candidate_scales)

    print(f"Calibrating weight scale for {encoding.upper()}, tau_m={tau_m_ms}ms, target={target_rate_hz}Hz, conn={connectivity}...")
    print(f"  Analytic s_est = {s_est:.4f}. Testing {K} parallel candidate scales in 1 build...")

    # Set up single Brian2 C++ run with K parallel subpopulations
    b2.device.reinit()
    b2.device.activate(build_on_run=False)

    build_path = Path(build_dir)
    if build_path.exists():
        shutil.rmtree(build_path, ignore_errors=True)

    if backend == "cpp_standalone":
        b2.set_device("cpp_standalone", directory=str(build_path), build_on_run=False)
        b2.prefs.devices.cpp_standalone.openmp_threads = 0
    else:
        b2.prefs.codegen.target = "cython"

    dt = 0.5 * ms
    b2.defaultclock.dt = dt
    T_enc = 100.0 * ms
    tau_m = tau_m_ms * ms
    v_rest = -65.0 * mV
    v_reset = -65.0 * mV
    v_th = -52.0 * mV
    t_ref = 5.0 * ms

    P = b2.SpikeGeneratorGroup(N_in, spk_idx, spk_t * ms)
    eqs = "dv/dt = (v_rest - v) / tau_m : volt (unless refractory)"
    G = b2.NeuronGroup(
        K * N_hid, eqs,
        threshold="v > v_th", reset="v = v_reset",
        refractory=t_ref, method="exact"
    )
    G.v = v_rest
    reset_op = G.run_regularly("v = v_rest", dt=T_enc, order=-1)

    S = b2.Synapses(P, G, "w : volt", on_pre="v_post += w")
    i_base, j_base = np.where(~np.isnan(raw_w))

    i_all = []
    j_all = []
    w_all = []
    for k, s in enumerate(candidate_scales):
        i_all.append(i_base)
        j_all.append(j_base + k * N_hid)
        w_all.append(raw_w[i_base, j_base] * s)

    S.connect(i=np.concatenate(i_all), j=np.concatenate(j_all))
    S.w = np.concatenate(w_all) * mV

    M_spikes = b2.SpikeMonitor(G)
    net = b2.Network(P, G, S, reset_op, M_spikes)
    net.run(n_sample * T_enc)

    if backend == "cpp_standalone":
        b2.device.build(directory=str(build_path), clean=True)

    spike_i = np.array(M_spikes.i)
    group_ids = spike_i // N_hid
    counts = np.bincount(group_ids, minlength=K)
    rates = counts / (n_sample * (100.0 / 1000.0) * N_hid)

    for k, s in enumerate(candidate_scales):
        print(f"  Scale {s:.4f} -> Mean Rate: {rates[k]:.2f} Hz")

    if backend == "cpp_standalone" and build_path.exists():
        shutil.rmtree(build_path, ignore_errors=True)

    # Find bracketing scales for log-linear interpolation
    best_scale = None
    for k in range(K - 1):
        r1, r2 = rates[k], rates[k + 1]
        s1, s2 = candidate_scales[k], candidate_scales[k + 1]
        if (r1 <= target_rate_hz <= r2) or (r2 <= target_rate_hz <= r1):
            if abs(r2 - r1) > 1e-5:
                alpha = (target_rate_hz - r1) / (r2 - r1)
                log_s = np.log(s1) + alpha * (np.log(s2) - np.log(s1))
                best_scale = float(np.exp(log_s))
            else:
                best_scale = float(s1)
            break

    if best_scale is None:
        idx = int(np.argmin(np.abs(rates - target_rate_hz)))
        best_scale = float(candidate_scales[idx])

    best_scale = float(np.round(best_scale, 4))
    print(f"Selected scale: {best_scale:.4f}")
    calibrated_w = raw_w * best_scale
    return best_scale, calibrated_w


def compute_response_diagnostics(
    spike_counts: np.ndarray,
    images: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute mechanistic hidden response diversity metrics:
    1. mean_response_corr: Mean pairwise Pearson correlation across active hidden neurons.
    2. effective_rank: Participation ratio effective rank (sum lambda)^2 / sum(lambda^2) of feature covariance.
    3. entropy_rank: Spectral entropy rank exp(-sum p_k ln p_k).
    4. mean_ink_corr: Mean correlation of neuron spike counts with total image ink sum_i(x_i).
    5. active_fraction: Fraction of neurons firing at least once across the evaluation set.
    """
    n_samples, n_hid = spike_counts.shape
    counts_f = spike_counts.astype(np.float64)
    stds = np.std(counts_f, axis=0)
    active_mask = stds > 1e-5
    n_active = int(np.sum(active_mask))
    active_fraction = float(n_active / n_hid)

    if n_active >= 2:
        active_counts = counts_f[:, active_mask]
        corr_mat = np.corrcoef(active_counts, rowvar=False)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        triu_idx = np.triu_indices_from(corr_mat, k=1)
        mean_response_corr = float(np.mean(corr_mat[triu_idx]))
    else:
        mean_response_corr = 0.0

    # Effective rank (Participation ratio on covariance eigenvalues)
    X_centered = counts_f - np.mean(counts_f, axis=0, keepdims=True)
    s_vals = np.linalg.svd(X_centered, compute_uv=False)
    eigs = (s_vals ** 2) / max(1, n_samples - 1)
    sum_eigs = np.sum(eigs)
    if sum_eigs > 1e-12:
        effective_rank = float((sum_eigs ** 2) / np.sum(eigs ** 2))
        p_k = eigs / sum_eigs
        p_k = p_k[p_k > 1e-12]
        entropy_rank = float(np.exp(-np.sum(p_k * np.log(p_k))))
    else:
        effective_rank = 1.0
        entropy_rank = 1.0

    # Correlation with total ink sum_i(x_i)
    if images is not None and n_active > 0:
        total_ink = np.sum(images, axis=1)
        ink_corrs = []
        for j in np.where(active_mask)[0]:
            r = np.corrcoef(total_ink, counts_f[:, j])[0, 1]
            if not np.isnan(r):
                ink_corrs.append(r)
        mean_ink_corr = float(np.mean(np.abs(ink_corrs))) if len(ink_corrs) > 0 else 0.0
    else:
        mean_ink_corr = 0.0

    return {
        "mean_response_corr": mean_response_corr,
        "effective_rank": effective_rank,
        "entropy_rank": entropy_rank,
        "mean_ink_corr": mean_ink_corr,
        "active_fraction": active_fraction,
        "n_active": n_active
    }


def compute_analytic_lif_rate(
    images: np.ndarray,
    w_matrix: np.ndarray,
    tau_m_ms: float = 20.0,
    v_rest_mv: float = -65.0,
    v_reset_mv: float = -65.0,
    v_th_mv: float = -52.0,
    t_ref_ms: float = 5.0,
    r_max_hz: float = 100.0
) -> np.ndarray:
    """
    Compute infinite-time deterministic LIF firing rate representation.
    For mean drive mu_j = tau_m * sum_i w_ij * (x_i * r_max):
    If mu_j > Delta V: r_j = 1 / (t_ref + tau_m * ln(mu_j / (mu_j - Delta V)))
    Else: r_j = 0.
    Returns rate matrix of shape (n_images, N_hid) in Hz.
    """
    w_clean = np.nan_to_num(w_matrix, nan=0.0)
    tau_s = tau_m_ms / 1000.0
    delta_v = v_th_mv - v_rest_mv
    # Drive in mV:
    mu = np.dot(images, w_clean) * (r_max_hz * tau_s)
    rates = np.zeros_like(mu, dtype=np.float32)
    supra = mu > delta_v
    if np.any(supra):
        t_ref_s = t_ref_ms / 1000.0
        isi_s = t_ref_s + tau_s * np.log(mu[supra] / (mu[supra] - delta_v))
        rates[supra] = 1.0 / isi_s
    return rates


def train_and_evaluate_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "logistic",
    C: float = 1.0,
    use_cv: bool = False,
    seed: int = 42
) -> Dict[str, Union[float, np.ndarray, str, object]]:
    """
    Standardize spike-count feature vectors and train linear classifier.
    Supports 5-fold cross-validation for hyperparameter tuning.
    Returns test accuracy, macro F1, test predictions, and trained model.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_type == "logistic":
        if use_cv:
            clf = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],
                cv=5,
                max_iter=1000,
                random_state=seed,
                solver="lbfgs"
            )
        else:
            clf = LogisticRegression(
                C=C, max_iter=1000, random_state=seed, solver="lbfgs"
            )
    elif model_type == "ridge":
        if use_cv:
            clf = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        else:
            clf = RidgeClassifier(alpha=1.0, random_state=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    acc = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, digits=4, zero_division=0)

    best_param = getattr(clf, "C_", C) if model_type == "logistic" else getattr(clf, "alpha_", 1.0)
    if isinstance(best_param, np.ndarray):
        best_param = float(np.mean(best_param))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "y_pred": preds,
        "confusion_matrix": cm,
        "classification_report": report,
        "classifier": clf,
        "scaler": scaler,
        "best_param": float(best_param)
    }

