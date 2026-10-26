"""
CPU throughput benchmark script for Phase 0b.
Tests Cython / C++ compiler integration in Brian2, measures execution time
on 100 and 500 MNIST samples, verifies spike statistics, and extrapolates runtimes.
"""

import time
import os
import sys
from pathlib import Path
import numpy as np

# Ensure Brian2 cache directory is local to avoid path issues
os.environ["BRIAN_CACHE_DIR"] = str(Path("results/brian_cache").resolve())

import brian2 as b2
from brian2 import ms, mV, Hz


def test_compiler():
    print("--- Compiler Smoke Test ---")
    print(f"Current codegen.target preference: {b2.prefs.codegen.target}")
    
    # Try Cython compilation
    b2.prefs.codegen.target = "cython"
    b2.seed(42)
    np.random.seed(42)
    
    eqs = """
    dv/dt = (v_rest - v)/tau_m : volt (unless refractory)
    """
    try:
        start = time.time()
        G = b2.NeuronGroup(10, eqs, threshold="v > -50*mV", reset="v = -65*mV",
                           refractory=5*ms, method="exact",
                           namespace={"v_rest": -65*mV, "tau_m": 20*ms})
        G.v = -65*mV
        net = b2.Network(G)
        net.run(10*ms)
        elapsed = time.time() - start
        print(f"SUCCESS: Cython target compiled and ran in {elapsed:.3f} s.")
        return "cython"
    except Exception as e:
        print(f"Cython compilation failed: {e}")
        print("Falling back to numpy target...")
        b2.prefs.codegen.target = "numpy"
        start = time.time()
        G = b2.NeuronGroup(10, eqs, threshold="v > -50*mV", reset="v = -65*mV",
                           refractory=5*ms, method="exact",
                           namespace={"v_rest": -65*mV, "tau_m": 20*ms})
        G.v = -65*mV
        net = b2.Network(G)
        net.run(10*ms)
        elapsed = time.time() - start
        print(f"Numpy target completed in {elapsed:.3f} s.")
        return "numpy"


def run_benchmark(n_images: int = 100, target: str = "cython"):
    print(f"\n--- Running Benchmark on {n_images} Images (Target: {target}) ---")
    b2.prefs.codegen.target = target
    
    # Load sample images from cache
    data_file = Path("data/mnist.npz")
    if not data_file.exists():
        print("Warning: data/mnist.npz not found, using synthetic random images for test.")
        images = np.random.uniform(0, 1, size=(n_images, 784)).astype(np.float32)
    else:
        mnist_data = np.load(data_file)
        images = mnist_data["X_train"][:n_images]
    
    N_in = 784
    N_hid = 400
    T_enc = 100 * ms
    dt = 0.5 * ms
    b2.defaultclock.dt = dt
    
    tau_m = 20 * ms
    v_rest = -65 * mV
    v_reset = -65 * mV
    v_th = -52 * mV
    t_ref = 5 * ms
    r_max = 100 * Hz
    
    # Weight calibration factor: with ~784 inputs at ~10-20 Hz each, w around 0.05-0.1 mV
    w_mean = 0.05 * mV
    
    # Structure simulation as a single contiguous run using TimedArray
    # Shape: (steps, N_in)
    steps_per_image = int(round(T_enc / dt))
    total_steps = n_images * steps_per_image
    
    print(f"Preparing TimedArray for {n_images} images ({total_steps} timesteps @ dt={dt})...")
    # Rate matrix: repeat each image row across its steps
    rates_matrix = np.repeat(images * float(r_max / Hz), steps_per_image, axis=0) * Hz
    rate_array = b2.TimedArray(rates_matrix, dt=dt)
    
    P = b2.PoissonGroup(N_in, rates="rate_array(t, i)")
    
    eqs = """
    dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
    """
    G = b2.NeuronGroup(N_hid, eqs, threshold="v > v_th", reset="v = v_reset",
                       refractory=t_ref, method="exact")
    G.v = v_rest
    
    # Synapses: dense
    np.random.seed(42)
    S = b2.Synapses(P, G, "w : volt", on_pre="v_post += w")
    S.connect(p=1.0)
    # Gaussian weights clipped to positive
    weights = np.clip(np.random.normal(0.5, 0.1, size=len(S)), 0, None)
    S.w = weights * (w_mean / (0.5 * mV)) * mV
    
    # Spike monitor on hidden layer
    M_spikes = b2.SpikeMonitor(G)
    
    # Reset hidden membrane potentials at start of each image
    reset_op = G.run_regularly("v = v_rest", dt=T_enc, order=-1)
    
    net = b2.Network(P, G, S, M_spikes, reset_op)
    
    print(f"Simulating {n_images * T_enc / ms:.0f} ms of biological time...")
    t_start = time.perf_counter()
    net.run(n_images * T_enc)
    wall_clock = time.perf_counter() - t_start
    
    total_spikes = len(M_spikes.i)
    spikes_per_image = total_spikes / n_images
    mean_rate = total_spikes / (n_images * float(T_enc / b2.second) * N_hid)
    s_per_image = wall_clock / n_images
    
    print(f"Wall-clock time: {wall_clock:.2f} s")
    print(f"Throughput: {s_per_image*1000:.2f} ms/image ({1.0/s_per_image:.1f} images/s)")
    print(f"Hidden spikes: {total_spikes} total ({spikes_per_image:.1f}/image, mean firing rate: {mean_rate:.2f} Hz)")
    
    # Projection for 12,000 images (10k train + 2k test)
    proj_12k_sec = s_per_image * 12000
    print(f"Projected runtime for 12,000 images (1 config): {proj_12k_sec/60:.2f} minutes")
    print(f"Projected runtime for 15 configs: {(proj_12k_sec * 15)/3600:.2f} hours")
    
    return {
        "n_images": n_images,
        "target": target,
        "wall_clock": wall_clock,
        "s_per_image": s_per_image,
        "spikes_per_image": spikes_per_image,
        "mean_rate_hz": mean_rate,
        "proj_12k_min": proj_12k_sec / 60
    }


if __name__ == "__main__":
    active_target = test_compiler()
    res100 = run_benchmark(n_images=100, target=active_target)
