import math
import threading
import time
from typing import Optional

import torch

try:
    import pynvml

    _HAS_NVML = True
except ImportError:
    try:
        import nvidia_ml_py as pynvml

        _HAS_NVML = True
    except ImportError:
        _HAS_NVML = False


class EnergyTracker:
    def __init__(
        self, device: Optional[torch.device] = None, sample_interval_s: float = 0.02
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.is_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.sample_interval_s = sample_interval_s
        self.samples: list[tuple[float, float]] = []  # (timestamp, watts)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.nvml_handle = None
        self.start_time = 0.0
        self.end_time = 0.0
        self.peak_memory_mb = 0.0
        self.energy_joules = 0.0
        self.avg_power_watts = 0.0
        self.elapsed_s = 0.0

        if self.is_cuda and _HAS_NVML:
            try:
                pynvml.nvmlInit()
                device_idx = self.device.index if self.device.index is not None else 0
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            except Exception:
                self.nvml_handle = None

    def _sample_loop(self):
        while self._running:
            t = time.perf_counter()
            power_w = 0.0
            if self.nvml_handle is not None:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                    power_w = power_mw / 1000.0
                except Exception:
                    power_w = 25.0
            else:
                power_w = 25.0
            self.samples.append((t, power_w))
            time.sleep(self.sample_interval_s)

    def start(self):
        self.samples.clear()
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.baseline_memory_bytes = torch.cuda.memory_allocated(self.device)
        self.start_time = time.perf_counter()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.end_time = time.perf_counter()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)

        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            baseline = getattr(self, "baseline_memory_bytes", 0)
            isolated_bytes = (
                max(0, peak_bytes - baseline) if baseline > 0 else peak_bytes
            )
            self.peak_memory_mb = isolated_bytes / (1024.0 * 1024.0)
        else:
            self.peak_memory_mb = 0.0

        self.elapsed_s = max(self.end_time - self.start_time, 1e-5)

        # Numerical integration of power over time (Riemann sum)
        if len(self.samples) >= 2:
            energy = 0.0
            for i in range(len(self.samples) - 1):
                t1, p1 = self.samples[i]
                t2, p2 = self.samples[i + 1]
                dt = t2 - t1
                energy += 0.5 * (p1 + p2) * dt
            total_sample_time = self.samples[-1][0] - self.samples[0][0]
            if total_sample_time > 0 and self.elapsed_s > total_sample_time:
                avg_p = energy / total_sample_time
                energy = avg_p * self.elapsed_s
            self.energy_joules = energy
            self.avg_power_watts = self.energy_joules / self.elapsed_s
        elif len(self.samples) == 1:
            p = self.samples[0][1]
            self.energy_joules = p * self.elapsed_s
            self.avg_power_watts = p
        else:
            p = 25.0
            if self.nvml_handle is not None:
                try:
                    p = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
                except Exception:
                    pass
            self.energy_joules = p * self.elapsed_s
            self.avg_power_watts = p

    def summary(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed_s, 3),
            "energy_joules": round(self.energy_joules, 2),
            "avg_power_w": round(self.avg_power_watts, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
