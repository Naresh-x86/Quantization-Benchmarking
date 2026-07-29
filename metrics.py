"""GPU telemetry tracker — modelled on Naresh's GPUTracker.

Polls pynvml every ~50ms during generation to capture power draw (W),
VRAM usage (GB), and GPU utilisation (%) as time series, then reduces
to avg/peak power, avg/peak VRAM, total energy (J), and avg utilisation
once generation stops.
"""

import time
import threading

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False
    print("Warning: pynvml not installed — GPU telemetry will be unavailable. "
          "Install with: pip install pynvml")

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def _mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def _max(lst):
    return max(lst) if lst else 0.0


class GPUTracker:
    """Background-thread GPU telemetry tracker.

    Usage:
        tracker = GPUTracker(gpu_id=0, poll_interval=0.05)
        tracker.start()
        # ... run generation ...
        metrics = tracker.stop()   # returns dict or None
    """

    def __init__(self, gpu_id: int = 0, poll_interval: float = 0.05):
        self.gpu_id = gpu_id
        self.poll_interval = poll_interval
        self.running = False
        self.thread = None

        self.power_readings: list = []
        self.memory_readings: list = []
        self.utilization_readings: list = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        self.handle = None
        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            except Exception as e:
                print(f"Warning: NVML initialisation failed — {e}")

    # ------------------------------------------------------------------
    # Internal poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self):
        while self.running:
            if self.handle is not None:
                try:
                    # Power in milliwatts → Watts
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    self.power_readings.append(power)

                    # Memory in bytes → GB
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    self.memory_readings.append(mem_info.used / (1024 ** 3))

                    # GPU utilisation %
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.utilization_readings.append(util.gpu)
                except Exception:
                    pass  # individual sample failures are non-fatal
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Begin background polling. Clears any previous readings."""
        if self.handle is None:
            return
        self.power_readings = []
        self.memory_readings = []
        self.utilization_readings = []
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict | None:
        """Stop polling and return aggregated GPU metrics dict, or None if unavailable."""
        if self.handle is None:
            return None
        self.running = False
        self.end_time = time.time()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        duration = self.end_time - self.start_time

        if not self.power_readings:
            return None

        if _NUMPY_AVAILABLE:
            avg_power   = float(np.mean(self.power_readings))
            peak_power  = float(np.max(self.power_readings))
            avg_mem     = float(np.mean(self.memory_readings))
            peak_mem    = float(np.max(self.memory_readings))
            avg_util    = float(np.mean(self.utilization_readings))
        else:
            avg_power   = _mean(self.power_readings)
            peak_power  = _max(self.power_readings)
            avg_mem     = _mean(self.memory_readings)
            peak_mem    = _max(self.memory_readings)
            avg_util    = _mean(self.utilization_readings)

        energy_joules = avg_power * duration

        return {
            "duration_sec":     duration,
            "avg_power_W":      avg_power,
            "peak_power_W":     peak_power,
            "total_energy_J":   energy_joules,
            "avg_vram_GB":      avg_mem,
            "peak_vram_GB":     peak_mem,
            "avg_gpu_util_%":   avg_util,
        }

    def shutdown(self):
        """Optionally call at end of process to cleanly shut down NVML."""
        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
