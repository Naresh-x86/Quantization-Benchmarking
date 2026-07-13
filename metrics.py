import time
import threading
import pynvml
import numpy as np

class GPUTracker:
    def __init__(self, gpu_id=0, poll_interval=0.1):
        self.gpu_id = gpu_id
        self.poll_interval = poll_interval
        self.running = False
        self.thread = None
        self.power_readings = []
        self.memory_readings = []
        self.utilization_readings = []
        self.start_time = None
        self.end_time = None

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except pynvml.NVMLError as e:
            print(f"Warning: NVML not available - {e}")
            self.handle = None

    def _poll_loop(self):
        while self.running:
            if self.handle:
                try:
                    # Power in milliwatts, convert to Watts
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    self.power_readings.append(power)

                    # Memory in bytes, convert to GB
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    self.memory_readings.append(mem_info.used / (1024 ** 3))

                    # Utilization %
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.utilization_readings.append(util.gpu)
                except pynvml.NVMLError:
                    pass
            time.sleep(self.poll_interval)

    def start(self):
        if not self.handle:
            return
        self.power_readings = []
        self.memory_readings = []
        self.utilization_readings = []
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._poll_loop)
        self.thread.start()

    def stop(self):
        if not self.handle:
            return None
        self.running = False
        self.end_time = time.time()
        if self.thread:
            self.thread.join()
        
        duration = self.end_time - self.start_time
        
        if not self.power_readings:
            return None
            
        avg_power = np.mean(self.power_readings)
        peak_power = np.max(self.power_readings)
        energy_joules = avg_power * duration
        
        avg_mem = np.mean(self.memory_readings)
        peak_mem = np.max(self.memory_readings)
        
        avg_util = np.mean(self.utilization_readings)
        
        return {
            "duration_sec": duration,
            "avg_power_W": avg_power,
            "peak_power_W": peak_power,
            "total_energy_J": energy_joules,
            "avg_vram_GB": avg_mem,
            "peak_vram_GB": peak_mem,
            "avg_gpu_util_%": avg_util
        }

    def shutdown(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass
