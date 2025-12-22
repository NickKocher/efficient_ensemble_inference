from math import e
from pathlib import Path
import os
import time
from sklearn import dummy
from contextlib import contextmanager
import psutil

from test.test_permissions import get_min_freq

@contextmanager
def SetFreqMgr(freq, cpu_cores=None, governor="performance", set_min=False):
    if cpu_cores is None:
        cpu_cores = list(os.sched_getaffinity(0))
    if not isinstance(cpu_cores, list):
        cpu_cores = [cpu_cores]
    
    for cpu in cpu_cores:
        set_governor(cpu, governor)
        set_max_freq(cpu, freq, governor)
        if set_min:
            set_min_freq(cpu, freq)
    yield
    for cpu in cpu_cores:
        set_governor(cpu, "performance")
        set_max_freq(cpu, 2800000)
        if set_min:
            set_min_freq(cpu, 1500000)



"""
class SetFreq:
    
    def __init__(self, frequency, governor="performance"):
        self.cores = list(os.sched_getaffinity(0))
        self.frequency = frequency
        self.governor = governor
        
    def __enter__(self):
        for c in self.cores:
            set_governor(c, self.governor)
            set_max_freq(c, self.frequency if self.frequency != -1 else 2800000)
            set_min_freq(c, self.frequency if self.frequency != -1 else 2800000)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for c in self.cores:
            set_governor(c, "performance")
            set_max_freq(c, 2800000)"""



def get_freq_values():
    return [1500000, 2100000, 2800000]


def get_governor(cpu):
    path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
    with open(path, "r") as f:
        return f.read().strip()
    
def get_boost():
    path = "/sys/devices/system/cpu/cpufreq/boost"
    with open(path, "r") as f:
        return f.read().strip()

def set_governor(cpu, governor="userspace"):
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    path.write_text(governor)
       
def get_cpu_freq(cpu):
    path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
    with open(path, "r") as f:
        return int(f.read())
    
def set_max_freq(cpu, freq_khz, governor="performance"):
    if governor == "performance":
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq")
        path.write_text(str(freq_khz))
    else:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_setspeed")
        path.write_text(str(freq_khz))
    
def set_min_freq(cpu, freq_khz):
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_min_freq")
    path.write_text(str(freq_khz))
        
def get_max_freq(cpu):
    path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq"
    with open(path, "r") as f:
        return int(f.read())
    
def get_cpu_freq_linux():
    freqs = []
    with open("/proc/cpuinfo") as f:
        for line in f:
            if "cpu MHz" in line:
                # Extract the number after "cpu MHz"
                freq = float(line.strip().split(":")[1])
                freqs.append(freq)
    return freqs
        
        
def psutil_triple():
    core = os.sched_getaffinity(0).pop()
    cpu_freq = psutil.cpu_freq(percpu=True)[core]
    cur_freq = cpu_freq.current
    min_freq = cpu_freq.min
    max_freq = cpu_freq.max
    return cur_freq, min_freq, max_freq

def _read_energy(core):
    with open(f"/sys/class/hwmon/hwmon3/energy{core+1}_input", "r") as f:
        energy = float(f.read()) / 1e6
    return energy

        
if __name__ == "__main__":
    import os
    import numpy as np
    import time
    
    print("CPU:", os.sched_getaffinity(0))
    cpu = os.sched_getaffinity(0).pop()
    
    def dummy_task(N=12000):
        np.random.seed(0)
        start_energy = _read_energy(cpu)
        start = time.time()
        X = np.random.rand(N, N)
        Y = np.median(X, axis=1)
        Z = np.mean(Y)
        end = time.time()
        end_energy = _read_energy(cpu)
        return end - start, end_energy - start_energy
    
    print("Dummy Task:", dummy_task())
      
    """for freq in [1500000, 2100000, 2800000]:
        with SetFreqMgr(freq, cpu, "performance"):
            freqs = []
            for i in range(10):
                dummy_task(1000)
                freqs.append(get_cpu_freq(cpu))
            print(freq, freqs)"""
    
    for freq in [1500000, 2100000, 2800000]:
        for governor in ["performance", "userspace"]:
            for set_min in [True, False] if governor == "performance" else [False]:
                with SetFreqMgr(freq, cpu, governor, set_min):
                    inf_time, energy = dummy_task(300)
                                
                print(f"Frequency: {freq}, Governor: {governor:>12}, Set min: {str(set_min):>5} => Dummy task: {inf_time:.5f} s, Energy: {energy:.10f} J")
            
    for freq in [1500000, 2100000, 2800000]:
        for governor in ["performance"]:
            for set_min in [True, False]:
                freqs1 = []
                freqs2 = []
                with SetFreqMgr(freq, cpu, governor, set_min):
                    for _ in range(12):
                        dummy_task(400)
                        freqs1.append(get_cpu_freq(cpu))
                        freqs2.append(psutil_triple())
                print(f"Frequency: {freq}, Governor: {governor:>12}, Set min: {str(set_min):>5} => Freqs: {freqs1}")
                
    
    """old_gov = get_governor(cpu)
    print("Old governor:", old_gov)
    
    print("Context Manager class:")
    for freq in [1500000, 2100000, 2800000, -1]:
        for governor in ["performance", "userspace"]:
            with SetFreq(freq, governor):
                inf_time, energy = dummy_task()
            
            print(f"Frequency: {freq}, Governor: {governor}, Dummy task: {inf_time:.3f} s, Energy: {energy:.3f} J")
    
    print("Plain:") 
    for freq in [1500000, 2100000, 2800000]:
        for governor in ["performance", "userspace"]:
            set_governor(cpu, governor)
            set_max_freq(cpu, freq)
            set_max_freq(cpu, freq)
            inf_time, energy = dummy_task()
            print(f"Frequency: {freq}, Governor: {governor}, Dummy task: {inf_time:.3f} s, Energy: {energy:.3f} J")
            
    print("Context Manager class:")
    for freq in [1500000, 2100000, 2800000, -1]:
        for governor in ["userspace", "performance"]:
            with SetFreq(freq, governor):
                inf_time, energy = dummy_task()
            
            print(f"Frequency: {freq}, Governor: {governor}, Dummy task: {inf_time:.3f} s, Energy: {energy:.3f} J")
    
    print("Plain:") 
    for freq in [1500000, 2100000, 2800000]:
        for governor in ["userspace", "performance"]:
            set_governor(cpu, governor)
            set_max_freq(cpu, freq)
            inf_time, energy = dummy_task()
            print(f"Frequency: {freq}, Governor: {governor}, Dummy task: {inf_time:.3f} s, Energy: {energy:.3f} J")
    
    set_governor(cpu, old_gov)
    print("Set governor to old:", get_governor(cpu))
    print("Max freq:", get_max_freq(cpu))
    print("Cur freq:", get_cpu_freq(cpu))
    
    for freq in [1500000, 2100000, 2800000, -1]:
        
        freqs1, freqs2 = [], []
        with SetFreq(freq, "userspace"):
            for _ in range(10):
                dummy_task(1000)
                freqs1.append(get_cpu_freq(cpu))
                freqs2.append(get_cpu_freq_linux()[cpu])
        print("USERSPACE: Freq:", freq, "=>", freqs1, freqs2)
        
        freqs1, freqs2 = [], []
        with SetFreq(freq, "performance"):
            for _ in range(10):
                dummy_task(1000)
                freqs1.append(get_cpu_freq(cpu))
                freqs2.append(get_cpu_freq_linux()[cpu])
        print("PERFORMANCE: Freq:", freq, "=>", freqs1, freqs2)"""
        
        
    # Reset to default
    set_governor(cpu, "performance")
    set_min_freq(cpu, 1500000)
    set_max_freq(cpu, 2800000)
    
    
    print("Dummy task:", dummy_task())
    
    