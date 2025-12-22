import os
import sys
import numpy as np
from contextlib import contextmanager
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_overall_max_freq():
    return 3800000.0


def get_max_freq(core=None):
    if core is None:
        core = os.sched_getaffinity(0).pop()

    cmd = "cat /sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq".format(core)
    stream = os.popen(cmd)
    return int(stream.read())

def get_overall_min_freq():
    return 800000.0


def get_min_freq(core=None):
    if core is None:
        core = os.sched_getaffinity(0).pop()
        
    cmd = "cat /sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq".format(core)
    stream = os.popen(cmd)
    return int(stream.read())
    

def get_cur_freq(core=None):
    if core is None:
        core = os.sched_getaffinity(0).pop()
        
    cmd = "cat /sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq".format(core)
    stream = os.popen(cmd)
    return int(stream.read())


@contextmanager
def disable_freq_scaling():
    """
    Disable frequency scaling for the duration of the context manager.
    """
    SetFreq._disable_freq_scaling = True
    yield
    SetFreq._disable_freq_scaling = False
       
    
class SetFreq:
    
    _disable_freq_scaling = False
    _active_cores = set()
    
    def __init__(self, freq, cores=None, reset_freq=3800000.0, min_freq=get_overall_min_freq(), max_freq=get_overall_max_freq(), verbose=False):
        
        self.cores = cores
        self.freq = freq
        self.reset_freq = reset_freq
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.verbose = verbose
        if self.cores is None:
            self.cores = list(os.sched_getaffinity(0))
        if self.cores == "all":
            self.cores = list(range(0, 56))
            
        assert self.freq is not None, "Frequency cannot be None"        
        assert np.all(np.array(self.cores) >= 0), "Invalid core"
        assert self.freq >= min_freq and self.freq <= max_freq, "Invalid frequency"
        assert self.reset_freq >= min_freq and self.reset_freq <= max_freq, "Invalid reset frequency"
        
    def __enter__(self):
        if not SetFreq._disable_freq_scaling:
            if len(SetFreq._active_cores.intersection(set(self.cores))) > 0:
                raise RuntimeError("Frequency scaling was called in a nested manner for at least one core. This is not allowed.")

            SetFreq._active_cores.update(set(self.cores))
            self.__set_max_freq()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not SetFreq._disable_freq_scaling:
            SetFreq._active_cores.difference_update(set(self.cores))
            self.__reset_max_freq()
            
    def __set_max_freq_per_core_pathlib(self, core, freq):
        path = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_max_freq")
        path.write_text(str(freq))
        
    
    def __set_max_freq(self, reset = False):
        if self.verbose:
            print("Setting max frequency to", self.freq)
            
        freq = self.reset_freq if reset else self.freq
                    
        for core in self.cores:
            self.__set_max_freq_per_core_pathlib(core, freq)
              
    def __reset_max_freq(self):
        if self.verbose:
            print("Resetting max frequency to", self.reset_freq)
            
        self.__set_max_freq(reset=True)

class SetFreqMethodSearch:
    
    _disable_freq_scaling = False
    _verbose_stack_trace = False
    _active_cores = set()
    
    def __init__(self, freq, cores="all", reset_freq=3800000.0, min_freq=get_overall_min_freq(), max_freq=get_overall_max_freq(), verbose=False, method="pathlib", max_workers=1):
        
        self.cores = cores
        self.freq = freq
        self.reset_freq = reset_freq
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.verbose = verbose
        if self.cores is None:
            self.cores = [os.sched_getaffinity(0).pop()]
        if self.cores == "all":
            self.cores = list(range(0, 28)) + list(range(56, 84))
        self.method = method 
        self.max_workers = max_workers
            
        assert self.freq is not None, "Frequency cannot be None"        
        assert np.all(np.array(self.cores) >= 0), "Invalid core"
        assert self.freq >= min_freq and self.freq <= max_freq, "Invalid frequency"
        assert self.reset_freq >= min_freq and self.reset_freq <= max_freq, "Invalid reset frequency"
        assert self.method in ["os", "subprocess", "pathlib"], "Invalid method. Use 'os', 'subprocess' or 'pathlib'."
        
    def __enter__(self):
        if not SetFreq._disable_freq_scaling:
            if SetFreq._verbose_stack_trace:
                import traceback
                
                print("Setting frequency to", self.freq, "for cores", self.cores)
                print("Stack trace:")
                for line in traceback.format_stack():
                    print(line.strip())
            
            if len(SetFreq._active_cores.intersection(set(self.cores))) > 0:
                raise RuntimeError("Frequency scaling was called in a nested manner for at least one core. This is not allowed.")

            SetFreq._active_cores.update(set(self.cores))
            self.__set_max_freq()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not SetFreq._disable_freq_scaling:
            SetFreq._active_cores.difference_update(set(self.cores))
            self.__reset_max_freq()
            
    def __set_max_freq_per_core_pathlib(self, core, freq):
        path = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_max_freq")
        path.write_text(str(freq))
        
    def __set_max_freq_per_core_os(self, core, freq):
        cmd = f"echo {freq} > /sys/devices/system/cpu/cpu{core}/cpufreq/scaling_max_freq"
        os.system(cmd)
    
    def __set_max_freq(self, reset = False):
        if self.verbose:
            print("Setting max frequency to", self.freq)
            
        freq = self.reset_freq if reset else self.freq
            
        if self.method == "os" and self.max_workers == 1:
            for core in self.cores:
                self.__set_max_freq_per_core_os(core, freq)
        elif self.method == "os" and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.__set_max_freq_per_core_os, core, freq) for core in self.cores]
                for future in as_completed(futures):
                    future.result()
        elif self.method == "subprocess":
            procs = []
            for core in self.cores:
                cmd = f"echo {freq} > /sys/devices/system/cpu/cpu{core}/cpufreq/scaling_max_freq"
                proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                procs.append(proc)
            for proc in procs:
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Error setting frequency for core {core}: {proc.stderr.read().decode()}")
        elif self.method == "pathlib" and self.max_workers == 1:
            for core in self.cores:
                self.__set_max_freq_per_core_pathlib(core, freq)
        elif self.method == "pathlib" and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.__set_max_freq_per_core_pathlib, core, freq) for core in self.cores]
                for future in as_completed(futures):
                    future.result()  
              
    def __reset_max_freq(self):
        if self.verbose:
            print("Resetting max frequency to", self.reset_freq)
            
        self.__set_max_freq(reset=True)
        

if __name__ == "__main__":
    
    if len(sys.argv) >= 2 and sys.argv[1] == "--reset":
        with SetFreq(get_overall_max_freq(), cores=list(range(0, 112)), verbose=False) as sf:
            pass
        
        print("\n" + ("="*50))
        print("Reset frequency to", get_overall_max_freq())
        print("="*50)
        print()
        
        num_max_correct = len([i for i in range(0, os.cpu_count()) if abs(get_max_freq(i) - get_overall_max_freq()) < 1e-6])
        num_min_correct = len([i for i in range(0, os.cpu_count()) if abs(get_min_freq(i) - get_overall_min_freq()) < 1e-6])
                
        print("SANITY CHECK:")
        print("# of cores with max frequency = 3800000.0: ", num_max_correct, "out of", os.cpu_count())
        print("# of cores with min frequency = 800000.0: ", num_min_correct, "out of", os.cpu_count())
        
        if not (num_max_correct == os.cpu_count() and num_min_correct == os.cpu_count()):
            print(("==="*50 + "\n")*3)
            print("WARNING: Some cores have not been reset to the default frequency.")
            print("Please check the frequency settings.")
            print(("==="*50 + "\n")*3)
        print("\n" + ("="*50))
        print()
        
    elif len(sys.argv) >= 2 and sys.argv[1] == "--test_speed":
        methods = ["os", "subprocess", "pathlib"]
        max_workers = list(range(1, 32))
        
        min_time = 1e6
        best_method = None
        best_max_worker = None
        
        for method in methods:
            max_workers_method = max_workers if method != "os" else [1]
            for max_worker in max_workers_method:
                times = []
                for i in range(100):
                    start = time.time()
                    with SetFreqMethodSearch(get_overall_max_freq(), cores="all", verbose=False, method=method, max_workers=max_worker) as sf:
                        pass
                    times.append(time.time() - start)
                mean = np.mean(times)
                print(f"Method: {method}, Max Workers: {max_worker}, Time: {mean:.7f} seconds")
            
                if mean < min_time:
                    min_time = mean
                    best_method = method
                    best_max_worker = max_worker
        
        print("\nBest method:", best_method)
        print("Best max workers:", best_max_worker)
        print("Minimum time:", min_time)
    else:
        raise ValueError("Invalid argument. Use --reset to reset the frequency.")