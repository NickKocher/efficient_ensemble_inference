import yaml
import os
import fcntl
import psutil
import logging
from time import time as time_s

class AffinityManager:
    
    def __init__(self, cores, n_cores_worker, file_path):
        self.cores = cores
        self.n_cores_worker = n_cores_worker
        self.file_path = file_path
        
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path))
        
        with open(self.file_path, "w") as f:
            yaml.safe_dump({c: -1 for c in self.cores}, f)
        
    def _get_free_cores(self, cores_dict):
        return [c for c in cores_dict if cores_dict[c] == -1]
    
    def _set_cores(self, cores_dict, cores, process_id):
        for core in cores:
            cores_dict[core] = process_id
        
        if process_id == -1:
            # Reset affinity
            p = psutil.Process(os.getpid())
            p.cpu_affinity(self.cores)
        else:
            # Set affinity
            p = psutil.Process(process_id)
            p.cpu_affinity(cores)
        
        return cores_dict
        
    def reserve_and_set_cores(self, n_cores=None):
        n_cores = self.n_cores_worker if n_cores is None else n_cores
        with open(self.file_path, "r+") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            
            cores_dict = yaml.safe_load(file)
            cores_dict = self._release_cores(cores_dict)
            cores_free = self._get_free_cores(cores_dict)
            if len(cores_free) < n_cores:
                raise ValueError(f"Not enough cores available. Expected at least {n_cores}, but got {len(cores_free)}.")
            cores_dict = self._set_cores(cores_dict, cores_free[:n_cores], os.getpid())
            
            file.seek(0)
            yaml.safe_dump(cores_dict, file)
            file.truncate()
            logging.debug(f"Reserving cores {cores_free[:n_cores]} for process {os.getpid()}.")
            
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            
    # def release_cores(self, cores=None):
    #     if cores is None:
    #         cores = list(os.sched_getaffinity(0))
        
    #     with open(self.file_path, "r+") as file:
    #         fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            
    #         cores_dict = yaml.safe_load(file)
    #         cores_dict = self._release_cores(cores_dict)
    #         logging.debug(f"Manually releasing cores {cores} for process {os.getpid()}.")
    #         cores_dict = self._set_cores(cores_dict, cores, -1)
            
    #         file.seek(0)
    #         yaml.safe_dump(cores_dict, file)
    #         file.truncate()
            
    #         fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                
    def _is_proc_running(self, proc_id):
        try:
            proc = psutil.Process(proc_id)
            return proc.is_running()
        except psutil.NoSuchProcess:
            return False
        
    def _release_cores(self, cores_dict):
        for core in self.cores:
            proc_id = cores_dict[core]
            if proc_id != -1:
                if not self._is_proc_running(proc_id):
                    cores_dict[core] = -1
                    logging.debug(f"Process {proc_id} finished. Releasing core {core}.")
        return cores_dict