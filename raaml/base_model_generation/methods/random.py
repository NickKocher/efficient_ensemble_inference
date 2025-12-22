import time
import numpy as np
from pynisher import limit, TimeoutException, MemoryLimitException
import ConfigSpace as CS
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import os
import pandas as pd
import traceback
import logging

from raaml.base_model_generation.pd_base_model_pool import BaseModelPool
from raaml.util.non_dominated_sorting import nsga2_sort 
from mosmac3.smac.utils.configspace import get_config_hash


class RandomSearch:
    
    def __init__(
        self,
        config_space,
        n_cores_search,
        t_search_sec,
        trial_search_sec,
        trial_memory_mb,
        seed,
        identifier,
        path,
        cost_names
    ):
        self.config_space = config_space
        self.n_cores_search = n_cores_search
        self.t_search_sec = t_search_sec
        self.trial_search_sec = trial_search_sec
        self.trial_memory_mb = trial_memory_mb
        self.seed = seed
        self.history = []
        self.identifier = identifier
        self.path = path
        self.cost_names = cost_names
    
    
    def _run_config_safe(self, eval_fn, config):
        try:
            with limit(eval_fn, wall_time=(self.trial_search_sec, "s"), memory=(self.trial_memory_mb, "MB")) as limited_eval_fn:
                return limited_eval_fn(config, self.seed)
        except (TimeoutException, MemoryLimitException) as e:
            logging.warning(f'{"Time" if isinstance(e, TimeoutException) else "Memory"} limit exceeded')
            logging.debug(traceback.format_exc())
            return "TIMEOUT" if isinstance(e, TimeoutException) else "MEMOUT", None
        except Exception as e:
            logging.warning(f"Exception occurred: {e}")
            logging.debug(traceback.format_exc())
            return "CRASHED", None
        
    def _add_new_config(self, pending_futures, future_info, executor, eval_fn, config):
        logging.debug(f"Adding new config to queue: {get_config_hash(config)}")
        future = executor.submit(self._run_config_safe, eval_fn, config)
        pending_futures.append(future)
        future_info[future] = config
        
    
    def _evaluate_runinfo(self, finished_info):
        pool = BaseModelPool("random", self.identifier, self.config_space, self.cost_names)
        
        for entry in finished_info:
            config = entry["config"]
            add_info = entry["add_info"]
            time = entry["time"]
            status = entry["status"]

            if add_info is not None:
                results = add_info["results"]
            else:
                results = {obj: [np.inf] for obj in self.cost_names}

            pool.add_entry(config, results, time, status)
        
        return pool
        
    def run(self, eval_fn, update_fun=None):
        start_time = time.time()
        pending_futures = []
        future_info = {}
        finished_info = []
        
        with ProcessPoolExecutor(max_workers=self.n_cores_search) as executor:
            for _ in range(self.n_cores_search):
                config = self.config_space.sample_configuration()
                self._add_new_config(pending_futures, future_info, executor, eval_fn, config)
                
            while True:
                done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
                for f in done:                    
                    pending_futures.remove(f)
                    config = future_info.pop(f)
                    result, add_info = f.result()
                    if type(result) == str:
                        finished_info.append({
                            "config": config,
                            "result": {k: np.inf for k in self.cost_names},
                            "add_info": add_info,
                            "status" : result
                        })
                    else:
                        finished_info.append({
                            "config": config,
                            "result": result,
                            "add_info": add_info,
                            "status" : "SUCCESS"
                        })
                    finished_info[-1]["time"] = time.time() - start_time
                    
                    if update_fun is not None:
                        update_fun("bmg", result if type(result) == str else "SUCCESS", None if type(result) == str else {obj: np.mean(result[obj]) for obj in result.keys()})

                if time.time() - start_time > self.t_search_sec:
                    executor.shutdown(wait=True, cancel_futures=True)
                    return self._evaluate_runinfo(finished_info)

                while len(pending_futures) < self.n_cores_search:
                    config = self.config_space.sample_configuration()
                    self._add_new_config(pending_futures, future_info, executor, eval_fn, config)
