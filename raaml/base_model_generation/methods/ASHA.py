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


class ASHA:
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
        cost_names,
        multi_objective,
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
        self.multi_objective = multi_objective

    def _run_config_safe(self, eval_fn, config, n_cv, max_n_cv):
        # Scale the trial search time linearly with the number of CV training workload
        trial_search_sec = int(self.trial_search_sec * ((n_cv-1) / (max_n_cv-1) if n_cv > 1 else 0.125))
        
        try:
            with limit(eval_fn, wall_time=(trial_search_sec, "s"), memory=(self.trial_memory_mb, "MB")) as limited_eval_fn:
                return limited_eval_fn(config, self.seed, n_cv, trial_search_sec)
        except (TimeoutException, MemoryLimitException) as e:
            logging.warning(f'{"Time" if isinstance(e, TimeoutException) else "Memory"} limit exceeded')
            logging.debug(traceback.format_exc())
            return "TIMEOUT" if isinstance(e, TimeoutException) else "MEMOUT", None
        except Exception as e:
            logging.warning(f"Exception occurred: {e}")
            logging.debug(traceback.format_exc())
            return "CRASHED", None

    def _evaluate_runinfo(self, finished_info, max_n_cv):
        pool = BaseModelPool("mo-asha", self.identifier, self.config_space, self.cost_names)

        for n_cv in reversed(finished_info.keys()):
            for info in finished_info[n_cv]:
                config = info["config"]
                add_info = info["add_info"]
                time = info["time"]
                status = info["status"]

                if add_info is not None:
                    results = add_info["results"]
                else:
                    results = {obj: [np.inf] for obj in self.cost_names}
                    
                status = "INTERMEDIATE_SUCCESS" if n_cv != max_n_cv and status == "SUCCESS" else status

                pool.add_entry(config, results, time, status)

        return pool
    
    def _add_new_config(self, pending_futures, future_info, executor, eval_fn, config, n_cv, max_n_cv):
        future = executor.submit(self._run_config_safe, eval_fn, config, n_cv, max_n_cv)
        pending_futures.append(future)
        future_info[future] = {
            "config": config,
            "n_cv": n_cv,
        }
        
    
    def _non_dominated_selection(self, finished_info, finished_success, n_cv, eta):
        # Create initial dataframe with data from current n_cv
        data_list = [entry["result"] for entry in finished_success]
        n_cv_list = [n_cv] * len(data_list)
        
        # Add data from other n_cv levels
        for n_cv_ in finished_info.keys():
            if n_cv_ == n_cv:
                data_list.extend(entry["result"] for entry in finished_info[n_cv][eta:])
                n_cv_list.extend([n_cv_] * len(finished_info[n_cv_][eta:]))
            else:
                data_list.extend(entry["result"] for entry in finished_info[n_cv_])
                n_cv_list.extend([n_cv_] * len(finished_info[n_cv_]))
        
        # Create dataframe at once instead of repeated appends
        dataframe = pd.DataFrame(data_list, columns=self.cost_names)
        
        # Apply NSGA-II sorting
        ranked = nsga2_sort(dataframe)
        
        # Sort by rank and crowding distance
        sorted_df = ranked.sort_values(['rank', 'crowding_distance'], 
                         ascending=[True, False])
        
        
        # Get the first index that's within the eta range
        for idx in sorted_df.index:
            if idx < eta:
                return finished_success[idx]["config"]
                    
                    
    def _single_objective_top_config(self, finished_success):
        results = [entry["result"][self.cost_names[0]] for entry in finished_success]
        best_idx = np.argmin(results)
        return finished_success[best_idx]["config"]
    

    def optimize(self, eval_fn, n_cv_steps, eta=3, update_fun=None):
        start_time = time.time()
        next_map = {n_cv_steps[i]: n_cv_steps[i + 1] for i in range(len(n_cv_steps) - 1)}
        
        pending_futures = []
        future_info = {}
        finished_info = {n : [] for n in n_cv_steps}
        
        with ProcessPoolExecutor(max_workers=self.n_cores_search) as executor:
            for _ in range(self.n_cores_search):
                config = self.config_space.sample_configuration()
                self._add_new_config(pending_futures, future_info, executor, eval_fn, config, n_cv_steps[0], n_cv_steps[-1])
            
            while True:
                done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
                for f in done:                    
                    pending_futures.remove(f)
                    f_info = future_info.pop(f)
                    result, add_info = f.result()
                    if type(result) == str:
                        finished_info[f_info["n_cv"]].append({
                            "config": f_info["config"],
                            "result": {k: np.inf for k in self.cost_names},
                            "add_info": add_info,
                            "status" : result
                        })
                    else:
                        finished_info[f_info["n_cv"]].append({
                            "config": f_info["config"],
                            "result": result,
                            "add_info": add_info,
                            "status" : "SUCCESS"
                        })
                        
                    if f_info["n_cv"] == n_cv_steps[-1] and update_fun is not None:
                        if add_info is None:
                            objective_values = None
                        else:
                            results = add_info["results"]
                            objective_values = {obj_name: np.mean(results[obj_name]) for obj_name in results.keys()}
                        update_fun("bmg", result if type(result) == str else "SUCCESS", objective_values)
                        
                    finished_info[f_info["n_cv"]][-1]["time"] = time.time() - start_time

                if time.time() - start_time > self.t_search_sec:
                    return self._evaluate_runinfo(finished_info, max_n_cv=n_cv_steps[-1])

                for n_cv in reversed(n_cv_steps[:-1]):
                    finished_success = [entry for entry in finished_info[n_cv] if entry["status"] == "SUCCESS"]
                    logging.debug(f"Finished success of n_cv {n_cv}: {[get_config_hash(entry['config']) for entry in finished_success]}")
                    while len(finished_success) >= eta:
                        if self.multi_objective:
                            top_config = self._non_dominated_selection(finished_info, finished_success, n_cv, eta)
                        else:
                            top_config = self._single_objective_top_config(finished_success)
                        logging.debug(f"Selected config: {get_config_hash(top_config)}")
                        for entry in finished_success[:eta]:
                            promoted = get_config_hash(entry["config"], 20) == get_config_hash(top_config, 20)
                            status = entry["status"]
                            entry["status"] = status if status != "SUCCESS" else f"INTERMEDIATE_SUCCESS{'_NOT' if not promoted else ''}_PROMOTED"
                            finished_success.remove(entry)
                        self._add_new_config(pending_futures, future_info, executor, eval_fn, top_config, next_map[n_cv], n_cv_steps[-1])
                    
                while len(pending_futures) < self.n_cores_search:
                    config = self.config_space.sample_configuration()
                    self._add_new_config(pending_futures, future_info, executor, eval_fn, config, n_cv_steps[0], n_cv_steps[-1])
