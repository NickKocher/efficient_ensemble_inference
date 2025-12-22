from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import is_
from distributed import Status
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import ConfigSpace as CS
import numpy as np
from ConfigSpace.util import deactivate_inactive_hyperparameters
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    NumericalHyperparameter,
    IntegerHyperparameter,
    Constant,
)
from pynisher import limit, TimeoutException, MemoryLimitException
import traceback
from time import time as time_s
import logging

from mosmac3.smac.utils.configspace import get_config_hash
from raaml.base_model_generation.pd_base_model_pool import BaseModelPool

def check_valid_clause_or_relation(vector, clause):
    return not clause.is_forbidden_vector(vector)

def check_valid_and_conjunction(vector, and_conjunction):
    """
    Check if the given vector satisfies the conditions of an AND conjunction.
    """
    for dlc in and_conjunction.dlcs:
        if isinstance(dlc, CS.ForbiddenAndConjunction):
            valid = check_valid_and_conjunction(vector, dlc)
        else:
            valid = check_valid_clause_or_relation(vector, dlc)
        if valid:
            return True
    return False

def is_valid(vector, forbiddens):
    for forbidden in forbiddens:
        if isinstance(forbidden, CS.ForbiddenAndConjunction):
            if not check_valid_and_conjunction(vector, forbidden):
                return False
        else:
            if not check_valid_clause_or_relation(vector, forbidden):
                return False
    return True

class NSGAIIProblem(Problem):
    def __init__(self, config_space, seed, objective_names, target_fun, n_jobs, time_limit_s=np.inf, memory_limit_mb=np.inf, update_fun=None):
        self.config_space = config_space
        self.seed = seed
        self.objectives = objective_names
        self.target_fun = target_fun
        self.meta = self.__preprocess_hyperparameters(config_space)
        self.n_jobs = n_jobs
        self.time_limit = time_limit_s
        self.memory_limit = memory_limit_mb
        self.update_fun = update_fun
        
        self.limit_args = {}
        
        if time_limit_s != np.inf:
            self.limit_args['wall_time'] = (time_limit_s, "s")
        if memory_limit_mb != np.inf:
            self.limit_args['memory'] = (memory_limit_mb, "MB")
        
        super().__init__(
            n_var=len(config_space),
            n_obj=len(objective_names), 
            n_ieq_constr=0,
            xl=[hp.lower_vectorized for hp in config_space.get_hyperparameters()],
            xu=[hp.upper_vectorized for hp in config_space.get_hyperparameters()],
            save_history=True
        )
        
    def set_start_time(self, time_s, max_runtime_s):
        self.start_time = time_s
        self.max_runtime_s = max_runtime_s
        
    def __preprocess_hyperparameters(self, cs):
        hps = list(cs.values())
        cat_indices, num_indices, int_indices, const_ind = [], [], [], []
        cat_lens, int_ranges = [], []
        default_values = [hp.to_vector(hp.default_value) for hp in hps]
        
        for i, hp in enumerate(hps):
            if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                cat_indices.append(i)
                if isinstance(hp, CategoricalHyperparameter):
                    cat_lens.append(len(hp.choices))
                else:
                    cat_lens.append(len(hp.sequence))
            elif isinstance(hp, IntegerHyperparameter):
                int_indices.append(i)
                int_ranges.append(hp.upper - hp.lower)
            elif isinstance(hp, NumericalHyperparameter):
                num_indices.append(i)
            elif isinstance(hp, Constant):
                const_ind.append(i)
            else:
                raise TypeError(f"Unknown hyperparameter type: {type(hp)}")
            
        meta = {
            'cat': np.array(cat_indices),
            'num': np.array(num_indices),
            'int': np.array(int_indices),
            'const': np.array(const_ind),
            'cat_lengths': np.array(cat_lens),
            'int_ranges': np.array(int_ranges),
            'default_values': np.array(default_values),
        }
            
        return meta
    
    @staticmethod
    def get_base_model_pool(result, nsga2_problem, identifier, bmg_start_time):
        """
        Create a SimpleBaseModelPool instance.
        """
              
        pool = BaseModelPool("nsga2", identifier, nsga2_problem.config_space, nsga2_problem.objectives)
        
        final_pop = []
        for ind in result.pop:
            add_info = ind.get("additional_info")
            if not add_info["status"] == "IGNORE":
                final_pop.append(get_config_hash(add_info["config"], 20))
                
        logging.info(f"Final population configs: {[get_config_hash(config) for config in final_pop]}")
        
        for i, alg in enumerate(result.history):
            opt = [get_config_hash(entry.get("additional_info").get("config"), 20) for entry in alg.opt]
            pop = [get_config_hash(entry.get("additional_info").get("config"), 20) for entry in alg.pop]
            for ind in alg.off:
                add_info = ind.get("additional_info")
                if not add_info["status"] == "IGNORE":
                    cfg_hash = get_config_hash(add_info["config"], 20)
                    is_final = cfg_hash in final_pop
                    is_pop = cfg_hash in pop
                    is_opt = cfg_hash in opt
                    nsga2_info = f"{i}_{is_final}_{is_pop}_{is_opt}"
                    pool.add_entry(
                        config = add_info["config"],
                        result = add_info["results"],
                        time = add_info["time"] - bmg_start_time,
                        status = add_info["status"],
                        nsga2_info = nsga2_info
                    )

        return pool
    
    
    @staticmethod  
    def evaluate_single_static(i, config_vector, config_space, target_fun, limit_args, seed, objectives, update_fun):
        config = CS.Configuration(config_space, vector=config_vector)

        try:
            with limit(target_fun, **limit_args) as limited_target_fun:
                results, additional_info = limited_target_fun(config, seed)
            status = "SUCCESS"
        except (TimeoutException, MemoryLimitException) as e:
            logging.warning(f'{"Time" if isinstance(e, TimeoutException) else "Memory"} limit exceeded for configuration {i}')
            logging.debug(traceback.format_exc())
            status = "TIMEOUT" if isinstance(e, TimeoutException) else "MEMOUT"
        except Exception as e:
            logging.warning(f"Exception occurred for configuration {i}: {e}")
            logging.debug(traceback.format_exc())
            status = "CRASHED"

        if status != "SUCCESS":
            results = {obj: np.inf for obj in objectives}
            additional_info = {
                "model_paths" : [None],
                "results" :  {obj: [np.inf] for obj in objectives}
            }
            
        if update_fun is not None:
            update_fun("bmg", status, {obj: np.mean(results[obj]) for obj in objectives} if status == "SUCCESS" else None)

        return (
            i,
            np.array([results[obj] for obj in objectives]),
            additional_info,
            config,
            time_s(),
            status
        )


    def _evaluate(self, x, out, *args, **kwargs):
        out_arr = np.zeros((len(x), len(self.objectives)))
        additional_info_pop = [None] * len(x)
        
        logging.debug(f"Evaluation {len(x)} configurations in NSGA-II iteration")
        logging.debug(f"Configurations: {[get_config_hash(CS.Configuration(self.config_space, vector=config)) for config in x]}")
            
        futures = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for i, config_vec in enumerate(x):
                futures.append(
                    executor.submit(
                        NSGAIIProblem.evaluate_single_static,
                        i,
                        config_vec,
                        self.config_space,
                        self.target_fun,
                        self.limit_args,
                        self.seed,
                        self.objectives,
                        self.update_fun
                    )
                )
                
            completed_futures = []
            for future in as_completed(futures):
                i, out_res, info = self.get_result(future)
                out_arr[i], additional_info_pop[i] = out_res, info
                completed_futures.append(future)
                
                if time_s() - self.start_time > self.max_runtime_s:
                    executor.shutdown(wait=True, cancel_futures=True)
                    for f in futures:
                        if not f.cancelled() and f not in completed_futures:
                            i, out_res, info = self.get_result(f)
                            out_arr[i], additional_info_pop[i] = out_res, info
                            completed_futures.append(future)
                    for i in range(len(x)):
                        if additional_info_pop[i] is None:
                            additional_info_pop[i] = {"status": "IGNORE", "config": None}
                            out_arr[i] = np.full(len(self.objectives), np.inf)
                    break
            
        out["F"] = out_arr
        out["additional_info"] = additional_info_pop
            
    def get_result(self, future):
        i, obj_array, additional_info, config, end_time, status = future.result()
        logging.debug(f"Finished evaluation for configuration {i} with status {status}")
        
        add_info = {
            "config": config,
            "results" : additional_info["results"],
            "time": end_time,
            "status" : status
        }
        
        return i, obj_array, add_info
        
        
class ConfigSpaceSampling(Sampling):
    def __init__(self):
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        configs = problem.config_space.sample_configuration(n_samples)
        return np.array([config.get_array() for config in configs])
    

class ConfigSpaceCrossover(Crossover):
    def __init__(self, max_attempts=100, probability=0.5):
        super().__init__(2, 1)
        self.max_attempts = max_attempts
        self.probability = probability

    def _do(self, problem, X, **kwargs):
        cs = problem.config_space
        
        n_matings = X.shape[1]
        ret = np.zeros((1, n_matings, len(cs)))
        for i in range(n_matings):
            config1_arr = X[0, i]
            config2_arr = X[1, i]
            active1 = ~np.isnan(config1_arr)
            active2 = ~np.isnan(config2_arr)
            
            fallback = np.full_like(config1_arr, np.nan)
            fallback[active1] = config1_arr[active1]
            fallback[active2] = config2_arr[active2]
            
            
            for _ in range(self.max_attempts):
                mask = np.random.rand(len(config1_arr)) < self.probability
                crossover_arr = np.where(mask, config1_arr, config2_arr)
                crossover_arr[np.isnan(crossover_arr)] = fallback[np.isnan(crossover_arr)]
                
                try:
                    config = CS.Configuration(cs, vector=crossover_arr, allow_inactive_with_values=True)
                    config = deactivate_inactive_hyperparameters(config, cs)
                    ret[0,i] = config.get_array()
                    break
                except CS.exceptions.ForbiddenValueError:
                    continue
            else:
                # Fallback if no valid crossover found
                ret[0,i] = config1_arr if np.random.rand() < 0.5 else config2_arr
                
        return ret
    
    
class ConfigSpaceMutation(Mutation):
    def __init__(self, std=0.1, probability=0.5, max_attempts=100):
        super().__init__(1)
        self.std = std
        self.probability = probability
        self.max_attempts = max_attempts
    
    def _do(self, problem, X, **kwargs):
        meta = problem.meta
        
        full_ret = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            arr = X[i]
            active = ~np.isnan(arr)
            
            logging.debug(f"Config before: {CS.Configuration(problem.config_space, vector=arr)}")
            
            for k in range(self.max_attempts):
                
                ret = np.full_like(arr, np.nan)
            
                cat_ind = meta['cat'] 
                num_ind = meta['num']
                int_ind = meta['int']
                const_ind = meta['const']
                cat_lens = meta['cat_lengths']
                int_ranges = meta['int_ranges']
                default_values = meta['default_values']
                
                cat_lens = cat_lens[active[cat_ind]]
                int_ranges = int_ranges[active[int_ind]]
                cat_ind = cat_ind[active[cat_ind]]
                num_ind = num_ind[active[num_ind]]
                int_ind = int_ind[active[int_ind]]
                if len(const_ind) > 0:
                    const_ind = const_ind[active[const_ind]]
                
                do_mutation_cat = np.random.choice([0,1], p=[1-self.probability, self.probability], size=len(cat_ind))
                new_cat_vals = np.array([np.random.choice(l) for l in cat_lens])
                ret[cat_ind] = (arr[cat_ind] * (1 - do_mutation_cat) + do_mutation_cat * new_cat_vals).astype(int)
                do_mutation_num = np.random.choice([0,1], p=[1-self.probability, self.probability], size=len(num_ind))
                ret[num_ind] = np.clip(arr[num_ind] + do_mutation_num * np.random.normal(0, self.std), 0, 1)
                do_mutation_int = np.random.choice([0,1], p=[1-self.probability, self.probability], size=len(int_ind))
                ret[int_ind] = np.clip(arr[int_ind] + do_mutation_int * np.random.normal(0, self.std), 0, 1)
                ret[int_ind] = np.clip(np.round(ret[int_ind] * int_ranges) / int_ranges, 0, 1)
                if len(const_ind) > 0:
                    ret[const_ind] = arr[const_ind] 
                
                ret[~active] = default_values[~active]
                
                
                valid = is_valid(ret, problem.config_space.forbidden_clauses)
                if not valid:
                    continue
                else:
                    try:
                        config = CS.Configuration(problem.config_space, vector=ret, allow_inactive_with_values=True)
                        config = deactivate_inactive_hyperparameters(config, problem.config_space)
                    except CS.exceptions.ForbiddenValueError:
                        continue
                    
                    full_ret[i] = config.get_array()
                    break
            else:
                # Fallback if no valid mutation found
                full_ret[i] = X[i]
                logging.debug("Could not find valid mutation")
                
        return full_ret