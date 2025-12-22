from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from raaml.ensembling.scheduling.scheduler import Task, ListScheduler
from raaml.ensembling.ensembling_predictions import EnsemblingPredictions


class AllotmentStrategy(ABC):
    
    def __init__(self, max_n_cores : int):
        self.max_n_cores = max_n_cores
    
    @abstractmethod
    def compute_allotments(self, ensembling_predictions : EnsemblingPredictions, scheduler : ListScheduler, config_ids : list):
        raise NotImplementedError()
    
    def compute_resources(self, ensembling_predictions : EnsemblingPredictions, scheduler : ListScheduler, allotment : dict, config_ids : list):
        tasks = []
        for config_id in config_ids:
            n_cores, frequency = allotment[config_id]
            
            inf_time, energy = ensembling_predictions.get_resources(config_id, n_cores, frequency, ["inference_time", "energy_consumption"])
            tasks.append(Task(config_id, n_cores, frequency, inf_time, energy))
            
        inf_time, energy, _, _ = scheduler.compute_offline_resources(tasks)
            
        return inf_time, energy
    
# class BaseAllotment(AllotmentStrategy):
#     def __init__(self, max_n_cores : int, frequency_strategy : str = "max"):
#         self.frequency_strategy = frequency_strategy
#         super().__init__(max_n_cores)
    
#     def compute_allotments(self, ensembling_predictions : EnsemblingPredictions, scheduler : ListScheduler, config_ids : list):
#         if self.frequency_strategy == "max":
#             frequency = ensembling_predictions.results["frequency"].max()
#         elif self.frequency_strategy == "min":
#             frequency = ensembling_predictions.results["frequency"].min()
#         else:
#             raise ValueError(f"Unknown frequency strategy: {self.frequency_strategy}")
        
#         return [{c_id : (1, frequency) for c_id in config_ids}]
   
    
class MaxNThreadsAllotment(AllotmentStrategy):
    
    def __init__(self, max_n_cores : int):
        super().__init__(max_n_cores)
        
    def compute_allotments(self, ensembling_predictions : EnsemblingPredictions, scheduler : ListScheduler, config_ids : list):
        max_frequency = np.max(ensembling_predictions.results["frequency"])
        
        return [{c_id : (self.max_n_cores, max_frequency) for c_id in config_ids}]
    
    
class FullFrequencyScalingAllotment(AllotmentStrategy):
    def __init__(self, max_n_cores : int):
        super().__init__(max_n_cores)
        
    def compute_allotments(self, ensembling_predictions : EnsemblingPredictions, scheduler : ListScheduler, config_ids : list):
        frequencies = ensembling_predictions.results["frequency"].unique()
        
        allotments = []
        for freq in frequencies:
            allotments.append({c_id : (self.max_n_cores, freq) for c_id in config_ids})
        
        return allotments

    
class GreedySpeedupAllotment(AllotmentStrategy):
    def __init__(self, max_n_cores : int, frequency_strategy : str = "min_energy", n_select : int = 5):
        self.frequency_strategy = frequency_strategy.lower()
        self.n_select = n_select
        if self.frequency_strategy not in ["min_energy", "min_time", "ranomd"]:
            raise ValueError(f"Unknown frequency strategy {self.frequency_strategy}. Must be one of ['min_energy', 'min_time']")
        super().__init__(max_n_cores)
        
        
    def _select_frequency(self, config_id : int, ensembling_predictions : EnsemblingPredictions, n_cores : int):
        resources = ensembling_predictions.results[
            (ensembling_predictions.results["config_id"] == config_id) &
            (ensembling_predictions.results["n_threads"] == n_cores)
        ]["frequency", "inference_time", "energy_consumption"]
        
        if self.frequency_strategy == "random":
            return np.random.choice(resources["frequency"].unique())
        
        avg_grouped = resources.groupby("frequency", as_index=False).mean()
        if self.frequency_strategy == "min_energy":
            return avg_grouped.loc[avg_grouped['energy_consumption'].idxmin(), 'frequency']
        elif self.frequency_strategy == "min_time":
            return avg_grouped.loc[avg_grouped['inference_time'].idxmin(), 'frequency']
        
     
    def _best_speedup_id(self, ensembling_predictions : EnsemblingPredictions, allotment : dict):
        relative_speedups = {}
        for config_id in ensembling_predictions.get_config_ids():
            n_cores, frequency = allotment[config_id]
            inf_time_old = ensembling_predictions.get_resources(config_id, n_cores, frequency, "inference_time")[0]
            inf_time_new = ensembling_predictions.get_resources(config_id, n_cores + 1, self._select_frequency(config_id, ensembling_predictions, n_cores), "inference_time")[0]
            relative_speedups[config_id] = (inf_time_old - inf_time_new) / inf_time_old
                    
        return max(relative_speedups, key=relative_speedups.get)
        
 
    def compute_allotments(self, ensembling_predictions : EnsemblingPredictions, scheduler : ListScheduler, config_ids : list):
        frequencies = ensembling_predictions.results["frequency"].unique()
        
        cores = [1] * len(config_ids)
        frequencies = [self._select_frequency(c_id, 1, frequencies) for c_id in config_ids]
        allotments = [{c_id: (cores[i], frequencies[i]) for i, c_id in enumerate(config_ids)}]
        inf_times, energies = zip(*[self.compute_resources(ensembling_predictions, scheduler, allotments[-1], config_ids)])
        
        while max(cores) < self.max_n_cores:
            best_speedup_config_id = self._best_speedup_id(ensembling_predictions, allotments[-1])
            best_speedup_index = config_ids.index(best_speedup_config_id)
            cores[best_speedup_index] += 1
            frequencies[best_speedup_index] = self._select_frequency(
                best_speedup_config_id, ensembling_predictions, cores[best_speedup_index]
            )
            
            allotments.append({c_id: (cores[i], frequencies[i]) for i, c_id in enumerate(config_ids)})
            inf_time, energy = self.compute_resources(ensembling_predictions, scheduler, allotments[-1], config_ids)
            inf_times.append(inf_time)
            energies.append(energy)
        
        min_indices = np.argsort(inf_times)[self.n_select:]
        
        return allotments[min_indices]




class GreedyWorkloadAllotment(AllotmentStrategy):
    def __init__(self, max_n_cores, frequency_strategy = "min_energy", n_select = 5):
        self.frequency_strategy = frequency_strategy.lower()
        self.n_select = n_select
        if self.frequency_strategy not in ["min_energy", "min_time", "random"]:
            raise ValueError(f"Unknown frequency strategy {self.frequency_strategy}. Must be one of ['min_energy', 'min_time', 'random']")
        super().__init__(max_n_cores)
        
        
    def _select_frequency(self, config_id, ensembling_predictions, n_cores):
        resources = ensembling_predictions.results[
            (ensembling_predictions.results["config_id"] == config_id) &
            (ensembling_predictions.results["n_threads"] == n_cores)
        ]["frequency", "inference_time", "energy_consumption"]
        
        if self.frequency_strategy == "random":
            return np.random.choice(resources["frequency"].unique())
        
        avg_grouped = resources.groupby("frequency", as_index=False).mean()
        if self.frequency_strategy == "min_energy":
            return avg_grouped.loc[avg_grouped['energy_consumption'].idxmin(), 'frequency']
        elif self.frequency_strategy == "min_time":
            return avg_grouped.loc[avg_grouped['inference_time'].idxmin(), 'frequency']
    
    
    def _max_inf_time_id(self, ensembling_predictions, allotment, config_ids):
        inf_times = {}
        for config_id in config_ids:
            n_cores, frequency = allotment[config_id]
            inf_times[config_id] = ensembling_predictions.get_resources(config_id, n_cores, frequency, "inference_time")[0]
        
        return max(inf_times, key=inf_times.get) 
    
    
    def _find_lowest_workload(self, ensembling_predictions, config_id, min_n_cores):
        workloads = []
        for n_cores in range(min_n_cores, self.max_n_cores+1):
            workloads.append(
                ensembling_predictions.get_resources(
                    config_id, n_cores, self._select_frequency(config_id, ensembling_predictions, n_cores), "inference_time"
                )[0]
            )
        
        return np.argmin(workloads) + min_n_cores
        
 
    def compute_allotments(self, ensembling_predictions, scheduler, config_ids):
        frequencies = ensembling_predictions.results["frequency"].unique()
        
        cores = [1] * len(config_ids)
        frequencies = [self._select_frequency(c_id, 1, frequencies) for c_id in config_ids]
        allotments = [{c_id: (cores[i], frequencies[i]) for i, c_id in enumerate(config_ids)}]
        inf_times, energies = zip(*[self.compute_resources(ensembling_predictions, scheduler, allotments[-1], config_ids)])
        
        while max(cores) < self.max_n_cores:
            max_inf_time_config_id = self._max_inf_time_id(ensembling_predictions, allotments[-1])
            max_inf_time_index = config_ids.index(max_inf_time_config_id)
            cores[max_inf_time_index] = self._find_lowest_workload(
                ensembling_predictions, max_inf_time_config_id, cores[max_inf_time_index]+1
            )
            frequencies[max_inf_time_index] = self._select_frequency(
                max_inf_time_config_id, ensembling_predictions, cores[max_inf_time_index]
            )
            
            allotments.append({c_id: (cores[i], frequencies[i]) for i, c_id in enumerate(config_ids)})
            inf_time, energy = self.compute_resources(ensembling_predictions, scheduler, allotments[-1], config_ids)
            inf_times.append(inf_time)
            energies.append(energy)
        
        min_indices = np.argsort(inf_times)[self.n_select:]
        
        return allotments[min_indices]
    



class NSGAIISchedulingProblem(Problem):
    def __init__(self, ensembling_predictions, scheduler, nsga2_allotment, max_n_cores, config_ids):
        self.ensembling_predictions = ensembling_predictions
        self.scheduler = scheduler
        self.config_ids = config_ids
        self.max_n_cores = max_n_cores
        self.nsga2_allotment = nsga2_allotment

        min_freq, max_freq = (
            self.ensembling_predictions.results["frequency"].astype(int).min(),
            self.ensembling_predictions.results["frequency"].astype(int).max()
        )

        xl = np.array([min_freq] * len(self.config_ids) + [1] * len(self.config_ids))
        xu = np.array([max_freq] * len(self.config_ids) + [self.max_n_cores] * len(self.config_ids))

        super().__init__(
            n_var = 2 * len(self.config_ids),
            n_obj = 2,
            xl=xl,
            xu=xu
        )
        
    @staticmethod
    def arr_to_allotment(x, config_ids):
        if not isinstance(x, np.ndarray):
            x = x.X

        n = len(config_ids)
        start_vals = x[:n]
        end_vals = x[n:2*n]

        return dict(zip(config_ids, zip(end_vals, start_vals)))

        
    def _evaluate(self, x, out, *args, **kwargs):
        out_arr = np.zeros((len(x), 2))
        for i, vec in enumerate(x):
            allotment = NSGAIISchedulingProblem.arr_to_allotment(vec, self.config_ids)
            inf_time, energy = self.nsga2_allotment.compute_resources(self.ensembling_predictions, self.scheduler, allotment, self.config_ids)
            out_arr[i] = [inf_time, energy]
        
        out["F"] = out_arr
    
        
class NSGA2SchedulingSampling(Sampling):
    def __init__(self):
        super().__init__()
   
    def _do(self, problem, n_samples, **kwargs):
        frequencies = problem.ensembling_predictions.results["frequency"].astype(int).unique()
        
        cores = np.ones((n_samples, problem.n_var // 2), dtype=int)
        frequencies = np.random.choice(frequencies, (n_samples, problem.n_var // 2))
        result = np.concatenate((frequencies, cores), axis=1)

        return result
    
    
class NSGA2SchedulingCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)
        
    def _do(self, problem, X, **kwargs): 
        _, n_matings, n_var = X.shape
        rand = np.random.rand(n_matings, n_var) < 0.5
        crossover = np.where(rand, X[0], X[1])
        return crossover[np.newaxis, ...]
    
    
class NSGA2SchedulingMutation(Mutation):
    def __init__(self, mut_prob=None):
        super().__init__()
        self.mut_prob = mut_prob
        

    def _do(self, problem, X, **kwargs):
        n_pop, n_var = X.shape
        n_configs = len(problem.config_ids)
        prob = self.mut_prob or (1.0 / n_var)

        X_mut = X.copy()

        freq_choices = problem.ensembling_predictions.results["frequency"].unique()
        freq_mask = np.random.rand(n_pop, n_configs) < prob

        if freq_mask.any():
            random_freqs = np.random.choice(freq_choices, size=freq_mask.sum())
            X_mut[:, :n_configs][freq_mask] = random_freqs

        core_mask = np.random.rand(n_pop, n_configs) < prob
        if core_mask.any():
            cur_cores = X[:, n_configs:][core_mask]
            core_add = np.random.choice([-1, 1], size=core_mask.sum())
            X_mut[:, n_configs:][core_mask] = np.clip(cur_cores + core_add, 1, problem.max_n_cores)
        
        return X_mut
            

class NSGA2Allotment(AllotmentStrategy):
    def __init__(self, max_n_cores, n_iterations = 15):
        super().__init__(max_n_cores)
        self.n_iterations = n_iterations

    def compute_allotments(self, ensembling_predictions, scheduler, config_ids):
        problem = NSGAIISchedulingProblem(ensembling_predictions, scheduler, self, self.max_n_cores, config_ids)
        
        algorithm = NSGA2(
            pop_size=10,
            sampling=NSGA2SchedulingSampling(),
            crossover=NSGA2SchedulingCrossover(),
            mutation=NSGA2SchedulingMutation(),
            eliminate_duplicates=True
        )
        
        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", self.n_iterations)
        )
        
        allotments = [NSGAIISchedulingProblem.arr_to_allotment(x, problem.config_ids) for x in res.opt]
        
        return allotments