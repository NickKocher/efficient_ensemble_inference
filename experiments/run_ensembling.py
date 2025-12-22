import argparse
from json import load
import logging
import os
import numpy as np
from time import time
from contextlib import contextmanager
import yaml
import traceback

from raaml.ensembling.scheduling.allotment import NSGA2Allotment, FullFrequencyScalingAllotment, MaxNThreadsAllotment
from raaml.ensembling.scheduling.scheduler import LongestProcessingTimeScheduler, HighestWorkloadScheduler, HighestThreadCountScheduler
from raaml.raaml import ResourceAwareAutoMLPipeline
from raaml.util.idle_powers import get_idle_power
from raaml.util.openml_data import load_openml_data_new, zero_based_index_to_amlb_index, get_amlb_meta, load_amlb_tasks
from raaml.util.progress_summary import ProgressSummary
from raaml.ensembling.filter import NThreadsFilter, NonDominatedSortingFilter, DiscoveryTimeFilter, FrequencyScalingFilter, TopNFilter, SiloTopNFilter
from raaml.util.ensemble_pool_eval import evaluate_multiple_ensemble_pools, evaluate_ensemble_pool



@contextmanager
def timeit(meta, name):
    start = time()
    yield
    end = time()
    
    if name not in meta:
        meta[name] = {}
        
    meta[name]["runtime"] = end - start
    
def get_experiment_id(task_id, seed, method):
    exps = get_experiments()
    for i, (t, s, m) in enumerate(exps):
        if t == task_id and seed in s and method in m:
            return i
    raise ValueError("Experiment not found")

    
def get_experiments():
    meta_df = get_amlb_meta()
    
    tasks_low = sorted(meta_df[meta_df["n_samples"] < 50000]["task_id"].to_list())
    tasks_medium = sorted(meta_df[(meta_df["n_samples"] >= 50000) & (meta_df["n_samples"] < 500000)]["task_id"].to_list())
    tasks_high = sorted(meta_df[meta_df["n_samples"] >= 500000]["task_id"].to_list())
    
    tasks_low = [t for t in tasks_low if not t in [189355, 190412, 10090]]
    tasks_medium = [t for t in tasks_medium if not t in [189355, 190412, 10090]]
    tasks_high = [t for t in tasks_high] + [189355, 190412, 10090] 
    
    experiments = []
    seeds = list(range(1,6))
    methods = ["random", "so-smac", "mo-smac", "nsga2", "so-asha", "mo-asha"]
    
    for t in tasks_low:
        experiments.append((t, seeds, methods))
    
    for t in tasks_medium:
        for method in methods:
            experiments.append((t, seeds, [method]))
    
    for t in tasks_high:
        for method in methods:
            for seed in seeds:
                experiments.append((t, [seed], [method]))
       
    ### ==> 547 experiments 
    
    return experiments


def get_experiments_both_allotment():
    tasks = [t[1] for t in load_amlb_tasks()]
    tasks_high = [167210, 359930, 359931, 359934, 359950, 359951, 2073, 146818, 168757, 168784, 190146, 359954, 359955, 359956, 359958, 359959]
    tasks_medium = [10090, 233214, 359932, 359933, 359935, 359944, 359945, 359948, 360945, 168350, 168911, 190137, 190392, 190411, 359953, 359957, 359961, 359962, 359963, 359964, 359966, 359968, 359974, 359980]
    tasks_low = list(set(tasks).difference(set(tasks_high)).difference(set(tasks_medium)))
    
    experiments = []
    seeds = list(range(1,6))
    methods = ["random", "so-smac", "mo-smac", "nsga2", "so-asha", "mo-asha"]
    
    for t in tasks_low:
        experiments.append((t, seeds, methods))
    
    for t in tasks_medium:
        for method in methods:
            experiments.append((t, seeds, [method]))
    
    for t in tasks_high:
        for method in methods:
            for seed in seeds:
                experiments.append((t, [seed], [method]))
                
    ### ==> 688 experiments
    return experiments


def get_experiment_id_both_allotment(task_id, seed, method):
    exps = get_experiments_both_allotment()
    for i, (t, s, m) in enumerate(exps):
        if t == task_id and seed in s and method in m:
            return i
    raise ValueError("Experiment not found")
        
    
    
    
def extract_meta_ensembling(pipeline, meta, name):
    n_setups, n_models = 0, 0
    sum_pred_time, sum_pred_energy = 0, 0
    if pipeline.ensembling_predictions is not None:
        n_setups = len(pipeline.ensembling_predictions.results) // 5 
        n_models = int(pipeline.ensembling_predictions.results["config_id"].nunique())
        sum_pred_time = float(pipeline.ensembling_predictions.results["inference_time"].sum())
        sum_pred_energy = float(pipeline.ensembling_predictions.results["energy_consumption"].sum())
    
    meta[name]["empty_ep"] = pipeline.ensemble_pool is None
    meta[name]["sum_pred_time"] = sum_pred_time
    meta[name]["sum_pred_energy"] = sum_pred_energy
    meta[name]["n_setups"] = n_setups
    meta[name]["n_models"] = n_models
    
    
def extract_meta_allotment(ensemble_pool, meta, name):    
    meta[name]["empty_ep"] = ensemble_pool is None or len(ensemble_pool) == 0


def ensembling_exp_exists(name, result_dir, results_meta):
    cv_exists =  os.path.exists(os.path.join(result_dir, f"{name}.csv"))
    meta_complete = (name in results_meta) and (len(results_meta[name]) == 6)
        
    return cv_exists and meta_complete
    
def allotment_exp_exists(name, result_dir, results_meta):
    cv_exists =  os.path.exists(os.path.join(result_dir, f"{name}.csv"))
    meta_complete = (name in results_meta) and (len(results_meta[name]) == 2)
    
    return cv_exists and meta_complete
    

def get_progress(result_dir, n_total):
    if not os.path.exists(result_dir):
        return 0
    else:
        files = [f for f in os.listdir(result_dir) if f.endswith(".csv")]
        return len(files) / n_total
    
    
def report_error(error_dir, name, traceback, method, seed, task_id):
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)
    with open(os.path.join(error_dir, f"{method}_{seed}_{task_id}_{name}.txt"), "w") as f:
        f.write(traceback)
    print(f"Error while running {name}: {traceback}")
        
       
def run_ensembling(
    pipeline, 
    result_dir, 
    name, 
    results_meta, 
    filters, 
    X_test, 
    y_test, 
    overwrite, 
    method, 
    err_dir, 
    bmg_method, 
    seed, 
    task_id,
):
    try:
        if overwrite or not ensembling_exp_exists(name, result_dir, results_meta):
            with timeit(results_meta, name):
                pipeline.run_ensembling(method, name, filters=filters, do_allotment=False)
            results = evaluate_ensemble_pool(pipeline.ensemble_pool, X_test, y_test)
            extract_meta_ensembling(pipeline, results_meta, name)
            results.to_csv(os.path.join(result_dir, f"{name}.csv"))
        
            with open(os.path.join(result_dir, f"meta_results.yaml"), "w") as f:
                yaml.safe_dump(results_meta, f)
        else:
            logging.info(f"Skipping {name} because it already exists ({os.path.join(result_dir, f'{name}.csv')}).")
    except Exception as e:
        err_traceback = f"Error while running {name}: {e}\n{traceback.format_exc()}"
        report_error(err_dir, name, err_traceback, bmg_method, seed, task_id)
        
        

def run_allotment_experiment(
    allotment, pipeline, X_test, y_test,
    result_dir, err_dir, bmg_method, name, ep_name, ensembling_predictions, results_meta,
    scheduler=None, frequency_scaling=True, parallelism=False, overwrite=False, max_n_cores=8
):
    try:
        if overwrite or not allotment_exp_exists(name, result_dir, results_meta):
            if frequency_scaling and not parallelism:
                scheduler = LongestProcessingTimeScheduler(1, "energy", get_idle_power(False))
                
                ensemble_pool = pipeline.load_ensemble_pool(ep_name)
                
                if ensemble_pool is not None:
                    ensemble_pool = ensemble_pool.copy()
                    ensemble_pool.ensembling_predictions = ensembling_predictions.copy()
                    with timeit(results_meta, name):
                        pool_extended, old_ensemble_indices = ensemble_pool.extend_by_allotment(
                            NSGA2Allotment(1) if allotment == "nsga2" else FullFrequencyScalingAllotment(1),
                            scheduler,
                            [
                                FrequencyScalingFilter("fs"),
                                NThreadsFilter(max_n_threads=1)
                            ]
                        )
                else:
                    with timeit(results_meta, name):
                        pool_extended = None
                results = evaluate_ensemble_pool(pool_extended, X_test, y_test)
                if ensemble_pool is not None:
                    results["old_ensemble_index"] = old_ensemble_indices
                results.to_csv(os.path.join(result_dir, f"{name}.csv"))
                extract_meta_allotment(pool_extended, results_meta, name)
                with open(os.path.join(result_dir, f"meta_results.yaml"), "w") as f:
                    yaml.safe_dump(results_meta, f)
            elif not frequency_scaling and parallelism:
                if scheduler is None:
                    raise ValueError("Scheduler must be provided for parallelism experiments")
                
                scheduler_map = {
                    "lpt": LongestProcessingTimeScheduler,
                    "hws": HighestWorkloadScheduler,
                    "htcs": HighestThreadCountScheduler
                }

                scheduler = scheduler_map[scheduler](max_n_cores, "energy", get_idle_power(False))

                ensemble_pool = pipeline.load_ensemble_pool(ep_name)
                
                if ensemble_pool is not None:
                    ensemble_pool = ensemble_pool.copy()
                    ensemble_pool.ensembling_predictions = ensembling_predictions.copy()
                    with timeit(results_meta, name):
                        pool_extended, old_ensemble_indices = ensemble_pool.extend_by_allotment(
                            NSGA2Allotment(max_n_cores=max_n_cores) if allotment == "nsga2" else MaxNThreadsAllotment(max_n_cores=max_n_cores),
                            scheduler,
                            [
                                FrequencyScalingFilter("max_fs"),
                                NThreadsFilter(max_n_threads=max_n_cores),
                            ]
                        )
                else:
                    with timeit(results_meta, name):
                        pool_extended = None
                results = evaluate_ensemble_pool(pool_extended, X_test, y_test)
                if ensemble_pool is not None:
                    results["old_ensemble_index"] = old_ensemble_indices
                results.to_csv(os.path.join(result_dir, f"{name}.csv"))
                extract_meta_allotment(pool_extended, results_meta, name)
                with open(os.path.join(result_dir, f"meta_results.yaml"), "w") as f:
                    yaml.safe_dump(results_meta, f)
            elif frequency_scaling and parallelism:
                if scheduler is None:
                    raise ValueError("Scheduler must be provided for parallelism experiments")
                if allotment != "nsga2":
                    raise ValueError("Only NSGA2 allotment is supported for frequency scaling + parallelism experiments")
                
                scheduler_map = {
                    "lpt": LongestProcessingTimeScheduler,
                    "hws": HighestWorkloadScheduler,
                    "htcs": HighestThreadCountScheduler
                }

                scheduler = scheduler_map[scheduler](max_n_cores, "energy", get_idle_power(False))

                ensemble_pool = pipeline.load_ensemble_pool(ep_name)
                
                if ensemble_pool is not None:
                    ensemble_pool = ensemble_pool.copy()
                    ensemble_pool.ensembling_predictions = ensembling_predictions.copy()
                    with timeit(results_meta, name):
                        pool_extended, old_ensemble_indices = ensemble_pool.extend_by_allotment(
                            NSGA2Allotment(max_n_cores=max_n_cores, n_iterations=50),
                            scheduler,
                            [
                                FrequencyScalingFilter("fs"),
                                NThreadsFilter(max_n_threads=max_n_cores),
                            ]
                        )
                else:
                    with timeit(results_meta, name):
                        pool_extended = None
                results = evaluate_ensemble_pool(pool_extended, X_test, y_test)
                if ensemble_pool is not None:
                    results["old_ensemble_index"] = old_ensemble_indices
                results.to_csv(os.path.join(result_dir, f"{name}.csv"))
                extract_meta_allotment(pool_extended, results_meta, name)
                with open(os.path.join(result_dir, f"meta_results.yaml"), "w") as f:
                    yaml.safe_dump(results_meta, f)
        else:
            logging.info(f"Skipping allotment experiment {name} because it already exists ({os.path.join(result_dir, f'{name}.csv')}).")
    except Exception as e:
        err_traceback = f"Error while running  allotment experiment {name}: {e}\n{traceback.format_exc()}"
        report_error(err_dir, name, err_traceback, bmg_method, seed, task_id)
            




def run_ensembling_experiments(task_id, bmg_method, pipeline_dir, result_dir, seed, exp_type="base", overwrite=False, time_restriction=None):
            
    err_dir = os.path.join(result_dir, exp_type, "errors")
    if not os.path.exists(err_dir):
        os.makedirs(err_dir)
    result_dir = os.path.join(result_dir, exp_type, str(task_id), str(seed), bmg_method)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir) 
        

    X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)
    
    results_meta = {}
    if os.path.exists(os.path.join(result_dir, "meta_results.yaml")):
        with open(os.path.join(result_dir, "meta_results.yaml"), "r") as f:
            results_meta = yaml.load(f, Loader=yaml.FullLoader)
    
    if exp_type == "base":
        time_values = np.arange(3600/4, 3600*8, 3600/4)
        
        summary = ProgressSummary("ensembling")
        
        pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
            directory=os.path.join(pipeline_dir, str(task_id)),
            identifier=f"{bmg_method}_{seed}",
        )
        
        pipeline.load_base_model_generation(X_train, y_train, meta)
        pipeline.load_ensembling_predictions()
        pipeline.load_final_predictions()
        
        n_models_ensembling = len(pipeline.ensembling_predictions.get_config_ids()) if pipeline.ensembling_predictions is not None else 0
        n_total = (len(time_values) + 2) + 2 * 6 * (7 if n_models_ensembling > 50 else 4)
        
        filters_boost = [
            NThreadsFilter(max_n_threads=1),
            FrequencyScalingFilter("boost")
        ]
        filters_maxfs = [
            NThreadsFilter(max_n_threads=1),
            FrequencyScalingFilter("max_fs")
        ]

        for time in list(time_values) + [-1]:
            if time == -1:
                name = f"base_boost_single_{time}"
                run_ensembling(
                    pipeline, result_dir, name, results_meta,
                    [*filters_boost, DiscoveryTimeFilter(max_discovery_time=time)],
                    X_test, y_test, overwrite, "single", 
                    err_dir, bmg_method, seed, task_id
                )            
                summary.set_progress(f"{seed}_{task_id}_{bmg_method}", get_progress(result_dir, n_total), print_progress=True)
            
            name = f"base_28_single_{time}"
            run_ensembling(
                pipeline, result_dir, name, results_meta,
                [*filters_maxfs, DiscoveryTimeFilter(max_discovery_time=time)],
                X_test, y_test, overwrite, "single", 
                err_dir, bmg_method, seed, task_id
            )  
            summary.set_progress(f"{seed}_{task_id}_{bmg_method}", get_progress(result_dir, n_total), print_progress=True)
            
            
        for method in ["ges", "mo-ges","qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]:   
            pruning_strategies = {
                "top_50" : TopNFilter(n=50),
                "nd_first_pareto": NonDominatedSortingFilter(max_pareto_rank=1),
                "nd_second_pareto": NonDominatedSortingFilter(max_pareto_rank=2),
                "nd_third_pareto": NonDominatedSortingFilter(max_pareto_rank=3),
            }
            
            if n_models_ensembling > 50:
                pruning_strategies.update({
                    "silo_top_50" : SiloTopNFilter(n=50),
                    "nd_50": NonDominatedSortingFilter(n_models=50),
                    "nd_50_silo": NonDominatedSortingFilter(n_models=50, silo=True),
                })
                     
            for pruning_strategy, pruning_filter in pruning_strategies.items():
                name = f"base_boost_{method}_{pruning_strategy}"
                run_ensembling(
                    pipeline, result_dir, name, results_meta, 
                    [*filters_boost, pruning_filter], 
                    X_test, y_test, overwrite, method,
                    err_dir, bmg_method, seed, task_id
                )
                summary.set_progress(f"{seed}_{task_id}_{bmg_method}", get_progress(result_dir, n_total), print_progress=True)


                name = f"base_28_{method}_{pruning_strategy}"
                run_ensembling(
                    pipeline, result_dir, name, results_meta,
                    [*filters_maxfs, pruning_filter],
                    X_test, y_test, overwrite, method,
                    err_dir, bmg_method, seed, task_id
                )
                summary.set_progress(f"{seed}_{task_id}_{bmg_method}", get_progress(result_dir, n_total), print_progress=True)
                    
       
    elif exp_type == "frequency_scaling":
        summary = ProgressSummary("fs_allotment")
        
        pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
            directory=os.path.join(pipeline_dir, str(task_id)),
            identifier=f"{bmg_method}_{seed}",
        )
        
        pipeline.load_base_model_generation(X_train, y_train, meta)
        ensembling_predictions = pipeline.load_ensembling_predictions()
        if ensembling_predictions is not None:
            ensembling_predictions = ensembling_predictions.copy()
        pipeline.load_final_predictions()
        
        n_models_ensembling = len(pipeline.ensembling_predictions.get_config_ids()) if pipeline.ensembling_predictions is not None else 0
        
        ensemble_pools = {
            "bmp": "base_28_single_-1",
            **({
                f"{method}|nds": f"base_28_{method}_nd_50_silo" for method in 
                ["ges", "mo-ges","qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]
            } if n_models_ensembling > 50 else {
                f"{method}|nds": f"base_28_{method}_top_50" for method in
                ["ges", "mo-ges","qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]
            })
        }   
        
        allotment_strategies = ["nsga2", "full"]
        
        n_total = len(allotment_strategies) * len(ensemble_pools) 
        
        for ep_short, ep_name in ensemble_pools.items():
            for allotment in allotment_strategies:
                run_allotment_experiment(
                    allotment, pipeline, X_test, y_test,
                    result_dir, err_dir, bmg_method, f"fs_allotment_{ep_short}_{allotment}",
                    ep_name, ensembling_predictions, results_meta,
                    frequency_scaling=True, parallelism=False
                )
                summary.set_progress(f"{seed}_{task_id}_{bmg_method}", get_progress(result_dir, n_total), print_progress=True)
                
    elif exp_type == "parallel":
        summary = ProgressSummary("parallel_allotment")
        
        pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
            directory=os.path.join(pipeline_dir, str(task_id)),
            identifier=f"{bmg_method}_{seed}",
        )
        
        pipeline.load_base_model_generation(X_train, y_train, meta)
        ensembling_predictions = pipeline.load_ensembling_predictions()
        if ensembling_predictions is not None:
            ensembling_predictions = ensembling_predictions.copy()
        pipeline.load_final_predictions()
        
        n_models_ensembling = len(pipeline.ensembling_predictions.get_config_ids()) if pipeline.ensembling_predictions is not None else 0
        
        ensemble_pools = {
            "bmp": "base_28_single_-1",
            **({
                f"{method}|nds": f"base_28_{method}_nd_50_silo" for method in 
                ["ges", "mo-ges","qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]
            } if n_models_ensembling > 50 else {
                f"{method}|nds": f"base_28_{method}_top_50" for method in
                ["ges", "mo-ges","qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]
            })
        }   
        
        allotment_sched_combinations = [("nsga2", "lpt"), ("nsga2", "hws"), ("nsga2", "htcs"), ("full", "lpt")]
        n_cores_list = [2,4,8]
        
        n_total = len(allotment_sched_combinations) * len(ensemble_pools) * len(n_cores_list)
        
        for ep_short, ep_name in ensemble_pools.items():
            for allotment, scheduler in allotment_sched_combinations:
                for max_n_cores in n_cores_list:
                    run_allotment_experiment(
                        allotment, pipeline, X_test, y_test,
                        result_dir, err_dir, bmg_method, f"parallel_allotment_{ep_short}_{allotment}_{scheduler}_{max_n_cores}",
                        ep_name, ensembling_predictions, results_meta,
                        scheduler=scheduler, frequency_scaling=False, parallelism=True, max_n_cores=max_n_cores
                    )
                    summary.set_progress(f"{seed}_{task_id}_{bmg_method}", get_progress(result_dir, n_total), print_progress=True)
                
    elif exp_type == "frequency_scaling+parallel":        
        summary = ProgressSummary("allotment_both")
        
        pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
            directory=os.path.join(pipeline_dir, str(task_id)),
            identifier=f"{bmg_method}_{seed}",
        )
        
        pipeline.load_base_model_generation(X_train, y_train, meta)
        ensembling_predictions = pipeline.load_ensembling_predictions()
        if ensembling_predictions is not None:
            ensembling_predictions = ensembling_predictions.copy()
        pipeline.load_final_predictions()
        
        n_models_ensembling = len(pipeline.ensembling_predictions.get_config_ids()) if pipeline.ensembling_predictions is not None else 0
        
        ensemble_pools = {
            "bmp": "base_28_single_-1",
            **({
                f"{method}|nds": f"base_28_{method}_nd_50_silo" for method in 
                ["ges", "mo-ges","qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]
            } if n_models_ensembling > 50 else {
                f"{method}|nds": f"base_28_{method}_top_50" for method in
                ["ges", "mo-ges","qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]
            })
        }   
        
        allotment_sched_combinations = [("nsga2", "lpt"), ("nsga2", "hws"), ("nsga2", "htcs")]
        n_cores_list = [2,4,8]
        
        n_total = len(allotment_sched_combinations) * len(ensemble_pools) * len(n_cores_list)
        
        for ep_short, ep_name in ensemble_pools.items():
            for allotment, scheduler in allotment_sched_combinations:
                for max_n_cores in n_cores_list:
                    run_allotment_experiment(
                        allotment, pipeline, X_test, y_test,
                        result_dir, err_dir, bmg_method, f"allotment_fs+p_{ep_short}_{allotment}_{scheduler}_{max_n_cores}",
                        ep_name, ensembling_predictions, results_meta,
                        scheduler=scheduler, frequency_scaling=True, parallelism=True, max_n_cores=max_n_cores
                    )
                    summary.set_progress(f"{seed}_{task_id}_{bmg_method}", get_progress(result_dir, n_total), print_progress=True)
    
    elif exp_type == "base_time":
        assert time_restriction is not None, "time_restriction must be provided for base_time experiments"
        time_restriction = int(time_restriction)
        
        summary = ProgressSummary(f"ensembling_{time_restriction}")
        
        pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
            directory=os.path.join(pipeline_dir, str(task_id)),
            identifier=f"{bmg_method}_{seed}",
        )
        
        pipeline.load_base_model_generation(X_train, y_train, meta)
        pipeline.load_ensembling_predictions()
        pipeline.load_final_predictions()
        
        n_models_ensembling = len(pipeline.ensembling_predictions.get_config_ids()) if pipeline.ensembling_predictions is not None else 0
        n_total = 6 * (7 if n_models_ensembling > 50 else 4)
        
        filters_maxfs = [
            DiscoveryTimeFilter(max_discovery_time=time_restriction),
            NThreadsFilter(max_n_threads=1),
            FrequencyScalingFilter("max_fs")
        ]
        
        for method in ["ges", "mo-ges","qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]:   
            pruning_strategies = {
                "top_50" : TopNFilter(n=50),
                "nd_first_pareto": NonDominatedSortingFilter(max_pareto_rank=1),
                "nd_second_pareto": NonDominatedSortingFilter(max_pareto_rank=2),
                "nd_third_pareto": NonDominatedSortingFilter(max_pareto_rank=3),
            }
            
            if n_models_ensembling > 50:
                pruning_strategies.update({
                    "silo_top_50" : SiloTopNFilter(n=50),
                    "nd_50": NonDominatedSortingFilter(n_models=50),
                    "nd_50_silo": NonDominatedSortingFilter(n_models=50, silo=True),
                })
                     
            for pruning_strategy, pruning_filter in pruning_strategies.items():                
                name = f"base_28_{method}_{pruning_strategy}_time_{time_restriction}"
                run_ensembling(
                    pipeline, result_dir, name, results_meta,
                    [*filters_maxfs, pruning_filter],
                    X_test, y_test, overwrite, method,
                    err_dir, bmg_method, seed, task_id
                )
                summary.set_progress(f"{seed}_{task_id}_{bmg_method}", get_progress(result_dir, n_total), print_progress=True)
    
    
    else:
        raise ValueError(f"Unknown type: {exp_type}")
                
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Resource Aware AutoML Pipeline")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raaml_output_gbmlp",
        help="Output directory for the AutoML pipeline. Default is 'raaml_output_gbmlp'."
    )
    parser.add_argument(
        "--result_output_dir",
        type=str,
        default="./ensemble_pool_results",
        help="Output directory for ensemble pool results."
    )
    parser.add_argument(
        "--exp_type",
        type=str,
        default="base",
        choices=["base", "base_time", "frequency_scaling", "parallel", "frequency_scaling+parallel"],
        help="Type of ensembling experiments to run. Default is 'base'."
    )    
    parser.add_argument(
        "--exp_id",
        type=str,
        required=True,
        help="Experiment ID."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing results."
    )
    parser.add_argument(
        "--time_restriction",
        type=int,
        default=None,
        help="Time restriction for the experiment in seconds."
    )

    args = parser.parse_args()
    
    exp_id = int(args.exp_id)
    
    assert len(os.sched_getaffinity(0)) == 1, "Number of available cores is supposed to be 1 for ensembling!"
    
    if args.exp_type == "frequency_scaling+parallel":
        assert exp_id >= 0 and exp_id < 688, "Experiment ID must be between 0 and 687"
        
        experiments = get_experiments_both_allotment()
        task_id, seeds, methods = experiments[exp_id]
    else:
        assert exp_id >= 0 and exp_id < 547, "Experiment ID must be between 0 and 546"
        
        experiments = get_experiments()
        task_id, seeds, methods = experiments[exp_id]
    
    logging.info(f"Running ensembling for OpenML task ID {task_id} with seeds {seeds} and methods {methods}")
    
    for seed in seeds:
        for method in methods:
            run_ensembling_experiments(
                task_id, method, args.output_dir, args.result_output_dir,
                seed, args.exp_type, args.overwrite, args.time_restriction
            )
    
    
    