import argparse
from json import load
import logging
import os
import numpy as np
from time import time
from contextlib import contextmanager
import yaml
import traceback
import tqdm

from raaml.ensembling.scheduling.allotment import NSGA2Allotment, FullFrequencyScalingAllotment, MaxNThreadsAllotment
from raaml.ensembling.scheduling.scheduler import LongestProcessingTimeScheduler, HighestWorkloadScheduler, HighestThreadCountScheduler
from raaml.raaml import ResourceAwareAutoMLPipeline
from raaml.util.idle_powers import get_idle_power
from raaml.util.openml_data import load_openml_data_new, zero_based_index_to_amlb_index, get_amlb_meta, load_amlb_tasks
from raaml.util.progress_summary import ProgressSummary
from raaml.ensembling.filter import NThreadsFilter, NonDominatedSortingFilter, DiscoveryTimeFilter, FrequencyScalingFilter, TopNFilter, SiloTopNFilter
from raaml.util.ensemble_pool_eval import evaluate_multiple_ensemble_pools, evaluate_ensemble_pool, evaluate_ensemble_pool_ens_preds

from experiments.run_ensembling import get_experiment_id, get_experiments, get_experiments_both_allotment, get_experiment_id_both_allotment
    
    
def collect_ensembling_predictions(task_id, bmg_method, pipeline_dir, result_dir, seed, exp_type="base"):
    pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
        directory=os.path.join(pipeline_dir, str(task_id)),
        identifier=f"{bmg_method}_{seed}",
    )

    assert exp_type == "base", "collect_ensembling_predictions only supports 'base' experiment type currently"

    X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)
    
    pipeline.load_base_model_generation(X_train, y_train, meta)
    ensembling_predictions = pipeline.load_ensembling_predictions()
    if ensembling_predictions is not None:
        ensembling_predictions = ensembling_predictions.copy()
    pipeline.load_final_predictions()
    y_ens = pipeline.data_split.get_ensembling_labels()
    
    n_models_ensembling = len(pipeline.ensembling_predictions.get_config_ids()) if pipeline.ensembling_predictions is not None else 0
    
    time_values = np.arange(3600/4, 3600*8, 3600/4)

    for time in list(time_values) + [-1]:
        if time == -1:
            name = f"base_boost_single_{time}"
            ensemble_pool = pipeline.load_ensemble_pool(name)
            res = evaluate_ensemble_pool_ens_preds(ensemble_pool, y_ens)
            res_file = os.path.join(result_dir, f"{name}_results_ens.csv")
            res.to_csv(res_file, index=False)
        
        name = f"base_28_single_{time}"
        ensemble_pool = pipeline.load_ensemble_pool(name)
        res = evaluate_ensemble_pool_ens_preds(ensemble_pool, y_ens)
        res_file = os.path.join(result_dir, f"{name}_results_ens.csv")
        res.to_csv(res_file, index=False)
            
            
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
            ensemble_pool = pipeline.load_ensemble_pool(name)
            res = evaluate_ensemble_pool_ens_preds(ensemble_pool, y_ens)
            res_file = os.path.join(result_dir, f"{name}_results_ens.csv")
            res.to_csv(res_file, index=False)

            name = f"base_28_{method}_{pruning_strategy}"
            ensemble_pool = pipeline.load_ensemble_pool(name)
            res = evaluate_ensemble_pool_ens_preds(ensemble_pool, y_ens)
            res_file = os.path.join(result_dir, f"{name}_results_ens.csv")
            res.to_csv(res_file, index=False)





    
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

    args = parser.parse_args()
    
    exp_id = int(args.exp_id)
    
    assert exp_id >= 0 and exp_id < 547, "Experiment ID must be between 0 and 546"
    
    experiments = get_experiments()
    task_id, seeds, methods = experiments[exp_id]
    
    logging.info(f"Running ensembling for OpenML task ID {task_id} with seeds {seeds} and methods {methods}")

    err_dir = os.path.join(args.result_output_dir, "errors_collect")
    os.makedirs(err_dir, exist_ok=True)

    n_total = len(seeds) * len(methods)
    with tqdm.tqdm(total=n_total, desc=f"Collecting ensembling predictions for task {task_id}") as pbar:
        for seed in seeds:
            for method in methods:
                try:
                    collect_ensembling_predictions(
                        task_id=task_id,
                        bmg_method=method,
                        pipeline_dir=args.output_dir,
                        result_dir=os.path.join(args.result_output_dir, args.exp_type, str(task_id), str(seed), method),
                        seed=seed,
                        exp_type=args.exp_type,
                    )
                except Exception as e:
                    err_file = os.path.join(err_dir, f"error_task_{task_id}_seed_{seed}_method_{method}.txt")
                    with open(err_file, "w") as f:
                        f.write(f"Error collecting ensembling predictions for task {task_id}, seed {seed}, method {method} (exp_id: {exp_id}):\n")
                        f.write(traceback.format_exc())
                    logging.error(f"Error collecting ensembling predictions for task {task_id}, seed {seed}, method {method}. See {err_file} for details.")
                pbar.update(1)
    
    
    