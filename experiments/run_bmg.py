from raaml.raaml import ResourceAwareAutoMLPipeline
from raaml.util.openml_data import load_openml_data_new, zero_based_index_to_amlb_index, get_resources
from raaml.util.raaml_progress import RAAMLProgressSummary
from raaml.util.split import split_indices

import argparse
import logging
import os
import gc

def run_raaml(task_id, zb_task_id, bmg_method, n_cores_target_fun, time_h, output_dir, seed, debug, hpo=False):
    
    assert n_cores_target_fun == len(os.sched_getaffinity(0)), "Number of jobs must be equal to the number of available CPU cores"
    
    if hpo:
        assert bmg_method == "random", "For stacking experiments, only 'random' bmg_method is supported"
        
        output_dir = os.path.join(output_dir, str(task_id)+"_stacking")
        
        progress = RAAMLProgressSummary(f"{task_id}_{bmg_method}", "bmg_progress_stacking")
        
        seed = 1
        
        X_train, _, y_train, _, meta = load_openml_data_new(task_id)
        
        train_indices, _ = split_indices(len(X_train), train_size=0.8, stratify=y_train if meta["task_type"] == "classification" else None, seed=seed)
        X_train, y_train = X_train.iloc[train_indices], y_train.iloc[train_indices]
        gc.collect()
        
    else:
        output_dir = os.path.join(output_dir, (str(task_id)+"_debug") if debug else str(task_id))
        
        if seed is None:
            seed = find_next_seed(output_dir, f"{bmg_method}")
            logging.info("No seed set, using seed %d", seed)
        
        progress = RAAMLProgressSummary(f"{task_id}_{bmg_method}_{seed}", "bmg_progress_gbmlp" if not debug else "test")
        
        X_train, _, y_train, _, meta = load_openml_data_new(task_id)
        
    pipeline = ResourceAwareAutoMLPipeline(
        identifier=f"{bmg_method}_{seed}",
        output_dir=output_dir,
        base_model_gen=bmg_method,
        n_cores_target_function=n_cores_target_fun,
        time_search_s=time_h * 3600,
        trial_walltime_limit_s=40 if debug else None,
        trial_memory_limit_mb=16 * 1024 * n_cores_target_fun,
        logging_level=logging.INFO, # if not debug else logging.DEBUG,
        seed=seed,
        overwrite=True
    )
    
    pipeline.set_raaml_progress(progress)
    
    pipeline.run_base_model_generation(X_train, y_train, meta)
    pipeline.base_model_pool.print_base_model_results()
    pipeline.fit_models_for_ensembling_predictions()
    pipeline.refit()
    
    
def find_next_seed(output_dir, identifier):
    i = 1
    while os.path.exists(os.path.join(output_dir, f"{identifier}_{i}")):
        i += 1
    return i    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Resource Aware AutoML Pipeline")
    parser.add_argument(
        "--amlb_task_id_zero_based",
        type=int,
        default=0,
        help="AMLB task ID (zero-based) to use for the AutoML pipeline. Default is 0."
    )
    parser.add_argument(
        "--bmg_method",
        type=str,
        default="mo-smac",
        choices=["random", "so-smac", "mo-smac", "so-asha", "mo-asha", "nsga2"],
        help="Base model generation method to use. Default is 'mo-smac'."
    )
    parser.add_argument(
        "--n_cores_target_fun",
        type=int,
        default=1,
        help="Number of jobs to use for base model generation. Default is 1."
    )
    parser.add_argument(
        "--time_search_h",
        type=float,
        default=8,
        help="Time limit for base model generation in hours. Default is 8."
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=None,
        help="Seed for random number generator. Default is 0."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="raaml_output_gbmlp",
        help="Output directory for the pipeline. Default is './raaml_output_gbmlp'."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Run bmg to generate data for HPO experiments"
    )
    
    args = parser.parse_args()
    
    openml_index = zero_based_index_to_amlb_index(args.amlb_task_id_zero_based, n_cpu_cores=args.n_cores_target_fun)
    logging.info(f"Running RAAML pipeline for OpenML task ID {openml_index} (zb: {args.amlb_task_id_zero_based}) for a total of {int(args.time_search_h * 3600)} seconds.")
    
    mem, n_cpus = get_resources(openml_index)
    
    assert n_cpus == len(os.sched_getaffinity(0)), "Number of cpus assigned to this task must be equal to the number of available CPU cores"
    assert n_cpus == args.n_cores_target_fun, "Number of cores for target function must be equal to the number of available CPU cores"
    
    run_raaml(
        openml_index,
        args.amlb_task_id_zero_based,
        args.bmg_method,
        args.n_cores_target_fun,
        args.time_search_h,
        args.output_dir,
        args.seed,
        args.debug,
        args.hpo
    )
    
    
    