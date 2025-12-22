from raaml.raaml import ResourceAwareAutoMLPipeline
from raaml.util.openml_data import load_openml_data_new, zero_based_index_to_amlb_index
from raaml.util.split import split_indices
from raaml.util.raaml_progress import RAAMLProgressSummary

import argparse
import logging
import gc
import os

def run_raaml(task_id, zb_task_id, bmg_method, output_dir, seed, n_threads, frequency_mode, debug, hpo=False):
    
    assert n_threads == len(os.sched_getaffinity(0)), "Number of available cores must be equal to the number of threads for prediction!"

    if hpo:
        assert bmg_method == "random", "For hpo experiments, only 'random' bmg_method is supported"
        
        progress = RAAMLProgressSummary(f"{task_id}_{bmg_method}", "bmg_progress_stacking")
        
        seed = 1

        X_train, _, y_train, _, meta = load_openml_data_new(task_id)
        train_indices, test_indices = split_indices(len(X_train), train_size=0.8, stratify=y_train if meta["task_type"] == "classification" else None, seed=seed)
        
        X_test, y_test = X_train.iloc[test_indices], y_train.iloc[test_indices]
        gc.collect()
        
        X_train, y_train = X_train.iloc[train_indices], y_train.iloc[train_indices]
        gc.collect()
        
        pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
            directory=os.path.join(output_dir, str(task_id)+"_stacking"),
            identifier=f"{bmg_method}_{seed}",
        )
        
        pipeline.set_raaml_progress(progress)
        
        pipeline.load_base_model_generation(X_train, y_train, meta)
        
    else:
        progress = RAAMLProgressSummary(f"{task_id}_{bmg_method}_{seed}", "bmg_progress_gbmlp" if not debug else "test")
        
        X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)
        
        pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
            directory=os.path.join(output_dir, (str(task_id)+"_debug") if debug else str(task_id)),
            identifier=f"{bmg_method}_{seed}",
        )
        
        pipeline.set_raaml_progress(progress)
        
        pipeline.load_base_model_generation(X_train, y_train, meta)
    

    try:
        pipeline.load_ensembling_predictions()
    except:
        logging.info("No ensembling predictions to load.")
    
    pipeline.create_ensembling_predictions(n_threads, frequency_mode, delete_after_pred=False)
    
    try:
        pipeline.load_final_predictions()
    except:
        logging.info("No final predictions to load.")
            
    pipeline.create_final_predictions(X_test, y_test, n_threads, frequency_mode)
    
    
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
        "--output_dir",
        type=str,
        default="raaml_output_gbmlp",
        help="Output directory for the AutoML pipeline. Default is 'raaml_output_gbmlp'."
    )
    parser.add_argument(
        "--seed", 
        type=int,
        required=True,
        help="Seed for random number generator. Default is 0."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Default is False."
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        required=True,
        help="Number of threads to use."
    )
    parser.add_argument(
        "--frequency_mode",
        type=str,
        choices=["boost", "scaling"],
        default="boost",
        help="Frequency mode to use for predictions. Default is 'boost'."
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Generate predictions for HPO runs. Default is False."
    )
    parser.add_argument(
        "--n_cpu_filter",
        type=int,
        choices=[1,2,4],
        default=None        
    )
    
    args = parser.parse_args()
    
    openml_index = zero_based_index_to_amlb_index(args.amlb_task_id_zero_based, n_cpu_cores=args.n_cpu_filter)
    logging.info(f"Running RAAML pipeline (step 2) for OpenML task ID {openml_index} (zb: {args.amlb_task_id_zero_based})")
    
    assert len(os.sched_getaffinity(0)) == args.n_threads, "Number of available cores must be equal to the number of threads!"

    run_raaml(openml_index, args.amlb_task_id_zero_based, args.bmg_method, args.output_dir, args.seed, args.n_threads, args.frequency_mode, args.debug, args.hpo)
