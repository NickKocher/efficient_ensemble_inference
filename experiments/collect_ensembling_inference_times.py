"""
Collects per-member (weight > 0) inference times for every ensemble in every pool.

Mirrors the loop structure of collect_ensembling_predictions.collect_ensembling_predictions
exactly (same pool names, same pruning-strategy selection, same method lists), but calls
extract_ensemble_inference_times instead of evaluate_ensemble_pool_ens_preds, and writes
"{name}_inference_times.csv" next to the existing "{name}_results_ens.csv" files so that
collect_ensembling_results.py can pick them both up from the same directory.

REQUIRES: extract_ensemble_inference_times added to raaml/util/ensemble_pool_eval.py:

    import json

    def extract_ensemble_inference_times(pool):
        if pool is None:
            return pd.DataFrame()

        resource_usage = np.zeros((pool.weight_matrix.shape[1], len(pool.assembled_resource_provider)))
        resource_names = pool.assembled_resource_provider.get_names()
        inf_time_index = resource_names.index("inference_time")

        for i, (config_id, n_cores, frequency) in enumerate(zip(pool.config_ids, pool.threads, pool.frequencies)):
            preds = pool.final_results.get_predictions(config_id)
            if preds is None:
                pool.weight_matrix[:, i] = 0.0
                continue
            objectives = pool.final_results.get_objectives(config_id, n_cores, frequency, resource_names)
            resource_usage[i] = objectives[resource_names]

        res = pd.DataFrame(columns=["inference_times"])
        for i in range(len(pool)):
            active = np.abs(pool.weight_matrix[i]) >= 1e-12  # same "in-use" threshold as ensemble_to_task_list
            inference_times = resource_usage[active, inf_time_index].tolist()
            res.loc[i] = [json.dumps(inference_times)]  # JSON string so it round-trips cleanly through CSV
        return res
"""
import argparse
import os
import traceback
import logging
import tqdm

from raaml.raaml import ResourceAwareAutoMLPipeline
from raaml.util.openml_data import load_openml_data_new, load_amlb_tasks
from raaml.util.ensemble_pool_eval import extract_ensemble_inference_times
from experiments.run_ensembling import  get_experiments

def collect_ensemble_inference_times(task_id, bmg_method, pipeline_dir, result_dir, seed, exp_type="base"):
    pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
        directory=os.path.join(pipeline_dir, str(task_id)),
        identifier=f"{bmg_method}_{seed}",
    )
    methods = ["ges"]
    assert exp_type in ("base"), \
        "collect_ensemble_inference_times only supports 'base' experiment type currently"

    X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)

    # pipeline.load_base_model_generation(X_train, y_train, meta)
    # pipeline.load_ensembling_predictions()
    pipeline.load_final_predictions()  # populates pool.final_results -- all extract_ensemble_inference_times needs

    n_models_ensembling = (
        len(pipeline.ensembling_predictions.get_config_ids())
        if pipeline.ensembling_predictions is not None else 0
    )

    name = "base_boost_single_-1"
    ensemble_pool = pipeline.load_ensemble_pool(name)
    res = extract_ensemble_inference_times(ensemble_pool)
    res.to_csv(os.path.join(result_dir, f"{name}_inference_times.csv"), index=False)

    for method in methods:
        pruning_strategy_names = ["top_50"]
        if n_models_ensembling > 50:
            pruning_strategy_names += ["nd_50_silo"]

        for pruning_strategy in pruning_strategy_names:
            name = f"base_boost_{method}_{pruning_strategy}"
            ensemble_pool = pipeline.load_ensemble_pool(name)
            res = extract_ensemble_inference_times(ensemble_pool)
            res.to_csv(os.path.join(result_dir, f"{name}_inference_times.csv"), index=False)


if __name__ == "__main__":
    # Driver loop inferred from collect_ensembling_results.py's task/seed/method iteration and
    # `./ensemble_pool_results/{exp_type}/{task}/{seed}/{method}` layout -- adjust if the real
    # driver for collect_ensembling_predictions differs.
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
            for method in ["so-smac"]:  # Only "so-smac" is run for allotment
                try:
                    collect_ensemble_inference_times(
                        task_id=task_id,
                        bmg_method=method,
                        pipeline_dir=args.output_dir,
                        result_dir=os.path.join(args.result_output_dir, args.exp_type, str(task_id), str(seed), method),
                        seed=seed,
                        exp_type=args.exp_type,
                    )
                except Exception:
                    print(f"Failed for task {task_id}, seed {seed}, method so-smac")
                    traceback.print_exc()
