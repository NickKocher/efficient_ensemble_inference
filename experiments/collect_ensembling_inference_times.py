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

import tqdm

from raaml.raaml import ResourceAwareAutoMLPipeline
from raaml.util.openml_data import load_openml_data_new, load_amlb_tasks
from raaml.util.ensemble_pool_eval import extract_ensemble_inference_times


def collect_ensemble_inference_times(task_id, bmg_method, pipeline_dir, result_dir, seed, exp_type="base"):
    pipeline = ResourceAwareAutoMLPipeline.loadRAAML(
        directory=os.path.join(pipeline_dir, str(task_id)),
        identifier=f"{bmg_method}_{seed}",
    )
    if exp_type == "ablations":
        methods = ["mo-ges-no-size-restriction", "mo-ges-no-removal", "mo-ges-no-size-no-removal", "mo-ges-loss",
                   "mo-ges-size-linear", "mo-ges-size-quadratic", "mo-ges-size-exp", "mo-ges-100", "mo-ges-500",
                   "mo-ges-size-linear-with-removal", "mo-ges-size-linear-with-removal-100",
                   "mo-ges-size-linear-with-removal-500"]
    elif exp_type == "baselines":
        methods = ["nsga2"]
    else:
        methods = ["ges", "mo-ges", "qdo-es", "infer-qdo-es", "size-qdo-es", "energy-qdo-es"]
    assert exp_type in ("base", "ablations", "baselines"), \
        "collect_ensemble_inference_times only supports 'base', 'ablations' or 'baselines' experiment type currently"

    X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)

    pipeline.load_base_model_generation(X_train, y_train, meta)
    pipeline.load_ensembling_predictions()
    pipeline.load_final_predictions()  # populates pool.final_results -- all extract_ensemble_inference_times needs

    n_models_ensembling = (
        len(pipeline.ensembling_predictions.get_config_ids())
        if pipeline.ensembling_predictions is not None else 0
    )

    # Single (non-ensembled base model) pool -- only the final snapshot, matching the
    # `if time == -1:` branch in collect_ensembling_predictions (the rest of that time loop
    # is currently a no-op there, so there's nothing else to mirror).
    name = "base_boost_single_-1"
    ensemble_pool = pipeline.load_ensemble_pool(name)
    res = extract_ensemble_inference_times(ensemble_pool)
    res.to_csv(os.path.join(result_dir, f"{name}_inference_times.csv"), index=False)

    for method in methods:
        pruning_strategy_names = ["top_50"]
        if n_models_ensembling > 50:
            pruning_strategy_names += ["silo_top_50", "nd_50_silo"]

        for pruning_strategy in pruning_strategy_names:
            name = f"base_boost_{method}_{pruning_strategy}"
            ensemble_pool = pipeline.load_ensemble_pool(name)
            res = extract_ensemble_inference_times(ensemble_pool)
            res.to_csv(os.path.join(result_dir, f"{name}_inference_times.csv"), index=False)


if __name__ == "__main__":
    # Driver loop inferred from collect_ensembling_results.py's task/seed/method iteration and
    # `./ensemble_pool_results/{exp_type}/{task}/{seed}/{method}` layout -- adjust if the real
    # driver for collect_ensembling_predictions differs.
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_dir", type=str, required=True,
                         help="Directory containing the RAAML pipeline output, one subfolder per task_id")
    parser.add_argument("--exp_type", type=str, default="base",
                         help="Type of experiment: base, ablations or baselines")
    args = parser.parse_args()

    tasks = [t[1] for t in load_amlb_tasks()]
    base_folder = f"./ensemble_pool_results/{args.exp_type}/"

    for task_id in tqdm.tqdm(tasks):
        for seed in range(1, 6):
            for bmg_method in ["so-smac"]:
                result_dir = os.path.join(base_folder, str(task_id), str(seed), bmg_method)
                if not os.path.exists(os.path.join(result_dir, "meta_results.yaml")):
                    print(f"Missing {result_dir} (run collect_ensembling_predictions first), skipping")
                    continue
                try:
                    collect_ensemble_inference_times(
                        task_id=task_id,
                        bmg_method=bmg_method,
                        pipeline_dir=args.pipeline_dir,
                        result_dir=result_dir,
                        seed=seed,
                        exp_type=args.exp_type,
                    )
                except Exception:
                    print(f"Failed for task {task_id}, seed {seed}, method {bmg_method}")
                    traceback.print_exc()
