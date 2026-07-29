from raaml.ensembling.ensemble_pool import SimpleEnsemblePool
from raaml.ensembling.scheduling.scheduler import Task
from raaml.util.non_dominated_sorting import is_pareto_optimal

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap





def compute_hv(results, pool_name, objectives, min_vals, max_vals, log=True):
    results_pool = results[results["pool"] == pool_name]
    ind = HV(ref_point=[1] * len(objectives))
    
    if log:
        objectives_res = np.log(results_pool[objectives]+1e-12)
        min_log, max_log = np.log(min_vals+1e-12), np.log(max_vals)
        objectives_res = (objectives_res - min_log) / (max_log - min_log)
    else:
        objectives_res = results_pool[objectives]
        objectives_res = (objectives_res - min_vals) / (max_vals - min_vals)
    
    hv_value = ind(objectives_res.values)
    return hv_value



def evaluate_multiple_ensemble_pools(pools : dict[str, SimpleEnsemblePool], X, y):
    results = []
    metric_name = list(pools.values())[0].metric_name
    for i, pool_name in enumerate(pools):
        assert pools[pool_name].metric_name == metric_name
        results_pd = evaluate_ensemble_pool(pools[pool_name], X, y)
        results_pd["pool"] = pool_name
        results.append(results_pd)
    results = pd.concat(results)
    
    objectives = [metric_name, "inference_time", "energy_consumption"]
    min_vals, max_vals = results[objectives].min(), results[objectives].max()
    
    results_pools = []
    for pool_name in pools:
        hv = compute_hv(results, pool_name, objectives, min_vals, max_vals)
        avg_ensemble_size = results[results["pool"] == pool_name]["ensemble_size"].mean()
        avg_gap = results[results["pool"] == pool_name]["avg_gap"].mean()
        avg_inference_time = results[results["pool"] == pool_name]["inference_time"].mean()
        avg_energy_consumption = results[results["pool"] == pool_name]["energy_consumption"].mean()
        avg_metric_value = results[results["pool"] == pool_name][metric_name].mean()
        
        results_pools.append({
            "pool_name" : pool_name, 
            "hypervolume" : hv, 
            "avg_ensemble_size" : avg_ensemble_size, 
            "avg_gap" : avg_gap,
            "avg_metric_value" : avg_metric_value,
            "avg_inference_time" : avg_inference_time, 
            "avg_energy_consumption" : avg_energy_consumption
        })
    
    results_pools = pd.DataFrame(results_pools)
    
    return results_pools, results

    
        
    
def ensemble_to_task_list(ensemble_pool, ensemble_index, resources_pd):
    weights = ensemble_pool.weight_matrix[ensemble_index]
    task_list = []
    
    for i, (config_id, n_cores, frequency) in enumerate(zip(ensemble_pool.config_ids, ensemble_pool.threads, ensemble_pool.frequencies)):
        if np.abs(weights[i]) < 1e-12:
            continue
        inf_time, energy = resources_pd[["inference_time", "energy_consumption"]].iloc[i]
        task_list.append(Task(config_id, n_cores, frequency, inf_time, energy))
        
    return task_list

def extract_ensemble_inference_times(pool: SimpleEnsemblePool):
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
        weights = pool.weight_matrix[i]
        active = np.abs(weights) >= 1e-12  # same "in-use" threshold as ensemble_to_task_list
        inference_times = resource_usage[active, inf_time_index].tolist()
        res.loc[i] = [inference_times]

    return res

def evaluate_ensemble_pool_ens_preds(pool : SimpleEnsemblePool, y_ens):
    if pool is None:
        return pd.DataFrame()
    
    predictions = []
    resource_usage = np.zeros((pool.weight_matrix.shape[1], len(pool.assembled_resource_provider)))
    
    scheduler = pool.scheduler
    
    resource_names = pool.assembled_resource_provider.get_names()
    
    predictions = pool.ensembling_predictions.get_predictions()
    config_ids = list(pool.ensembling_predictions.get_config_ids())
    config_ids_pool = pool.config_ids
    indices = [config_ids.index(cid) for cid in config_ids_pool if cid in config_ids]
    predictions = [predictions[i] for i in indices]

    for i in range(pool.weight_matrix.shape[1]):
        config_id = pool.config_ids[i]
        n_cores = pool.threads[i]
        frequency = pool.frequencies[i]
        resource_usage[i] = pool.ensembling_predictions.get_resources(config_id, n_cores, frequency, resource_names)

    predictions = np.array(predictions)
        
    result_columns = [f"{pool.metric_name}_ens", "inference_time_ens", "energy_consumption_ens"]
    res = pd.DataFrame(columns=result_columns)   

    inf_time_index = resource_names.index("inference_time")
    energy_index = resource_names.index("energy_consumption")
    
    for i in range(len(pool)):
        ensemble_pred = pool._weighted_ensemble(
            predictions, 
            pool.weight_matrix[i]
        )
        score = pool.metric(y_ens, ensemble_pred)

        inf_time = np.sum(resource_usage[:, inf_time_index] * pool.weight_matrix[i])
        energy = np.sum(resource_usage[:, energy_index] * pool.weight_matrix[i])
        
        res.loc[i] = [score, inf_time, energy]
        
    return res
    
    
def evaluate_ensemble_pool(pool : SimpleEnsemblePool, X, y):
    if pool is None:
        return pd.DataFrame()
    
    predictions = []
    resource_usage = np.zeros((pool.weight_matrix.shape[1], len(pool.assembled_resource_provider)))
    scores = []
    
    scheduler = pool.scheduler
    
    resource_names = pool.assembled_resource_provider.get_names()
    
    first_cid = pool.final_results.results["config_id"].unique()[0]
    pred_shape = pool.final_results.get_predictions(first_cid).shape
    
    pred_missing = False            
    for i, (config_id, n_cores, frequency) in enumerate(zip(pool.config_ids, pool.threads, pool.frequencies)):
        preds = pool.final_results.get_predictions(config_id)
        if preds is None:
            pool.weight_matrix[:, i] = 0.0
            pred_missing = True
            preds = np.zeros(pred_shape)
        else:
            preds = preds.squeeze()
            objectives = pool.final_results.get_objectives(config_id, n_cores, frequency, [pool.metric.name, *resource_names])
            resource_usage[i] = objectives[resource_names]
        predictions.append(preds)
        scores.append(objectives[pool.metric.name])
        
    if pred_missing:
        # Re-normalize weights
        pool.weight_matrix = pool.weight_matrix[np.abs(pool.weight_matrix.sum(axis=1)) > 1e-12]
        pool.weight_matrix = pool.weight_matrix / pool.weight_matrix.sum(axis=1, keepdims=True)
        
    predictions = np.array(predictions)
        
    result_columns = [pool.metric_name, "inference_time", "energy_consumption", "ensemble_size", "avg_gap"]
    res = pd.DataFrame(columns=result_columns)   
    
    for i in range(len(pool)):
        ensemble_pred = pool._weighted_ensemble(
            predictions, 
            pool.weight_matrix[i]
        )
        
        schedule_list = pool.schedule_lists[i]
        task_list = ensemble_to_task_list(
            pool, 
            i, 
            pd.DataFrame(resource_usage, columns=resource_names),
        )
        
        inf_time, energy, avg_gap, _ = scheduler.compute_online_resources(
            task_list,
            schedule_list
        )
        
        ensemble_size = len(np.unique(schedule_list))
        score = pool.metric(y, ensemble_pred)
        
        res.loc[i] = [score, inf_time, energy, ensemble_size, avg_gap / inf_time]
        
    return res
    
    
def plot_pareto_fronts(results, metric, save_path):
    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))

    # Compute Pareto ranks
    results["pareto_opt"] = is_pareto_optimal(results[[metric, "inference_time", "energy_consumption"]])

    # Three pairwise plots
    for i, pair in enumerate([
        (metric, "inference_time"),
        (metric, "energy_consumption"),
        ("inference_time", "energy_consumption")
    ]):
        pools = results["pool"].unique()
        cmap = get_cmap("tab10")
        
        for j, pool_name in enumerate(pools):
            if "+" in pool_name:
                continue

            pool_results = results[results["pool"] == pool_name]

            # Non-optimal points: faded (alpha=0.3)
            pool_non_opt = pool_results[~pool_results["pareto_opt"]]
            ax[i].scatter(
                pool_non_opt[pair[0]],
                pool_non_opt[pair[1]],
                s=20,
                label=None,
                alpha=0.6,
                color=cmap(j),
                edgecolor="none"
            )

            # Optimal points: fully visible (alpha=1)
            pool_opt = pool_results[pool_results["pareto_opt"]]
            ax[i].scatter(
                pool_opt[pair[0]],
                pool_opt[pair[1]],
                s=40,
                label=f"{pool_name}",
                alpha=1.0,
                color=cmap(j),
                edgecolor="black",
                linewidth=0.3
            )

        # Axes formatting
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_xlabel(pair[0])
        ax[i].set_ylabel(pair[1])
        ax[i].set_title(f"Pareto Front for {pair[0]} vs {pair[1]}")
        ax[i].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    
"""def plot_pareto_fronts(results, metric, save_path):
    fig, ax = plt.subplots(
        ncols=3,
        figsize=(16, 6)
    )
    
    _, ranks = non_dominated_sort(results[[metric, "inference_time", "energy_consumption"]])
    indices_optimal = np.array(ranks) == 0
    
    for i, pair in enumerate([
        (metric, "inference_time"),
        (metric, "energy_consumption"),
        ("inference_time", "energy_consumption")
    ]):
        for j, pool_name in enumerate(results["pool"].unique()):
            if "+" in pool_name:
                continue
            pool_results = results[results["pool"] == pool_name]
            ax[i].scatter(
                pool_results[pair[0]], 
                pool_results[pair[1]], 
                s=10,
                label=f"{pool_name}"
            )
            ax[i].set_xlabel(pair[0])
            ax[i].set_ylabel(pair[1])
            ax[i].set_title(f"Pareto Front for {pair[0]} vs {pair[1]}")
        
        ax[i].scatter(
            results.iloc[indices_optimal, pair[0]], 
            results.iloc[indices_optimal, pair[1]],
            s=10,
            label="Pareto Front",
            marker="x"
        )
            
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].legend()
        
    plt.tight_layout()
    fig.savefig(save_path)"""
