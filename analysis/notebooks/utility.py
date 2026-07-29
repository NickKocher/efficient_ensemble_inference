import numpy as np
import pandas as pd
import warnings
from scipy.stats import rankdata
import tqdm
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def compute_min_max_data(data, task_id):    
    min = {}
    max = {}
    for col in ["RMSE", "Log Loss", "1 - ROC AUC", "inference_time", "energy_consumption"]:
        data_task_id = data[data["task_id"].astype(int) == task_id]
        if (~data_task_id[col].isna()).sum() > 0:
            min[col] = float(data_task_id[col].min())
            max[col] = float(data_task_id[col].max())
            if not np.isfinite(min[col]) or not np.isfinite(max[col]):
                raise ValueError(f"Non-finite min/max for task ID {task_id}, column {col}")
    return min, max



def merge_min_max(min_old, max_old, min_new, max_new):
    for col in ["RMSE", "Log Loss", "1 - ROC AUC", "inference_time", "energy_consumption"]:
        if col in min_old:
            min_old[col] = min(min_old[col], min_new[col])
            max_old[col] = max(max_old[col], max_new[col])
    return [min_old, max_old]



def compute_min_max(all_dataframes):
    result = {}
    for i, df in enumerate(all_dataframes):
        print(f"Processing dataframe {i+1}/{len(all_dataframes)}")
        for task_id in tqdm.tqdm(df["task_id"].unique()):
            task_id = int(task_id)
            min, max = compute_min_max_data(df, task_id)
            
            if task_id in result:
                old_min, old_max = result[task_id]
                result[task_id] = merge_min_max(old_min, old_max, min, max)
            else:
                result[task_id] = [min, max]
    return result


def unique_rows_with_tolerance(arr, tol=1e-10):
    
    # Round for comparison only
    rounded = np.round(arr / tol) * tol
    
    # Use structured array trick for np.unique
    _, idx = np.unique(rounded, return_index=True, axis=0)
    
    # Return original values at unique indices
    return arr[np.sort(idx)]


def _find_extremes(metric, min_val, all_metrics, task_id, min, max, dataframes):
    results_all = []
    for df in tqdm.tqdm(dataframes):
        start = time.time()
        df_task = df[df["task_id"].astype(int) == task_id]
        df_task_min = df_task[df_task[metric] == min_val]
        print(f"Time to filter dataframe for task {task_id} and metric {metric} => {len(df_task_min)}: {time.time() - start:.2f} seconds")
        start = time.time()
        results = df_task_min[all_metrics].values
        results = unique_rows_with_tolerance(results, tol=1e-10)
        results_all.append(results)
        print(f"Time to process rows for task {task_id} and metric {metric}: {time.time() - start:.2f} seconds")
    results = np.vstack(results_all)

    if len(results) == 0:
        raise ValueError(f"No entries found for task ID {task_id} with min {metric} value {min_val}")
    
    min_arr, max_arr = np.array([min[key] for key in all_metrics]), np.array([max[key] for key in all_metrics])
    results = (results - min_arr) / (max_arr - min_arr)
    
    if len(results) == 1:
        return {key: results[0][i] for i, key in enumerate(all_metrics)}

    results = pd.DataFrame(results, columns=all_metrics)

    #print("Results normalized", results)
    min_avg, min_index = np.inf, -1
    tie_cols = [m for m in all_metrics if m != metric]
    #print("tie_cols", tie_cols)
    for i in range(results.shape[0]):
        mean = results.loc[i, tie_cols].mean()
        if mean < min_avg:
            min_avg = mean
            min_index = i

    return results.iloc[min_index].to_dict()

def _almost_equal_entries(a, b, rtol=1e-8, atol=1e-12):
    for key in a.keys():
        if not np.isclose(a[key], b[key], rtol=rtol, atol=atol):
            return False
    return True


def compute_extremes(min_max_dict, all_dataframes):    
    result = {}

    for task_id in tqdm.tqdm(min_max_dict.keys()):
        result[task_id] = {}
        metrics = list(min_max_dict[task_id][0].keys())
        min, max = min_max_dict[task_id]
        for metric in metrics:
            dict_ = _find_extremes(
                metric,
                min_max_dict[task_id][0][metric],
                metrics,
                task_id,
                min,
                max,
                all_dataframes
            )
            
            for key in dict_.keys():
                dict_[key] = float(dict_[key])
                
            result[task_id][metric] = dict_

    return result


def get_all_dataframes():
    base_model_pools = pd.read_csv("/home/kocher/RA-AML/analysis/data/base_model_pools.csv", index_col=0)
    ensemble_pools = pd.read_csv("/home/kocher/RA-AML/analysis/data/base_ensemble_pools.csv", index_col=0)
    fs_eps = pd.read_csv("/home/kocher/RA-AML/analysis/data/frequency_scaling_alloted_eps.csv", index_col=0)
    parallel_eps = pd.read_csv("/home/kocher/RA-AML/analysis/data/parallel_alloted_eps.csv", index_col=0)
    fs_parallel_eps = pd.read_csv("/home/kocher/RA-AML/analysis/data/frequency_scaling+parallel_alloted_eps.csv", index_col=0)

    dataframes = [base_model_pools, ensemble_pools, fs_eps, parallel_eps, fs_parallel_eps]

    #warnings.warn("Also add ensemble pools with multi-threading and frequency scaling etc. here!")   
    return dataframes


def get_pareto_front(data):
    less_equal = np.all(data[None, :, :] <= data[:, None, :], axis=2)
    strictly_less = np.any(data[None, :, :] <  data[:, None, :], axis=2)
    dominates = less_equal & strictly_less
    pareto = ~np.any(dominates, axis=1)
    return pareto



def compute_ranks(data, mode, axis=1):
    """
    Compute ranks along the given axis.
    mode: "min"  -> smallest value gets rank 1
          "max"  -> largest value gets rank 1
    Ties use average ranks (same as scipy.stats.rankdata(method="average"))
    """
    assert mode.lower() in ["min", "max"], "Mode must be 'min' or 'max'"

    # Move axis to last for easy reshaping
    data = np.asarray(data)
    data_swapped = np.moveaxis(data, axis, -1)

    # Prepare output array
    ranks = np.empty_like(data_swapped, dtype=float)

    # Determine if we should invert values for max-mode ranking
    invert = (mode.lower() == "max")

    # Rank each slice
    it = np.nditer(data_swapped[..., 0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        values = data_swapped[idx]  # 1D vector along the target axis
        
        if invert:
            # Larger values should get rank 1 -> rank negative values
            r = rankdata(-values, method='average')
        else:
            # Smaller values get rank 1
            r = rankdata(values, method='average')
        
        ranks[idx] = r
        it.iternext()

    # Move axis back
    return np.moveaxis(ranks, -1, axis)


def get_group_task_ids(name):
    with open(f"/home/kocher/RA-AML/analysis/data/groups/{name}.yaml", "r") as f:
        return yaml.safe_load(f)