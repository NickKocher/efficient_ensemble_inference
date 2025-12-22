import numpy as np
import os
from itertools import product

from raaml.ensembling.ensemble_pool import SimpleEnsemblePool
from raaml.util.model_paths import get_final_model_paths, filter_refit_ens_models


def repeat(lst, n):
    return [x for x in lst for _ in range(n)]
    

def base_model_pool_to_ensemble_pool(
    base_model_pool,
    base_dir,
    metric,
    max_n_threads,
    frequencies,
    ensembling_predictions,
    scheduler
):
    """
    Convert a BaseModelPool to an EnsemblePool.
    
    Parameters
    ----------
    base_model_pool : BaseModelPool
        The base model pool to convert.
    
    Returns
    -------
    EnsemblePool
        The converted ensemble pool.
    """
    
    ids = base_model_pool.get_config_ids()
    ids = filter_refit_ens_models(base_dir, ensembling_predictions.method, ids)
    ep_unique_ids = ensembling_predictions.results["config_id"].unique()
    ids = [id for id in ids if id in ep_unique_ids]

    configs = base_model_pool.get_configs(ids, refit_success=False)
    paths = get_final_model_paths(base_dir, ids, ensembling_predictions.method)
    
    n_frequencies = len(frequencies)
    
    ids = np.repeat(ids, max_n_threads * n_frequencies)
    configs = repeat(configs, max_n_threads * n_frequencies)
    paths = repeat(paths, max_n_threads * n_frequencies)
    
    n_threads, frequencies = zip(*product(list(range(1, max_n_threads + 1)), frequencies))
    n_threads, frequencies = np.tile(n_threads, len(ids)), np.tile(frequencies, len(ids))
    
    weight_matrix = np.eye(len(ids))
    schedule_lists = [[id] for id in ids]
    
    
    return SimpleEnsemblePool(
        config_ids=ids,
        configs=configs,
        model_paths=paths,
        threads=n_threads,
        frequencies=frequencies,
        metric=metric,
        scheduler=scheduler,
        schedule_lists=schedule_lists,
        ensembling_predictions=ensembling_predictions,
        weight_matrix=weight_matrix
    )
        