from joblib import parallel_backend
import psutil
import os
from raaml.util.warning_filter import filter_warnings

def multi_threaded_predict(model, X, n_cores, labels, assembled_res_provider):
    affinity = list(os.sched_getaffinity(0))
    if len(affinity) < n_cores:
        raise ValueError(f"Not enough CPU cores available. Expected at least {n_cores}, but got {len(affinity)}.")
    p = psutil.Process(os.getpid())
    p.cpu_affinity(affinity[:n_cores])
    
    with filter_warnings():
        if hasattr(model, "predict_proba"):
            with assembled_res_provider:
                with parallel_backend("threading", n_jobs=n_cores):
                    y_pred = model.predict_proba(X)
            y_pred = model.rearrange_probas(y_pred, labels)
        else:
            with assembled_res_provider:
                with parallel_backend("threading", n_jobs=n_cores):
                    y_pred = model.predict(X)
            y_pred = y_pred.squeeze()
                
    p.cpu_affinity(affinity)
    return y_pred, assembled_res_provider.get_resource()