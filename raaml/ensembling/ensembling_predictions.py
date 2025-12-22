from math import exp
import pandas as pd
import os
import yaml
import numpy as np
import logging
import copy

from raaml.util.non_dominated_sorting import non_dominated_sort
from raaml.util.openml_data import get_resources

from frequency.freq_api_amd import get_freq_values

class EnsemblingPredictions:
    
    def __init__(self, method, objective_name, resource_names):
        self.method = method
        self.objective_name = objective_name
        self.resource_names = resource_names

        self.results = pd.DataFrame(columns=["config_id", "cv_index", "n_threads", "frequency", "predictions_path", "discovery_time"] + [objective_name] + resource_names)

    def add_cv_fold_predictions(self, config_id, cv_index, n_threads, frequency, predictions_path, objectives, discovery_time):
        self.results.loc[len(self.results)] = [config_id, cv_index, n_threads, frequency, predictions_path, discovery_time] + [objectives[o] for o in [self.objective_name] + self.resource_names]

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.results.to_csv(os.path.join(output_dir, "ensembling_predictions.csv"), index=False)
        meta = {"method": self.method, "objective_name": self.objective_name, "resource_names": self.resource_names}
        with open(os.path.join(output_dir, "ensembling_predictions_meta.yaml"), "w") as f:
            yaml.dump(meta, f)
            
            
            
    def filter_configs_incomplete(self, config_ids, n_cores, frequency_mode):
        if frequency_mode == "boost":
            config_sizes = self.results[(self.results["n_threads"] == n_cores) & (self.results["frequency"] == "boost")].groupby("config_id").size()
            config_ids_complete = config_sizes[config_sizes == 5].index
        elif frequency_mode == "scaling":
            freq_vals = [str(f) for f in get_freq_values()]
            config_sizes = self.results[(self.results["n_threads"] == n_cores) & (self.results["frequency"].isin(freq_vals))].groupby("config_id").size()
            config_ids_complete = config_sizes[config_sizes == len(freq_vals) * 5].index
        
        config_ids_incomplete = set(config_ids).difference(set(config_ids_complete))
        print("\nEnsembling Predictions Filtering:")
        print("Config ids:", config_ids)
        print("Config ids complete:", config_ids_complete)
        print(" ==> Config ids incomplete:", config_ids_incomplete)
        return config_ids_incomplete
    
    
    def remove_config_ids(self, config_ids, n_cores, frequency_mode):
        freq_vals = ["boost"] if frequency_mode == "boost" else [str(f) for f in get_freq_values()]
        remove = self.results["config_id"].isin(config_ids) & (self.results["n_threads"] == n_cores) & (self.results["frequency"].isin(freq_vals))
        print(f"Ensembling: Removing {len(self.results[remove])} / {len(self.results)} predictions")
        self.results = self.results[~remove]
    
            

    def get_ensembling_data(self, do_allotment):
        # Base info for n_threads=1
        config_ids = self.get_config_ids()
        preds = self.get_predictions()
        
        if len(config_ids) != len(preds):
            raise ValueError(f"Number of configs does not match number of predictions! ({len(config_ids)} != {len(preds)})")

        # Collect per-config thread choices
        inference_settings = self.get_inference_settings(config_ids, do_allotment)

        # Expand predictions & resources according to n_threads choices
        expanded_preds = []
        expanded_resources = []
        expanded_config_ids = []
        expanded_n_threads = []
        expanded_frequencies = []

        for idx, config_id in enumerate(config_ids):
            preds_for_config = preds[idx]
            for n, freq in inference_settings[idx]:
                expanded_preds.append(preds_for_config)
                expanded_resources.append(
                    self.get_resources(
                        config_id, n, freq, self.resource_names
                    )
                )
                expanded_config_ids.append(config_id)
                expanded_n_threads.append(n)
                expanded_frequencies.append(freq)

        return (
            np.array(expanded_config_ids),
            np.array(expanded_preds),
            np.array(expanded_resources),
            np.array(expanded_n_threads),
            np.array(expanded_frequencies)
        )
    
    def get_inference_settings(self, config_ids, do_allotment):
        result = []
        for config_id in config_ids:
            data_config_id = self.results[self.results["config_id"] == config_id]
            if do_allotment:
                for freq in data_config_id["frequency"].unique():
                    data_freq = data_config_id[data_config_id["frequency"] == freq].groupby("n_threads")[["n_threads", "inference_time"]].mean()
                    lowest_n_threads = data_freq["n_threads"].astype(int).min()
                    baseline_time = data_freq.loc[data_freq["n_threads"] == lowest_n_threads, "inference_time"].iloc[0]
                    n_threads = [lowest_n_threads] + data_freq[data_freq["inference_time"] < baseline_time]["n_threads"].astype(int).unique().tolist()
                    result.append([(n, freq) for n in n_threads])
            else:
                frequencies = np.unique(data_config_id["frequency"])
                max_freq = frequencies.max()
                result.append([(1, max_freq)])
        return result
    
    def get_predictions(self):
        groups = self.results[self.results["n_threads"] == 1].groupby("config_id")
        preds = []
        for _, group in groups:
            group_preds = []
            max_cv = group["cv_index"].max()
            for cv_fold in range(max_cv + 1):
                path = group[group["cv_index"] == cv_fold]["predictions_path"].iloc[0]
                group_preds.append(np.load(path).squeeze()    )
            preds.append(np.concatenate(group_preds, axis=0))
            
            
        return np.array(preds)


    """def get_objectives(self, ret_corresponding_setups=False):
        df = self.results
        df = df.groupby(["config_id", "n_threads"])[["config_id", "n_threads"] + [self.objective_name] + self.resource_names].mean()
        objectives = df[[self.objective_name] + self.resource_names]
        
        if ret_corresponding_setups:
            setups = df[["config_id", "n_threads"]]
            return objectives, setups
            
        return objectives"""
        
    def get_resources(self, config_id, n_cores, frequency, res_names):        
        resources = self.results[
            (self.results["config_id"] == config_id) &
            (self.results["n_threads"] == n_cores) & 
            (self.results["frequency"] == str(frequency))
        ][res_names]
        
        return resources.mean().to_numpy()
        
    
    def get_config_ids(self):
        return np.sort(self.results["config_id"].unique())
    
    def remove_config_id(self, config_id):
        self.results = self.results[self.results["config_id"] != config_id]
        
    def apply_filter(self, filter, base_model_pool):
        if filter is None:
            return
        
        filter.filter(self, base_model_pool)
    
    
    def copy(self):
        return copy.deepcopy(self)
        
    def __eq__(self, other):
        return (
            self.method == other.method and
            self.objective_name == other.objective_name and
            self.resource_names == other.resource_names and
            pd.util.hash_pandas_object(self.results).sum() == pd.util.hash_pandas_object(other.results).sum()
        )
        

@staticmethod
def load_ensembling_predictions(input_dir, method):
    ens_preds_dir = os.path.join(input_dir, f"ensembling_predictions_{method}")
    
    if (
        not os.path.exists(ens_preds_dir) or
        not os.path.exists(os.path.join(ens_preds_dir, "ensembling_predictions_meta.yaml")) or
        not os.path.exists(os.path.join(ens_preds_dir, "ensembling_predictions.csv"))
    ):
        return None
    
    with open(os.path.join(ens_preds_dir, "ensembling_predictions_meta.yaml"), "r") as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)
    df = pd.read_csv(os.path.join(ens_preds_dir, "ensembling_predictions.csv"), dtype={"frequency": str})
    ensemble_predictions = EnsemblingPredictions(method=meta["method"], objective_name=meta["objective_name"], resource_names=meta["resource_names"])
    ensemble_predictions.results = df
    
    return ensemble_predictions