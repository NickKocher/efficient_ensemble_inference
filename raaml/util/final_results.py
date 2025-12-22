import pandas as pd
import os
import yaml
import numpy as np

from frequency.freq_api_amd import get_freq_values

class FinalResults:
    
    def __init__(self, objective_names):
        self.objective_names = objective_names
        self.results = pd.DataFrame(columns=["config_id", "repeat_index", "n_cores", "frequency", *objective_names, "predictions_path"])
        
    def get_predictions(self, config_id):
        if config_id not in self.results["config_id"].values:
            return None
        path = self.results[self.results["config_id"] == config_id]["predictions_path"].iloc[0]
        return np.load(path)
    
    def get_objectives(self, config_id, n_cores, frequency, objective_order, index=0):
        return self.results[
            (self.results["config_id"] == config_id) &
            (self.results["n_cores"] == n_cores) & 
            (self.results["frequency"] == str(frequency))
        ][objective_order].iloc[index]
        
    def add_results(self, config_id, repeat_index, n_cores, frequency, result, predictions_path):
        self.results.loc[len(self.results)] = [config_id, repeat_index, n_cores, frequency] + [result[k] for k in self.objective_names] + [predictions_path]
        
    def save(self, output_dir):
        self.results.to_csv(os.path.join(output_dir, "final_results.csv"), index=False)
        with open(os.path.join(output_dir, "final_results_meta.yaml"), "w") as f:
            yaml.dump({"objective_names": self.objective_names}, f)
            
    def filter_configs_incomplete(self, config_ids, n_cores, frequency_mode):
        if frequency_mode == "boost":
            config_sizes = self.results[(self.results["n_cores"] == n_cores) & (self.results["frequency"] == "boost")].groupby("config_id").size()
            config_ids_complete = config_sizes[config_sizes == 5].index
        elif frequency_mode == "scaling":
            freq_vals = [str(f) for f in get_freq_values()]
            config_sizes = self.results[(self.results["n_cores"] == n_cores) & (self.results["frequency"].isin(freq_vals))].groupby("config_id").size()
            config_ids_complete = config_sizes[config_sizes == len(freq_vals) * 5].index
            
        config_ids_incomplete = set(config_ids).difference(set(config_ids_complete))
        
        return config_ids_incomplete
    
    def remove_config_ids(self, config_ids, n_cores, frequency_mode):
        freq_vals = ["boost"] if frequency_mode == "boost" else [str(f) for f in get_freq_values()]
        remove = self.results["config_id"].isin(config_ids) & (self.results["n_cores"] == n_cores) & (self.results["frequency"].isin(freq_vals))        
        self.results = self.results[~remove]
        
        
    @staticmethod
    def load_results(output_dir):
        if not os.path.exists(os.path.join(output_dir, "final_results.csv")):
            return None
        
        with open(os.path.join(output_dir, "final_results_meta.yaml")) as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        results = pd.read_csv(os.path.join(output_dir, "final_results.csv"), dtype={"frequency": str})
        final_res = FinalResults(meta["objective_names"])
        final_res.results = results
        return final_res