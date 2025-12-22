from contextlib import nullcontext
import pandas as pd
import os
import numpy as np
import yaml
import ConfigSpace as CS
import pickle
from pynisher import limit, MemoryLimitException
import logging
import traceback

from raaml.util.warning_filter import filter_warnings
from raaml.util.non_dominated_sorting import non_dominated_sort, nsga2_sort
from raaml.util.raaml_progress import exception_to_status
from mosmac3.smac.utils.configspace import get_config_hash

class BaseModelPool:
    def __init__(
        self, 
        method_name,
        identifier, 
        config_space, 
        objective_names
    ):
        self.method_name = method_name
        self.identifier = identifier
        self.objective_names = objective_names
        self.configs = pd.DataFrame(columns=["config_id"] + list(config_space.keys()))
        self.results = pd.DataFrame(columns=["config_id", "config_hash", "fold", "nsga2_info"] + objective_names + ["time", "status"])
        self.config_space = config_space
        

    def add_entry(self, config, result, time, status, nsga2_info=""):
        config_id = len(self.configs)
        self.configs.loc[len(self.configs)] = [config_id] + list(config.get_array())
        for fold in range(len(result[self.objective_names[0]])):
            self.results.loc[len(self.results)] = [config_id, get_config_hash(config), fold, nsga2_info] + [result[k][fold] for k in self.objective_names] + [time, status]


    def save(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        with open(os.path.join(dir, "meta.yaml"), "w") as f:
            yaml.dump({
                "method_name": self.method_name,
                "identifier": self.identifier,
                "objective_names": self.objective_names
            }, f)

        self.configs.to_csv(os.path.join(dir, "configs.csv"), index=False)
        self.results.to_csv(os.path.join(dir, "results.csv"), index=False)
            
    
    def get_discovery_time(self, config_id):
        return self.results[self.results["config_id"] == config_id]["time"].min() # Should be the same for all folds


    def get_configs(self, config_ids = None, ret_ids=False, refit_success=True, base_dir=None):
        configs = []
        if config_ids is None:
            config_ids = self.results[
                (self.results["status"] == "SUCCESS") | 
                (self.results["status"] == "INTERMEDIATE_SUCCESS")
            ]["config_id"].unique()
            
        if refit_success:
            if base_dir is None:
                raise ValueError("base_dir must be provided if refit_success is True")
            config_ids = [_id for _id in config_ids if os.path.exists(os.path.join(base_dir, "models", "refit", f"{_id}.pkl"))]
        
        for config_id in config_ids:
            rows = self.configs[self.configs["config_id"] == config_id]
            if not rows.empty:
                numpy_vector = rows.iloc[0].to_numpy()
                if ret_ids:
                    configs.append((int(numpy_vector[0]), CS.Configuration(self.config_space, vector=numpy_vector[1:])))
                else:
                    configs.append(CS.Configuration(self.config_space, vector=numpy_vector[1:]))
            else:
                raise ValueError(f"Unknown config_id: {config_id}")
            
        return configs
    
    def get_config_ids(self):
        return self.results[self.results["status"].isin(["SUCCESS", "INTERMEDIATE_SUCCESS"])]["config_id"].unique()
    
    
    def get_sorted_config_ids_nsga2(self, config_ids):
        objectives = (
            self.results[self.results["config_id"].isin(config_ids)]
            .groupby("config_id")[self.objective_names]
            .mean()
            .reindex(config_ids)
        )
        res = nsga2_sort(objectives)
        res["config_id"] = config_ids
        res.reset_index(drop=True, inplace=True)
        res = res.sort_values(["rank", "crowding_distance"], ascending=[True, False])
        return res
    
    
    def get_pareto_ranks(self, config_ids):
        objectives = (
            self.results[self.results["config_id"].isin(config_ids)]
            .groupby("config_id")[self.objective_names]
            .mean()
            .reindex(config_ids)
        )
        _, ranks = non_dominated_sort(objectives)
        return ranks
    
    def get_refit_model_paths(self, base_dir, ens_method="cv", config_ids = None):
        final_model_paths = []
        if config_ids is None:
            config_ids = self.results[self.results["status"] == "SUCCESS"]["config_id"].unique()
            
        for config_id in config_ids:
            path_refit = os.path.join(base_dir, "models", "refit", f"{config_id}.pkl")
            if os.path.exists(path_refit):
                final_model_paths.append(path_refit)
        
        return final_model_paths
    

    def __len__(self):
        return len(self.configs)

    def print_base_model_results(self, print_options = {}):
        print("Base Model Results (On BMG CV-Folds)")
        with pd.option_context(
            'display.max_rows', print_options.get('max_rows', None),
            'display.width', print_options.get('width', 500),
            'display.max_columns', print_options.get('max_columns', None)
        ):
            print(self.results.groupby("config_id").agg({
                **{obj: 'mean' for obj in self.objective_names},
                'status': 'first',
                'time': 'first',
                'config_hash': 'first'
            }).reset_index())
            
            
    def refit_base_models(self, X, y, limit_args, load_fn, output_dir, affinity_manager, raaml_progress, n_cores):
        def fit(model, X, y, save_path):
            affinity_manager.reserve_and_set_cores()
            
            model.set_n_threads(n_cores)
            if hasattr(model, "set_timeout"):
                model.set_timeout(limit_args["wall_time"][0] * 0.7)
            
            model.fit(X, y)
            
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
        
        if not os.path.exists(os.path.join(output_dir, "models", "refit")):            
            os.makedirs(os.path.join(output_dir, "models", "refit"))
        else:
            logging.info("Base Model Pool is already refitted. Skipping refit.")
            return
                
        for config_id, config in self.get_configs(ret_ids=True, refit_success=False):
            model = load_fn(config)
            
            refit_path = os.path.join(output_dir, "models", "refit", f"{config_id}.pkl")

            try:
                with filter_warnings():
                    with limit(fit, **limit_args) as limited_fit:
                        try:
                            limited_fit(model, X, y, refit_path)
                        except Exception as e:
                            # Make sure all OOM errors are reported as such
                            if "can't allocate memory" in str(e):
                                logging.warning(f"Out of memory error while running _target_fun with config: {get_config_hash(config)}")
                                raise MemoryLimitException("Catched OOM error in target function")
                            else:
                                raise e
                raaml_progress.update_progress("refit", "SUCCESS")
            except Exception as e:
                short_exc = str(e).split("\n")[0]
                logging.info(f"Refitting of model {config_id} failed: {short_exc}. Using original model instead.")
                logging.debug(traceback.format_exc())
                
                if os.path.exists(refit_path):
                    os.remove(refit_path)
                
                raaml_progress.update_progress("refit", exception_to_status(e))
                continue
                
            
        

@staticmethod
def load_base_model_pool_from_dir(load_dir, config_space):
    
    bmp_load_dir = os.path.join(load_dir, "base_model_pool")
    
    if not os.path.exists(bmp_load_dir):
        raise ValueError(f"Could not find base model pool in {load_dir}")
    
    with open(os.path.join(bmp_load_dir, "meta.yaml")) as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)
    pool = BaseModelPool(
        method_name=meta["method_name"],
        identifier=meta["identifier"],
        config_space=config_space,
        objective_names=meta["objective_names"]
    )
    pool.configs = pd.read_csv(os.path.join(bmp_load_dir, "configs.csv"))
    pool.results = pd.read_csv(os.path.join(bmp_load_dir, "results.csv"))
    
    return pool
        
        
        
def base_model_pool_from_smac_history(smac_hist, multi_objective, identifier, cs, objective_names, start_time):
    pool = BaseModelPool("mo-smac" if multi_objective else "smac", identifier, cs, objective_names)

    for key, entry in smac_hist.items():
        config = smac_hist.ids_config[key.config_id]
        additional_infos = entry.additional_info
        if "results" in additional_infos:
            results = additional_infos["results"]
        else:
            results = {obj: [np.inf] for obj in objective_names}

        pool.add_entry(config, results, entry.endtime - start_time, entry.status.name)

    return pool