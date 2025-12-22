from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd

from raaml.base_model_generation.pd_base_model_pool import BaseModelPool
from raaml.ensembling.ensembling_predictions import EnsemblingPredictions


class EnsemblingFilter(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def filter(self, ensembling_predictions : EnsemblingPredictions, base_model_pool : BaseModelPool):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def verbose_filter(self, len_before, len_after):
        len_before, len_after = len_before//5, len_after//5
        if len_before == 0:
            logging.info(f"{self.name} did not remove any configurations (0 of 0). Remaining configurations: {len_after}")
        else:
            logging.info(f"{self.name} removed {len_before - len_after} of {len_before} configurations ({(len_before - len_after) / len_before * 100:.2f} %). Remaining configurations: {len_after}")


class NonDominatedSortingFilter(EnsemblingFilter):
    def __init__(self, max_pareto_rank=None, n_models=None, silo=False):
        self.max_pareto_rank = max_pareto_rank
        self.n_models = n_models
        self.silo = silo
        
        if (self.max_pareto_rank is None and self.n_models is None):
            raise ValueError("Either max_pareto_rank or n_models must be specified, but not both.")
        
        super().__init__("Pareto Rank Filter" + (" (Silo)" if silo else ""))

    def filter(self, ensembling_predictions: EnsemblingPredictions, base_model_pool: BaseModelPool):
        len_before = len(ensembling_predictions.results)
        config_ids = ensembling_predictions.get_config_ids()

        # NSGA-II sorting: returns dataframe with columns ['config_id', 'rank', 'crowding_distance', ...]
        sorted_df = base_model_pool.get_sorted_config_ids_nsga2(config_ids)
        
        if not self.silo:
            config_ids_keep = self._select_non_silo(sorted_df)
        else:
            config_ids_keep = self._select_silo(sorted_df, base_model_pool)
        
        # Apply filter to ensemble predictions
        ensembling_predictions.results = ensembling_predictions.results[
            ensembling_predictions.results["config_id"].isin(config_ids_keep)
        ]

        len_after = len(ensembling_predictions.results)
        self.verbose_filter(len_before, len_after)

    # ------------------------
    # Non-silo (global) logic
    # ------------------------
    def _select_non_silo(self, sorted_df):
        if self.max_pareto_rank is not None:
            config_ids_keep = sorted_df[sorted_df["rank"] <= (self.max_pareto_rank - 1)]["config_id"].tolist()
        elif self.n_models is not None:
            config_ids_keep = (
                sorted_df.sort_values(["rank", "crowding_distance"], ascending=[True, False])
                ["config_id"]
                .head(self.n_models)
                .tolist()
            )
        else:
            raise ValueError("Either max_pareto_rank or n_models must be specified.")
        return config_ids_keep

    # ------------------------
    # Silo (per-model) logic
    # ------------------------
    def _select_silo(self, sorted_df, base_model_pool):
        configs = base_model_pool.configs[
            base_model_pool.configs["config_id"].isin(sorted_df["config_id"])
        ][["config_id", "model"]]
        sorted_df = sorted_df.merge(configs, on="config_id", how="left")

        selected_ids = []

        if self.max_pareto_rank is not None:
            raise NotImplementedError("max_pareto_rank not supported in silo mode.")
        elif self.n_models is not None:
            grouped = [
                g.sort_values(["rank", "crowding_distance"], ascending=[True, False])
                for _, g in sorted_df.groupby("model")
            ]
            round_idx = 0
            while len(selected_ids) < self.n_models and any(round_idx < len(g) for g in grouped):
                for g in grouped:
                    if round_idx < len(g):
                        selected_ids.append(g.iloc[round_idx]["config_id"])
                        if len(selected_ids) == self.n_models:
                            break
                round_idx += 1

        else:
            raise ValueError("Either max_pareto_rank or n_models must be specified in silo mode.")

        return selected_ids
        
        
class TopNFilter(EnsemblingFilter):
    def __init__(self, n):
        self.n = n
        super().__init__("Top N Filter")
        
    def filter(self, ensembling_predictions : EnsemblingPredictions, base_model_pool : BaseModelPool):
        len_before = len(ensembling_predictions.results)
        config_ids = ensembling_predictions.get_config_ids()
        
        bmp_res = base_model_pool.results[base_model_pool.results["config_id"].isin(config_ids)]
        metric_name = ensembling_predictions.objective_name
        
        mean_scores = (
            bmp_res.groupby("config_id", as_index=False)[metric_name]
            .mean()
            .sort_values(metric_name, ascending=True)
        )
        
        config_ids_to_keep = mean_scores.head(self.n)["config_id"]
        
        ensembling_predictions.results = ensembling_predictions.results[
            ensembling_predictions.results["config_id"].isin(config_ids_to_keep)
        ]
        
        len_after = len(ensembling_predictions.results)
        self.verbose_filter(len_before, len_after)
 
        
class SiloTopNFilter(EnsemblingFilter):
    def __init__(self, n):
        self.n = n
        super().__init__("Silo Top N Filter")
        
    def filter(self, ensembling_predictions: EnsemblingPredictions, base_model_pool: BaseModelPool):
        len_before = len(ensembling_predictions.results)
        config_ids = ensembling_predictions.get_config_ids()
        
        bmp_res = base_model_pool.results[base_model_pool.results["config_id"].isin(config_ids)]
        configs = base_model_pool.configs[base_model_pool.configs["config_id"].isin(config_ids)]
        bmp_res = bmp_res.merge(configs[["config_id", "model"]], on="config_id", how="left")
        
        metric_name = ensembling_predictions.objective_name
        
        mean_scores = (
            bmp_res.groupby(["model", "config_id"], as_index=False)[metric_name]
            .mean()
            .sort_values([ "model", metric_name ], ascending=True)
        )
        
        # Group by model type
        grouped = [g for _, g in mean_scores.groupby("model")]
        
        selected_configs = []
        round_idx = 0
        
        # Round-robin selection: pick top 1 from each model
        while len(selected_configs) < self.n and any(round_idx < len(g) for g in grouped):
            for g in grouped:
                if round_idx < len(g):
                    selected_configs.append(g.iloc[round_idx]["config_id"])
                    if len(selected_configs) == self.n:
                        break
            round_idx += 1
        
        # Filter results to only keep selected configs
        ensembling_predictions.results = ensembling_predictions.results[
            ensembling_predictions.results["config_id"].isin(selected_configs)
        ]
        
        len_after = len(ensembling_predictions.results)
        self.verbose_filter(len_before, len_after)

        
class NThreadsFilter(EnsemblingFilter):
    def __init__(self, max_n_threads):
        self.max_n_threads = max_n_threads
        super().__init__("Max. N Threads Filter")
        
    def filter(self, ensembling_predictions : EnsemblingPredictions, base_model_pool : BaseModelPool):
        len_before = len(ensembling_predictions.results)
        ensembling_predictions.results = ensembling_predictions.results[ensembling_predictions.results["n_threads"] <= self.max_n_threads]
        len_after = len(ensembling_predictions.results)
        self.verbose_filter(len_before, len_after)
        
class FrequencyScalingFilter(EnsemblingFilter):
    def __init__(self, setup):
        self.setup = setup
        if self.setup not in ["boost", "fs", "max_fs"]:
            raise ValueError(f"Unknown setup: {self.setup}")
        super().__init__("Frequency Scaling Filter")

        
    def filter(self, ensembling_predictions : EnsemblingPredictions, base_model_pool : BaseModelPool):
        len_before = len(ensembling_predictions.results)
        
        if self.setup == "boost":
            ensembling_predictions.results = ensembling_predictions.results[
                ensembling_predictions.results["frequency"] == "boost"
            ]
        elif self.setup == "fs":
            ensembling_predictions.results = ensembling_predictions.results[
                ensembling_predictions.results["frequency"] != "boost"
            ]
        elif self.setup == "max_fs":
            max_fs = ensembling_predictions.results[
                ensembling_predictions.results["frequency"] != "boost"
            ]["frequency"].astype(int).max()
            ensembling_predictions.results = ensembling_predictions.results[
                ensembling_predictions.results["frequency"] == str(max_fs)
            ]
            
        len_after = len(ensembling_predictions.results)
        self.verbose_filter(len_before, len_after)
        
class DiscoveryTimeFilter(EnsemblingFilter):
    def __init__(self, max_discovery_time):
        self.max_discovery_time = max_discovery_time
        super().__init__("Max. Discovery Time Filter")
        
    def filter(self, ensembling_predictions : EnsemblingPredictions, base_model_pool : BaseModelPool):
        len_before = len(ensembling_predictions.results)
        if self.max_discovery_time > 0:
            ensembling_predictions.results = ensembling_predictions.results[ensembling_predictions.results["discovery_time"] <= self.max_discovery_time]
        len_after = len(ensembling_predictions.results)
        self.verbose_filter(len_before, len_after)

        
    
    