from collections import Counter
import time
import numpy as np
import pandas as pd
import logging
import copy

from raaml.ensembling.scheduling.allotment import Task

class SimpleEnsemblePool:

    def __init__(
        self,
        config_ids,
        configs,
        model_paths,
        threads,
        frequencies,
        metric,
        scheduler,
        schedule_lists=None,
        ensembling_predictions=None, 
        weight_matrix=None
    ):
        self.config_ids = config_ids
        self.configs = configs
        self.model_paths = model_paths
        self.threads = threads
        self.frequencies = frequencies
        self.ensembling_predictions = ensembling_predictions
        
        self.scheduler = scheduler
        self.schedule_lists = schedule_lists

        self.metric = metric
        self.metric_name = metric.name
        self.opt_score_val = metric.optimum_value
        
        self.models = None
        self.y_labels = None
        self.final_results = None
                
        if weight_matrix is not None:
            self.set_weight_matrix(weight_matrix)


    def set_y_labels(self, y_labels):
        self.y_labels = y_labels


    def set_resource_providers(self, assembled, resource_providers):
        self.assembled_resource_provider = assembled
        self.resource_providers = resource_providers
        
        
    def set_final_results(self, final_results):
        self.final_results = final_results

    @staticmethod
    def compute_default_schedule_lists(config_ids, threads, frequencies, inference_times, energies, weight_matrix, scheduler):
        tasks = [Task(config_id, threads[i], frequencies[i], inference_times[i], energies[i]) for i, config_id in enumerate(config_ids)]
        tasks = np.array(tasks)
        sorted = [scheduler.sort_tasks(tasks[weight_matrix[i] > 0]) for i in range(len(weight_matrix))]
        return [[t.config_id for t in task_list] for task_list in sorted]
    

    def _filter_duplicate_ensembles(self):
        rounded_weights = np.round(self.weight_matrix, 10)
        _, indices = np.unique(rounded_weights, axis=0, return_index=True)
        indices = np.sort(indices)
        self.weight_matrix = self.weight_matrix[indices,:]   
        self.schedule_lists = [self.schedule_lists[i] for i in indices]     


    def _filter_unused_models(self):
        used_model_indices = np.where(self.weight_matrix.sum(axis=0) > 1e-12)[0]
        self.weight_matrix = self.weight_matrix[:, used_model_indices]
        self.model_paths = [self.model_paths[i] for i in used_model_indices]
        self.configs = [self.configs[i] for i in used_model_indices]
        self.config_ids = [self.config_ids[i] for i in used_model_indices]
        self.threads = [self.threads[i] for i in used_model_indices]
        self.frequencies = [self.frequencies[i] for i in used_model_indices]
    
    
    def set_weight_matrix(self, weight_matrix):
        if not isinstance(weight_matrix, np.ndarray):
            weight_matrix = np.array(weight_matrix)
        
        assert np.all(np.isfinite(weight_matrix)), "Weight matrix contains NaN or Inf values."
        assert weight_matrix.ndim == 2, f"Weight matrix must be a 2D numpy array (Got {weight_matrix.ndim}D instead)."
        assert len(self.configs) == weight_matrix.shape[1], f"Number of configs ({len(self.configs)}) must match the number of columns ({weight_matrix.shape[1]}) in the weight matrix."
        assert len(self.model_paths) == weight_matrix.shape[1], f"Number of model paths ({len(self.model_paths)}) must match the number of columns ({weight_matrix.shape[1]}) in the weight matrix."
        assert np.all(weight_matrix >= 0), "Weight matrix must not contain negative values."
        
        self.weight_matrix = weight_matrix
        
        self._filter_duplicate_ensembles()
        self._filter_unused_models()
    
    
    def __len__(self):
        """
        Get the number of ensembles in the pool.
        
        Returns
        -------
        int
            Number of ensembles in the pool.
        """
        
        if self.weight_matrix is None:
            raise ValueError("Weight matrix must be set before getting the length of the ensemble pool.")
        
        return len(self.weight_matrix)
    
    
    def score(self, X, y):
        predictions = []
        resource_usage = np.zeros((self.weight_matrix.shape[1], len(self.assembled_resource_provider)))
        scores = []
        
        resource_names = self.assembled_resource_provider.get_names()
                
        for i, (config_id, n_cores, frequency) in enumerate(zip(self.config_ids, self.threads, self.frequencies)):
            preds = self.final_results.get_predictions(config_id).squeeze()
            predictions.append(preds)
            objectives = self.final_results.get_objectives(config_id, n_cores, frequency, [self.metric.name, *resource_names])
            resource_usage[i] = objectives[resource_names]
            scores.append(objectives[self.metric.name])
        
        predictions = np.array(predictions)
            
        res = pd.DataFrame(columns=[self.metric_name] + self.assembled_resource_provider.get_names())   
            
        for i in range(len(self)):
            ensemble_pred = self._weighted_ensemble(predictions, self.weight_matrix[i])
            ensemble_res = (resource_usage * (self.weight_matrix[i, :, None] > 1e-12)).sum(axis=0)
            score = self.metric(y, ensemble_pred)
            res.loc[i] = [score] + list(ensemble_res)
            
        return res
    
    
    def copy(self):
        return copy.deepcopy(self)
    
    
    def extend_by_allotment(self, allotment, scheduler, filters=None):
        start = time.time()
        if filters is None:
            filters = []
            
        for f in filters:
            self.ensembling_predictions.apply_filter(f, base_model_pool=None)
            
                    
        c_id_weight_map = {}
        assert len(np.unique(self.config_ids)) == len(self.config_ids), "Config IDs must be unique to extend by allotment."
        for i, c_id in enumerate(self.config_ids):
            c_id_weight_map[c_id] = i
            
        weight_matrix_indexed = []
        triplet_sets = []
        old_ensemble_indices = []
        
        for i in range(len(self)):
            mask = np.abs(self.weight_matrix[i]) > 1e-12
            config_ids = [self.config_ids[j] for j in np.where(mask)[0]]
            allotments = allotment.compute_allotments(
                self.ensembling_predictions,
                scheduler,
                config_ids
            )
            for allot in allotments:
                weight_vector = []
                for c_id in allot:
                    triplet = (c_id, *allot[c_id])
                    if triplet not in triplet_sets:
                        triplet_sets.append(triplet)
                    idx = triplet_sets.index(triplet)
                    weight_vector.append((idx, self.weight_matrix[i, c_id_weight_map[c_id]]))
                weight_matrix_indexed.append(weight_vector)
                old_ensemble_indices.append(i)
        
        config_ids, n_cores_list, frequencies = zip(*triplet_sets)
        config_paths = [self.model_paths[self.config_ids.index(c_id)] for c_id in config_ids]
        configs = [self.configs[self.config_ids.index(c_id)] for c_id in config_ids]
        weight_matrix = np.zeros((len(weight_matrix_indexed), len(triplet_sets)))
        for i, wv in enumerate(weight_matrix_indexed):
            for idx, weight in wv:
                weight_matrix[i, idx] = weight

        inf_times, energies = [], []
        for c_id, n_cores, frequency in zip(config_ids, n_cores_list, frequencies):
            inf_time, energy = self.ensembling_predictions.get_resources(c_id, n_cores, frequency, ["inference_time", "energy_consumption"])
            inf_times.append(inf_time)
            energies.append(energy)

        schedule_lists = SimpleEnsemblePool.compute_default_schedule_lists(
            config_ids,
            n_cores_list,
            frequencies,
            inf_times,
            energies,
            weight_matrix,
            scheduler
        )        
                
        ep = SimpleEnsemblePool(
            config_ids=list(config_ids),
            configs=configs,
            model_paths=config_paths,
            threads=n_cores_list,
            frequencies=list(frequencies),
            metric=self.metric,
            schedule_lists=schedule_lists,
            scheduler=scheduler,
            ensembling_predictions=self.ensembling_predictions,
            weight_matrix=weight_matrix
        )
        
        ep.set_final_results(self.final_results)
        ep.set_y_labels(self.y_labels)
        if hasattr(self, "assembled_resource_provider"):
            ep.set_resource_providers(self.assembled_resource_provider, self.resource_providers)
            
        logging.info(f"Extended ensemble pool by allotment in {time.time() - start:.2f} seconds. {len(self)} => {len(ep)} ensembles.")    
        
        return ep, old_ensemble_indices
                    
            
                        
            
            
            
            
        
        
    
    
    def _weighted_ensemble(self, predictions, weights):
        if predictions.ndim == 2:
            return weights @ predictions / weights.sum()
        elif predictions.ndim == 3:
            return np.einsum("m,msc->sc", weights, predictions) / weights.sum()
        else:
            raise ValueError("Predictions must be a 2D or 3D numpy array.")
    
    
    def print_ensemble_results(self, X, y, print_options = {}):
        """
        Print the results of each ensemble in the pool.
        
        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        """        
        res = self.score(X, y)
        
        res_ext = res.copy()
        res_ext["ensemble_size"] = [np.sum(self.weight_matrix[i] > 1e-12) for i in range(len(self))]
        res_ext["ensemble_members"] = [",".join([str(j) for j in range(self.weight_matrix.shape[1]) if self.weight_matrix[i, j] > 1e-12]) for i in range(len(self))]

        print("Ensemble Results:")
        with pd.option_context(
            'display.max_rows', print_options.get('max_rows', None),
            'display.width', print_options.get('width', 500),
            'display.max_columns', print_options.get('max_columns', None)
        ):
            print(res_ext)
        print()
        print("Models:") 
        for i in range(self.weight_matrix.shape[1]):
            print(f"Model {i}: n_threads = {self.threads[i]}, frequency = {self.frequencies[i]}, path = {self.model_paths[i]}")
        print()
        return res_ext
        
        
    def __add__(self, other):
        return SimpleEnsemblePool.join_ensemble_pools(self, other)    
        
        
    @staticmethod
    def join_ensemble_pools(pool1, pool2, scheduler = None):
        if pool1.metric != pool2.metric:
            raise ValueError("Metrics of the two pools must be the same.")
        #if pool1.ensembling_predictions != pool2.ensembling_predictions:
        #    raise ValueError("Ensembling predictions of the two pools must be the same.")
        if pool1.scheduler != pool2.scheduler and scheduler is None:
            raise ValueError("Schedulers of the two pools must be the same or a common scheduler must be provided.")
        
        def find_duplicates(setups):
            counts = Counter(setups)
            return [setup for setup, count in counts.items() if count > 1]
        
        setups1 = list(zip(pool1.config_ids, pool1.threads, pool1.frequencies))
        setups2 = list(zip(pool2.config_ids, pool2.threads, pool2.frequencies))
        
        assert len(set(setups1)) == len(setups1), f"Duplicate setups in pool 1: {find_duplicates(setups1)} duplicates"
        assert len(set(setups2)) == len(setups2), f"Duplicate setups in pool 2: {find_duplicates(setups2)} duplicates"
                      
        unified_setups = list(set(setups1).union(set(setups2)))
                
        # Map from setup → index
        setup_to_index = {s: i for i, s in enumerate(unified_setups)}
        
        # Extract combined fields
        combined_config_ids = []
        combined_configs = []
        combined_model_paths = []
        combined_threads = []
        combined_frequencies = []
        
        for s in unified_setups:
            if s in setups1:
                idx = setups1.index(s)
                combined_config_ids.append(pool1.config_ids[idx])
                combined_configs.append(pool1.configs[idx])
                combined_model_paths.append(pool1.model_paths[idx])
                combined_threads.append(pool1.threads[idx])
                combined_frequencies.append(pool1.frequencies[idx])
            else:
                idx = setups2.index(s)
                combined_config_ids.append(pool2.config_ids[idx])
                combined_configs.append(pool2.configs[idx])
                combined_model_paths.append(pool2.model_paths[idx])
                combined_threads.append(pool2.threads[idx])
                combined_frequencies.append(pool2.frequencies[idx])
        
        n_models = len(unified_setups)
        
        # Expand pool1 weights
        w1_expanded = np.zeros((pool1.weight_matrix.shape[0], n_models))
        for i, s in enumerate(setups1):
            j = setup_to_index[s]
            w1_expanded[:, j] = pool1.weight_matrix[:, i]
        
        # Expand pool2 weights
        w2_expanded = np.zeros((pool2.weight_matrix.shape[0], n_models))
        for i, s in enumerate(setups2):
            j = setup_to_index[s]
            w2_expanded[:, j] = pool2.weight_matrix[:, i]
        
        # Combine
        new_weight_matrix = np.vstack([w1_expanded, w2_expanded])
        new_schedule_lists = [*pool1.schedule_lists, *pool2.schedule_lists]
        
        # Construct combined pool
        new_pool = SimpleEnsemblePool(
            config_ids=combined_config_ids,
            configs=combined_configs,
            model_paths=combined_model_paths,
            threads=combined_threads,
            frequencies=combined_frequencies,
            metric=pool1.metric,
            schedule_lists=new_schedule_lists,
            scheduler=pool1.scheduler if scheduler is None else scheduler,
            ensembling_predictions=pool1.ensembling_predictions,
            weight_matrix=new_weight_matrix
        )
        
        # Optionally copy shared attributes
        if pool1.final_results is not None:
            new_pool.set_final_results(pool1.final_results)
        if pool1.y_labels is not None:
            new_pool.set_y_labels(pool1.y_labels)
        if hasattr(pool1, "assembled_resource_provider"):
            new_pool.set_resource_providers(pool1.assembled_resource_provider, pool1.resource_providers)
        
        return new_pool
