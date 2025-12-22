# Code Taken from here with adaptions to be usable:
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py
from __future__ import annotations

from gc import enable
from operator import is_
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn import ensemble
from sklearn.utils import check_random_state

from ConfigSpace import Configuration

from raaml.ensembling import ensembling_predictions
from raaml.ensembling.ensemble_pool import SimpleEnsemblePool
from raaml.ensembling.methods.ensembling_method import AbstractEnsemblingStrategy
from raaml.util.metrics import RAAMLMetric
from raaml.util.model_paths import get_final_model_paths
from raaml.ensembling.scheduling.allotment import Task

from pymoo.indicators.hv import HV


class MOGreedyEnsembleSelection(AbstractEnsemblingStrategy):

    def __init__(
        self,
        n_iterations: int,
        metric: RAAMLMetric,
        enable_reset=True,
        restriction_mode="quadratic",
        filter_duplicates=True,
        exp_factor = 4.0,
        max_ensemble_size : int = 20,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:

        self.n_iterations = n_iterations
        self.metric = metric
        self.max_ensemble_size = max_ensemble_size
        self.random_state = random_state
        self.enable_reset = enable_reset
        self.restriction_mode = restriction_mode
        self.filter_duplicates = filter_duplicates
        self.exp_factor = exp_factor
        
        if self.max_ensemble_size < 2:
            raise ValueError("The maximum ensemble size must be at least 2")
        
        if self.restriction_mode not in ["linear", "quadratic", "none", "exponential"]:
            raise ValueError(
                "The restriction mode must be one of 'linear', 'quadratic', "
                f"'exponential' or 'none', but is '{self.restriction_mode}'"
            )
            
        

    def ensemble_fit(
        self,
        y_true,
        ensembling_predictions,
        base_model_pool,
        model_base_dir,
        do_allotment,
        scheduler
    ) -> SimpleEnsemblePool:
        self.n_iterations = int(self.n_iterations)
        if self.n_iterations < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        if not isinstance(self.metric, RAAMLMetric):
            raise ValueError(
                "The provided metric must be an instance of a RAAMLMetric, "
                f"nevertheless it is {self.metric}({type(self.metric)})"
            )
            
        config_ids, predictions, resource_consumptions, threads, frequencies = ensembling_predictions.get_ensembling_data(do_allotment)
        
        #resource_names = ensembling_predictions.resource_names
        configs = base_model_pool.get_configs(config_ids, refit_success=False)
        model_paths = get_final_model_paths(model_base_dir, config_ids, ensembling_predictions.method)
        
        resource_names = ensembling_predictions.resource_names
        inf_time_index = resource_names.index("inference_time")
        energy_index = resource_names.index("energy_consumption")
        inference_times, energies = resource_consumptions[:, inf_time_index], resource_consumptions[:, energy_index]

        self._fit(predictions, y_true, inference_times, energies, threads, frequencies, config_ids, scheduler)
        self._calculate_final_weights()

        # -- Set metadata correctly
        self.iteration_batch_size_ = len(predictions)

        return SimpleEnsemblePool(
            config_ids,
            configs,
            model_paths,
            threads,
            frequencies,
            self.metric,
            scheduler,
            self.schedule_lists,
            ensembling_predictions,
            self.weight_matrix
        )
    
    
    def _is_multiple(self, arr1, arr2):
        assert arr1.dtype == np.int32 and arr2.dtype == np.int32, "Both arrays must be of type np.int32"

        # Check if zero positions match
        if not np.array_equal(arr1 == 0, arr2 == 0):
            return False

        # Mask out zero elements
        mask = arr2 != 0
        if not np.any(mask):  # all zeros
            return True

        # Compute ratio only once
        ratio = arr1[mask] / arr2[mask]
        return np.allclose(ratio, ratio[0])
    
    
    def _check_loop(self, weight_matrix, weights):
        for i in range(len(weight_matrix)):
            if self._is_multiple(weight_matrix[i], weights):
                return True
        return False
    

    def _fit(self,
             predictions: list[np.ndarray],
             labels: np.ndarray,
             inference_times,
             energies,
             threads,
             frequencies,
             config_ids,
             scheduler
        ) -> None:
        
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)
        rand = check_random_state(self.random_state)
        
        assert (
            len(predictions) == len(inference_times) and
            len(predictions) == len(energies) and 
            len(predictions) == len(threads) and 
            len(predictions) == len(frequencies)
        ), "Configs, inference times and energies should have the same length"
        
        ensemble_weights = np.zeros((len(predictions)), dtype=np.int32)
        trajectory = []  # contains iteration best
        
        tasks = []
        for i, c_id in enumerate(config_ids):
            task = Task(c_id, threads[i], frequencies[i], inference_times[i], energies[i])
            tasks.append(task)
        tasks = np.array(tasks)

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )
        
        objectives_history = np.zeros((self.n_iterations, 3))
        
        weight_matrix = []
        schedule_lists = []

        for i in range(self.n_iterations):
            progress = (i + 1) / self.n_iterations
            if self.restriction_mode == "linear":
                max_n = progress * (self.max_ensemble_size - 2) + 2
            elif self.restriction_mode == "quadratic":
                max_n = (progress)**2 * (self.max_ensemble_size - 2) + 2
            elif self.restriction_mode == "exponential":    
                exp_norm = (np.exp(self.exp_factor * progress) - 1) / (np.exp(self.exp_factor) - 1)
                max_n = 2 + exp_norm * (self.max_ensemble_size - 2)
            elif self.restriction_mode == "none":
                max_n = self.max_ensemble_size
            
            cur_ensemble_size = np.sum(ensemble_weights > 0).astype(np.int32)
            
            losses = []
            fant_inf_times = []
            fant_energies = []
            fant_ensemble_weights = []
            optimization_indices = []
            
            pred_indices = list(range(len(predictions)))

            for j in pred_indices:
                pred = predictions[j]
                ew = ensemble_weights.copy()
                ew[j] += 1
                
                is_loop = self.filter_duplicates and self._check_loop(weight_matrix, ew)
                is_too_large = (ew > 0).sum() > max_n
                if is_loop or is_too_large:
                    continue
                
                fant_ensemble_weights.append(ew)
                
                # Calculate fant_ensemble_prediction as the average of the current ensemble plus the new prediction
                np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
                np.multiply(
                    fant_ensemble_prediction,
                    (1.0 / float(np.sum(ensemble_weights) + 1)),
                    out=fant_ensemble_prediction,
                )

                losses.append(self.metric(labels, fant_ensemble_prediction, to_loss=True))
                optimization_indices.append(j)
            
            # Consider removing models if ensemble size >= 2 and reset is enabled
            if cur_ensemble_size >= 2 and self.enable_reset:    
                for j, w in enumerate(ensemble_weights):
                    if w > 0:
                        ew = ensemble_weights.copy()
                        ew[j] = 0
                        
                        is_loop = self.filter_duplicates and self._check_loop(weight_matrix, ew)
                        if is_loop:
                            continue
                        
                        fant_ensemble_weights.append(ew)
                        
                        pred = w * predictions[j]
                        np.subtract(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
                        np.multiply(
                            fant_ensemble_prediction,
                            (1.0 / float(np.sum(ensemble_weights) - w)),
                            out=fant_ensemble_prediction,
                        )

                        losses.append(self.metric(labels, fant_ensemble_prediction, to_loss=True))
                        optimization_indices.append(-j-1)
            
            if len(losses) == 0:
                continue
            
            if np.all(np.isnan(losses)):
                raise ValueError("All losses are NaN. Check predictions or labels for validity.")

            resources = [scheduler.compute_offline_resources(tasks[weights > 0]) for weights in fant_ensemble_weights]
            fant_inf_times, fant_energies, _, fant_schedules = zip(*resources)
            
            loss_concat = np.concatenate((objectives_history[:i, 0], losses))
            min_loss, max_loss = np.nanmin(loss_concat), np.nanmax(loss_concat)
            inf_time_concat = np.concatenate((objectives_history[:i, 1], fant_inf_times))
            min_inf_time, max_inf_time = np.nanmin(inf_time_concat), np.nanmax(inf_time_concat)
            energy_concat = np.concatenate((objectives_history[:i, 2], fant_energies))
            min_energy, max_energy = np.nanmin(energy_concat), np.nanmax(energy_concat)
            
            losses_norm = (losses - min_loss) / (max_loss - min_loss + 1e-8)
            fant_inf_times_norm = (fant_inf_times - min_inf_time) / (max_inf_time - min_inf_time + 1e-8)
            fant_energies_norm = (fant_energies - min_energy) / (max_energy - min_energy + 1e-8) 
            
            min = np.array([[min_loss, min_inf_time, min_energy]])
            max = np.array([[max_loss, max_inf_time, max_energy]])
            objs_history_norm = (objectives_history[:i] - min) / (max - min + 1e-8)
            objs_list = [
                np.concatenate((
                    objs_history_norm,
                    np.array([[losses_norm[i], fant_inf_times_norm[i], fant_energies_norm[i]]])
                )) for i in range(len(losses_norm))
            ]
            ind = HV(ref_point=[1] * 3)
            hypervolumes = np.array([ind(objs) for objs in objs_list])

            all_best = np.argwhere(hypervolumes == np.nanmax(hypervolumes)).flatten()
            best = rand.choice(all_best)
            
            objectives_history[i] = np.array([losses[best], fant_inf_times[best], fant_energies[best]])
            schedule_lists.append(fant_schedules[best])

            opt_index = optimization_indices[best]
            if opt_index >= 0:
                add_index = opt_index
                if len(weight_matrix) == 0:
                    new_weight_vector = np.zeros(len(predictions), dtype=np.int32)
                else:
                    new_weight_vector = weight_matrix[-1].copy()
                new_weight_vector[add_index] += 1
                weight_matrix.append(new_weight_vector)
                
                np.add(
                    weighted_ensemble_prediction, predictions[add_index], out=weighted_ensemble_prediction
                )
            else:
                remove_index = -opt_index - 1
                
                new_weight_vector = weight_matrix[-1].copy()
                old_weight = new_weight_vector[remove_index]
                new_weight_vector[remove_index] = 0
                weight_matrix.append(new_weight_vector)
                
                np.subtract(
                    weighted_ensemble_prediction,
                    predictions[remove_index] * old_weight,
                    out=weighted_ensemble_prediction,
                )
                
            ensemble_weights = np.array(new_weight_vector, dtype=np.int32)
            

            trajectory.append(hypervolumes[best])

            # Handle special cases
            if len(predictions) == 1:
                break

        self.trajectory_ = trajectory
        self.weight_matrix = weight_matrix
        self.schedule_lists = schedule_lists
        
        
    def _compute_losses_mp(self, weighted_ensemble_prediction, labels, predictions, s):
        # -- Process Iteration Solutions
        func_args = (weighted_ensemble_prediction, labels, s, self.metric, predictions)
        pred_i_list = list(range(len(predictions)))

        with ProcessPoolExecutor(self._n_jobs, initializer=_pool_init, initargs=func_args) as ex:
            results = ex.map(_init_wrapper_evaluate_single_solution, pred_i_list)

        return np.array(list(results))
           

    def _calculate_final_weights(self) -> None:
        self.weight_matrix = np.array(self.weight_matrix).astype(np.float64) / np.sum(self.weight_matrix, axis=1, keepdims=True)


def _pool_init(_weighted_ensemble_prediction, _labels, _sample_size, _score_metric, _predictions):
    global p_weighted_ensemble_prediction, p_labels, p_sample_size, p_score_metric, p_predictions
    p_weighted_ensemble_prediction = _weighted_ensemble_prediction
    p_labels = _labels
    p_sample_size = _sample_size
    p_score_metric = _score_metric
    p_predictions = _predictions


def _init_wrapper_evaluate_single_solution(pred_index):
    return evaluate_single_solution(
        p_weighted_ensemble_prediction,
        p_labels,
        p_sample_size,
        p_score_metric,
        p_predictions[pred_index],
    )


def evaluate_single_solution(weighted_ensemble_prediction, labels, sample_size, score_metric, pred):
    fant_ensemble_prediction = np.add(weighted_ensemble_prediction, pred)
    np.multiply(
        fant_ensemble_prediction, (1.0 / float(sample_size + 1)), out=fant_ensemble_prediction
    )

    return score_metric(labels, fant_ensemble_prediction, to_loss=True)
