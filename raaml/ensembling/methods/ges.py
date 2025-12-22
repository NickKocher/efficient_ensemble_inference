# Code Taken from here with adaptions to be usable:
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.utils import check_random_state

from ConfigSpace import Configuration

from raaml.ensembling.ensemble_pool import SimpleEnsemblePool
from raaml.ensembling.methods.ensembling_method import AbstractEnsemblingStrategy
from raaml.util.metrics import RAAMLMetric
from raaml.util.model_paths import get_final_model_paths


class GreedyEnsembleSelection(AbstractEnsemblingStrategy):

    def __init__(
        self,
        n_iterations: int,
        metric: RAAMLMetric,
        n_jobs: int = -1,
        random_state: int | np.random.RandomState | None = None,
        use_best: bool = True,
    ) -> None:

        self.ensemble_size = n_iterations
        self.metric = metric
        self.use_best = use_best

        # -- Code for multiprocessing
        if (n_jobs == 1) or (os.name == "nt"):
            self._use_mp = False
        else:
            if n_jobs == -1:
                n_jobs = len(os.sched_getaffinity(0))
            self._n_jobs = n_jobs
            self._use_mp = True

        self.random_state = random_state
        

    def ensemble_fit(
        self,
        y_true,
        ensembling_predictions,
        base_model_pool,
        model_base_dir,
        do_allotment,
        scheduler
    ) -> SimpleEnsemblePool:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        if not isinstance(self.metric, RAAMLMetric):
            raise ValueError(
                "The provided metric must be an instance of a RAAMLMetric, "
                f"nevertheless it is {self.metric}({type(self.metric)})"
            )
            
        config_ids, predictions, resource_consumption, threads, frequencies = ensembling_predictions.get_ensembling_data(do_allotment)
        
        resource_names = ensembling_predictions.resource_names
        inf_time_index = resource_names.index("inference_time")
        energy_index = resource_names.index("energy_consumption")
        inference_times, energies = resource_consumption[:, inf_time_index], resource_consumption[:, energy_index]
        
        #resource_names = ensembling_predictions.resource_names
        configs = base_model_pool.get_configs(config_ids, refit_success=False)
        model_paths = get_final_model_paths(model_base_dir, config_ids, ensembling_predictions.method)

        
        self._fit(predictions, y_true)
        self.apply_use_best()
        self._calculate_final_weights()

        # -- Set metadata correctly
        self.iteration_batch_size_ = len(predictions)
        
        schedule_lists = SimpleEnsemblePool.compute_default_schedule_lists(
            config_ids,
            threads, 
            frequencies,
            inference_times,
            energies,
            self.weight_matrix,
            scheduler
        )

        return SimpleEnsemblePool(
            config_ids,
            configs,
            model_paths,
            threads,
            frequencies,
            self.metric,
            scheduler,
            schedule_lists,
            ensembling_predictions,
            self.weight_matrix
        )
    

    def _fit(self, predictions: list[np.ndarray], labels: np.ndarray) -> None:
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)
        rand = check_random_state(self.random_state)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []  # contains iteration best
        self.val_loss_over_iterations_ = []  # contains overall best
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )

        for _i in range(ensemble_size):
            # Process Iteration Solutions
            if self._use_mp:
                losses = self._compute_losses_mp(
                    weighted_ensemble_prediction, labels, predictions, len(ensemble)
                )
            else:
                losses = np.zeros((len(predictions)), dtype=np.float64)

                for j, pred in enumerate(predictions):
                    # Calculate fant_ensemble_prediction as the average of the current ensemble plus the new prediction
                    np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
                    np.multiply(
                        fant_ensemble_prediction,
                        (1.0 / float(len(ensemble) + 1)),
                        out=fant_ensemble_prediction,
                    )

                    losses[j] = self.metric(labels, fant_ensemble_prediction, to_loss=True)

            if np.all(np.isnan(losses)):
                raise ValueError("All losses are NaN. Check predictions or labels for validity.")

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
            best = rand.choice(all_best)
            ensemble_loss = losses[best]

            ensemble.append(predictions[best])
            order.append(best)

            # Update the weighted_ensemble_prediction to include the best new prediction
            np.add(
                weighted_ensemble_prediction, predictions[best], out=weighted_ensemble_prediction
            )

            trajectory.append(ensemble_loss)

            if (
                not self.val_loss_over_iterations_
                or self.val_loss_over_iterations_[-1] > ensemble_loss
            ):
                self.val_loss_over_iterations_.append(ensemble_loss)
            else:
                self.val_loss_over_iterations_.append(self.val_loss_over_iterations_[-1])

            # Handle special cases
            if len(predictions) == 1:
                break

            if ensemble_loss == 0:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        
        
    def _compute_losses_mp(self, weighted_ensemble_prediction, labels, predictions, s):
        # -- Process Iteration Solutions
        func_args = (weighted_ensemble_prediction, labels, s, self.metric, predictions)
        pred_i_list = list(range(len(predictions)))

        with ProcessPoolExecutor(self._n_jobs, initializer=_pool_init, initargs=func_args) as ex:
            results = ex.map(_init_wrapper_evaluate_single_solution, pred_i_list)

        return np.array(list(results))
    

    def apply_use_best(self):
        if self.use_best:
            # Basically from autogluon the code
            min_score = np.min(self.trajectory_)
            idx_best = self.trajectory_.index(min_score)
            self.indices_ = self.indices_[: idx_best + 1]
            self.trajectory_ = self.trajectory_[: idx_best + 1]
            self.ensemble_size = idx_best + 1
            self.validation_loss_ = self.trajectory_[idx_best]
        else:
            self.validation_loss_ = self.trajectory_[-1]
            self.val_loss_over_iterations_ = self.trajectory_
            

    def _calculate_final_weights(self) -> None:
        # Use ensemble history to create pool of ensembles 
        weight_matrix = np.zeros(
            (self.ensemble_size, self.num_input_models_),
        )
        
        for i in range(self.ensemble_size):
            ensemble_members = Counter(self.indices_[:i + 1]).most_common()
            weights = np.zeros(
                (self.num_input_models_,),
                dtype=np.float64,
            )
            for ensemble_member in ensemble_members:
                weight = float(ensemble_member[1]) / self.ensemble_size
                weights[ensemble_member[0]] = weight

            if np.sum(weights) < 1:
                weights = weights / np.sum(weights)
            
            weight_matrix[i, :] = weights

        self.weight_matrix = weight_matrix



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
