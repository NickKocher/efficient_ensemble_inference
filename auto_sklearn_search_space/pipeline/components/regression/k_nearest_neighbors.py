from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
)

from auto_sklearn_search_space.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space.pipeline.components.base import AutoSklearnRegressionAlgorithm
from auto_sklearn_search_space.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq


class KNearestNeighborsRegressor(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_neighbors, weights, p, frequency, random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.frequency = frequency
        self.random_state = random_state

    def fit(self, X, y):
        import sklearn.neighbors

        self.n_neighbors = int(self.n_neighbors)
        self.p = int(self.p)

        self.estimator = sklearn.neighbors.KNeighborsRegressor(
            n_neighbors=self.n_neighbors, weights=self.weights, p=self.p
        )

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        with SetFreq(self.frequency):
            return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "KNN",
            "name": "K-Nearest Neighbor Classification",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1
        )
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform"
        )
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        
        frequency = UniformFloatHyperparameter(
            name="frequency",
            lower=get_overall_min_freq(),
            upper=get_overall_max_freq(),
            default_value=get_overall_max_freq(),
            log=False,
        )

        cs.add_hyperparameters([n_neighbors, weights, p, frequency])

        return cs
