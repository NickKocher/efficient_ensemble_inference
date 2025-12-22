from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from auto_sklearn_search_space.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space.pipeline.components.base import (
    AutoSklearnRegressionAlgorithm,
    IterativeComponent,
)
from auto_sklearn_search_space.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from auto_sklearn_search_space.util.common import check_for_bool, check_none
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq



class RandomForest(
    IterativeComponent,
    AutoSklearnRegressionAlgorithm,
):
    def __init__(
        self,
        criterion,
        max_features,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        bootstrap,
        max_leaf_nodes,
        min_impurity_decrease,
        frequency,
        random_state=None,
    ):
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.frequency = frequency
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 512

    def get_current_iter(self):
        return self.estimator.n_estimators

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        from sklearn.ensemble import RandomForestRegressor

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_estimators = int(self.n_estimators)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)

            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)

            self.max_features = float(self.max_features)

            self.bootstrap = check_for_bool(self.bootstrap)

            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)

            self.min_impurity_decrease = float(self.min_impurity_decrease)

            self.estimator = RandomForestRegressor(
                n_estimators=n_iter,
                criterion=self.criterion,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                bootstrap=self.bootstrap,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                n_jobs=None,
                warm_start=True,
            )
        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(
                self.estimator.n_estimators, self.n_estimators
            )

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        self.estimator.fit(X, y)
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False

        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        with SetFreq(self.frequency):
            return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "RF",
            "name": "Random Forest Regressor",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "prefers_data_normalized": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        criterion = CategoricalHyperparameter(
            "criterion", ["squared_error", "friedman_mse", "absolute_error"]
        )

        # In contrast to the random forest classifier we want to use more max_features
        # and therefore have this not on a sqrt scale
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1.0
        )

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True"
        )
        
        frequency = UniformFloatHyperparameter(
            name="frequency",
            lower=get_overall_min_freq(),
            upper=get_overall_max_freq(),
            default_value=get_overall_max_freq(),
            log=False,
        )

        cs.add_hyperparameters(
            [
                criterion,
                max_features,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                min_impurity_decrease,
                bootstrap,
                frequency
            ]
        )

        return cs
