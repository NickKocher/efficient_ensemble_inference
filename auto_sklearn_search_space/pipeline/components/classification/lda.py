from typing import Optional

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from auto_sklearn_search_space.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space.pipeline.components.base import AutoSklearnClassificationAlgorithm
from auto_sklearn_search_space.pipeline.constants import DENSE, PREDICTIONS, UNSIGNED_DATA
from auto_sklearn_search_space.pipeline.implementations.util import softmax
from auto_sklearn_search_space.util.common import check_none
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq



class LDA(AutoSklearnClassificationAlgorithm):
    def __init__(self, shrinkage, tol, frequency, shrinkage_factor=0.5, random_state=None):
        self.shrinkage = shrinkage
        self.tol = tol
        self.frequency = frequency
        self.shrinkage_factor = shrinkage_factor
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.discriminant_analysis
        import sklearn.multiclass

        if check_none(self.shrinkage):
            self.shrinkage_ = None
            solver = "svd"
        elif self.shrinkage == "auto":
            self.shrinkage_ = "auto"
            solver = "lsqr"
        elif self.shrinkage == "manual":
            self.shrinkage_ = float(self.shrinkage_factor)
            solver = "lsqr"
        else:
            raise ValueError(self.shrinkage)

        self.tol = float(self.tol)

        estimator = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            shrinkage=self.shrinkage_, tol=self.tol, solver=solver
        )

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=None)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        with SetFreq(self.frequency):
            return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        
        with SetFreq(self.frequency):
            df = self.estimator.predict_proba(X)
            return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "LDA",
            "name": "Linear Discriminant Analysis",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        shrinkage = CategoricalHyperparameter(
            "shrinkage", ["None", "auto", "manual"], default_value="None"
        )
        shrinkage_factor = UniformFloatHyperparameter("shrinkage_factor", 0.0, 1.0, 0.5)
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True
        )
        
        frequency = UniformFloatHyperparameter(
            name="frequency",
            lower=get_overall_min_freq(),
            upper=get_overall_max_freq(),
            default_value=get_overall_max_freq(),
            log=False,
        )
        
        cs.add_hyperparameters([shrinkage, shrinkage_factor, tol, frequency])

        cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))
        return cs
