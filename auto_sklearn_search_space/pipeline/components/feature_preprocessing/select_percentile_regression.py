from typing import Optional

from functools import partial

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from auto_sklearn_search_space.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from auto_sklearn_search_space.pipeline.components.feature_preprocessing.select_percentile import (
    SelectPercentileBase,
)
from auto_sklearn_search_space.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq


class SelectPercentileRegression(
    SelectPercentileBase, AutoSklearnPreprocessingAlgorithm
):
    def __init__(self, percentile, frequency, score_func="f_regression", random_state=None):
        """Parameters:
        random state : ignored

        score_func : callable, Function taking two arrays X and y, and
                     returning a pair of arrays (scores, pvalues).
        """
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.frequency = frequency
        self.percentile = int(float(percentile))
        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info":
            self.score_func = partial(
                sklearn.feature_selection.mutual_info_regression,
                random_state=self.random_state,
            )
        else:
            raise ValueError("Don't know this scoring function: %s" % score_func)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "SPR",
            "name": "Select Percentile Regression",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }
        
    def transform(self, X):
        with SetFreq(self.frequency):
            return super().transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        percentile = UniformFloatHyperparameter(
            "percentile", lower=1, upper=99, default_value=50
        )

        score_func = CategoricalHyperparameter(
            name="score_func", choices=["f_regression", "mutual_info"]
        )
        
        frequency = UniformFloatHyperparameter(
            name="frequency",
            lower=get_overall_min_freq(),
            upper=get_overall_max_freq(),
            default_value=get_overall_max_freq(),
            log=False,
        )

        cs = ConfigurationSpace()
        cs.add_hyperparameters([percentile, score_func, frequency])
        return cs
