from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from auto_sklearn_search_space.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from auto_sklearn_search_space.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq


class Densifier(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, frequency, random_state=None):
        self.frequency = frequency

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        from scipy import sparse

        with SetFreq(self.frequency):
            if sparse.issparse(X):
                return X.todense().getA()
            else:
                return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "RandomTreesEmbedding",
            "name": "Random Trees Embedding",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, UNSIGNED_DATA),
            "output": (DENSE, INPUT),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        frequency = UniformFloatHyperparameter(
            name="frequency",
            lower=get_overall_min_freq(),
            upper=get_overall_max_freq(),
            default_value=get_overall_max_freq(),
            log=False,
        )
        
        cs = ConfigurationSpace()
        
        cs.add_hyperparameter(frequency)
        
        return cs
