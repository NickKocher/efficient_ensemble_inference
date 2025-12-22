from typing import Dict, Optional, Tuple, Union

import numpy as np
import sklearn.feature_selection
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from auto_sklearn_search_space.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from auto_sklearn_search_space.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from auto_sklearn_search_space.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq


class VarianceThreshold(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self, frequency = None, random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        # VarianceThreshold does not support fit_transform (as of 0.19.1)!
        self.frequency = frequency
        self.random_state = random_state

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "VarianceThreshold":
        self.preprocessor = sklearn.feature_selection.VarianceThreshold(threshold=0.0)
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        with SetFreq(self.frequency):
            return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "Variance Threshold",
            "name": "Variance Threshold (constant feature removal)",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        
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
