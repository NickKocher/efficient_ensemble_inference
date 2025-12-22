from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator
from ConfigSpace.configuration_space import ConfigurationSpace

from auto_sklearn_search_space_no_freq.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from auto_sklearn_search_space_no_freq.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from auto_sklearn_search_space_no_freq.pipeline.components.data_preprocessing.rescaling.abstract_rescaling import (  # noqa: E501
    Rescaling,
)
from auto_sklearn_search_space_no_freq.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space_no_freq.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class NoRescalingComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    
    def __init__(
        self, random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        self.preprocessor: Optional[BaseEstimator] = None
    
    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "AutoSklearnPreprocessingAlgorithm":
        self.preprocessor = "passthrough"
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        return X

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "NoRescaling",
            "name": "NoRescaling",
            "handles_missing_values": False,
            "handles_nominal_values": False,
            "handles_numerical_features": True,
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            # TODO find out if this is right!
            "handles_sparse": True,
            "handles_dense": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
            "preferred_dtype": None,
        }
        
    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        # No hyperparameters for this component
        return ConfigurationSpace()
