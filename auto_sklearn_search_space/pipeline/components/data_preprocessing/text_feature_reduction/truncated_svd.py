from typing import Dict, Optional, Tuple, Union

import ConfigSpace.hyperparameters as CSH
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.decomposition import TruncatedSVD

from auto_sklearn_search_space.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from auto_sklearn_search_space.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from auto_sklearn_search_space.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq


class TextFeatureReduction(AutoSklearnPreprocessingAlgorithm):
    """
    Reduces the features created by a bag of words encoding
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        frequency : Optional[float] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.n_components = n_components
        self.frequency = frequency
        self.random_state = random_state

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "TextFeatureReduction":
        if X.shape[1] > self.n_components:
            self.preprocessor = TruncatedSVD(
                n_components=self.n_components, random_state=self.random_state
            )
            self.preprocessor.fit(X)
        elif X.shape[1] <= self.n_components and X.shape[1] != 1:
            self.preprocessor = TruncatedSVD(
                n_components=X.shape[1] - 1, random_state=self.random_state
            )
            self.preprocessor.fit(X)
        elif X.shape[1] == 1:
            self.preprocessor = "passthrough"
        else:
            raise ValueError(
                "The text embedding consists only of a single dimension.\n"
                "Are you sure that your text data is necessary?"
            )
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        elif self.preprocessor == "passthrough":
            return X
        else:
            with SetFreq(self.frequency):
                return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "TextFeatureReduction",
            "name": "TextFeatureReduction",
            "handles_missing_values": True,
            "handles_nominal_values": True,
            "handles_numerical_features": True,
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
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
            "preferred_dtype": None,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        
        n_components = CSH.UniformIntegerHyperparameter(
            "n_components", lower=1, upper=10000, default_value=100, log=True
        )
        
        frequency = CSH.UniformFloatHyperparameter(
            name="frequency",
            lower=get_overall_min_freq(),
            upper=get_overall_max_freq(),
            default_value=get_overall_max_freq(),
            log=False,
        )
        
        cs = ConfigurationSpace()
        
        cs.add_hyperparameters([n_components, frequency])
        return cs
