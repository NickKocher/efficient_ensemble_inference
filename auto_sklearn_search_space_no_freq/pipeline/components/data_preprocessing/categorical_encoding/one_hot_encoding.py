from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from sklearn.preprocessing import OneHotEncoder as DenseOneHotEncoder

from auto_sklearn_search_space_no_freq.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space_no_freq.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from auto_sklearn_search_space_no_freq.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from auto_sklearn_search_space_no_freq.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from auto_sklearn_search_space_no_freq.pipeline.implementations.SparseOneHotEncoder import SparseOneHotEncoder


class OneHotEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self, random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.random_state = random_state

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "OneHotEncoder":
        if scipy.sparse.issparse(X):
            self.preprocessor = SparseOneHotEncoder()
        else:
            self.preprocessor = DenseOneHotEncoder(
                sparse_output=False, categories="auto", handle_unknown="ignore"
            )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "1Hot",
            "name": "One Hot Encoder",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            # TODO find out of this is right!
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
        
        cs = ConfigurationSpace()
        
        return cs
