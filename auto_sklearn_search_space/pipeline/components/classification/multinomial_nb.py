from typing import Optional

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from auto_sklearn_search_space.askl_typing import FEAT_TYPE_TYPE
from auto_sklearn_search_space.pipeline.components.base import AutoSklearnClassificationAlgorithm
from auto_sklearn_search_space.pipeline.constants import DENSE, PREDICTIONS, SIGNED_DATA, SPARSE
from auto_sklearn_search_space.util.common import check_for_bool
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq



class MultinomialNB(AutoSklearnClassificationAlgorithm):
    def __init__(self, alpha, fit_prior, frequency, random_state=None, verbose=0):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.frequency = frequency
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, y):
        import scipy.sparse
        import sklearn.naive_bayes

        self.fit_prior = check_for_bool(self.fit_prior)
        self.alpha = float(self.alpha)
        self.n_iter = 0
        self.fully_fit_ = False
        self.estimator = sklearn.naive_bayes.MultinomialNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
        )
        self.classes_ = np.unique(y.astype(int))

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if scipy.sparse.issparse(X):
            X.data[X.data < 0] = 0.0
        else:
            X[X < 0] = 0.0

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass

            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=None
            )
        self.estimator.fit(X, y)

        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        with SetFreq(self.frequency):
            return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        with SetFreq(self.frequency):
            return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "MultinomialNB",
            "name": "Multinomial Naive Bayes classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

        # the smoothing parameter is a non-negative float
        # I will limit it to 100 and put it on a logarithmic scale. (SF)
        # Please adjust that, if you know a proper range, this is just a guess.
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1e-2, upper=100, default_value=1, log=True
        )

        fit_prior = CategoricalHyperparameter(
            name="fit_prior", choices=["True", "False"], default_value="True"
        )
        
        frequency = UniformFloatHyperparameter(
            name="frequency",
            lower=get_overall_min_freq(),
            upper=get_overall_max_freq(),
            default_value=get_overall_max_freq(),
            log=False,
        )

        cs.add_hyperparameters([alpha, fit_prior, frequency])

        return cs
