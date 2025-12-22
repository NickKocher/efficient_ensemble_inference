from abc import ABC, abstractmethod

from raaml.ensembling.ensemble_pool import SimpleEnsemblePool
import numpy as np


class AbstractEnsemblingStrategy(ABC):
    """
    Abstract base class for ensembling methods.
    """

    @abstractmethod
    def ensemble_fit(
        self, 
        y_true, 
        ensembling_predictions, 
        base_model_pool, 
        model_base_dir, 
        do_allotment,
        scheduler
    ) -> SimpleEnsemblePool:
        pass