from abc import ABC, abstractmethod


class PipelineFactory(ABC):
    """
    Abstract base class for providing configuration spaces and loading pipelines from a configuration.
    """
    
    @abstractmethod
    def __init__(self, X, y, meta, include=None, exclude=None):
        """
        Initializes the PipelineFactory.
        """
        self.meta = meta
        self.include = include
        self.exclude = exclude

    @abstractmethod
    def get_config_space(self, seed=None):
        """
        X: Feature matrix.
        y: Target vector.
        meta: Meta information about the dataset.
        
        Returns the configuration space.
        """
        pass
    
    @abstractmethod
    def load_pipeline(self, config, seed):
        """
        Loads the algorithm with the given configuration.
        
        config: Configuration to load.
        seed: Random seed for reproducibility.
        
        Returns the loaded algorithm with
        """
        pass
