from raaml.config_spaces.config_space_factory import PipelineFactory
from auto_sklearn_search_space.configurations import create_configuration_space, load_algorithm_with_hyperparameters
from auto_sklearn_search_space_no_freq.configurations import create_configuration_space as create_configuration_space_no_freq, load_algorithm_with_hyperparameters as load_algorithm_with_hyperparameters_no_freq

class AutoSklearnPipelineFactoryFrequencyScaling(PipelineFactory):
    """
    Provides configuration spaces for Auto-sklearn.
    """        
    
    def __init__(self, X, y, meta, include=None, exclude=None):
        """
        Initializes the AutoSklearnPipelineFactory.
        
        X: Feature matrix.
        y: Target vector.
        meta: Meta information about the dataset.
        include: List of algorithms to include in the configuration space.
        exclude: List of algorithms to exclude from the configuration space.
        """
        super().__init__(X, y, meta, include, exclude)
        
        self.space, self.props = create_configuration_space(
            self.meta["task_type"],
            self.meta["feature_types"],
            X, y,
            self.include, self.exclude
        )

    def get_config_space(self, seed=None):
        self.space.seed(seed)
        return self.space
    
    def load_pipeline(self, config, seed):
        # Load the algorithm with the given configuration
        return load_algorithm_with_hyperparameters(
            self.meta["task_type"],
            self.meta["feature_types"],
            config,
            self.props,
            seed,
            self.include,
            self.exclude
        )


class AutoSklearnPipelineFactory(PipelineFactory):
    """
    Provides configuration spaces for Auto-sklearn.
    """
    
    def __init__(self, X, y, meta, include=None, exclude=None):
        """
        Initializes the AutoSklearnPipelineFactory.
        
        X: Feature matrix.
        y: Target vector.
        meta: Meta information about the dataset.
        include: List of algorithms to include in the configuration space.
        exclude: List of algorithms to exclude from the configuration space.
        """
        super().__init__(X, y, meta, include, exclude)
        
        self.space, self.props = create_configuration_space_no_freq(
            self.meta["task_type"],
            self.meta["feature_types"],
            X, y,
            self.include, self.exclude
        )

    def get_config_space(self, seed=None):
        self.space.seed(seed)
        return self.space
    
    def load_pipeline(self, config, seed):
        # Load the algorithm with the given configuration
        return load_algorithm_with_hyperparameters_no_freq(
            self.meta["task_type"],
            self.meta["feature_types"],
            config,
            self.props,
            seed,
            self.include,
            self.exclude
        )