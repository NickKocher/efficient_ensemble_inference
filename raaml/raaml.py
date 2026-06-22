from time import time_ns, time as time_s
import os
from mosmac3.smac.acquisition.function.expected_hypervolume import PHVI
from mosmac3.smac import Scenario, HyperparameterOptimizationFacade as HPOFacade
from mosmac3.smac.facade.multi_objective_facade import MultiObjectiveFacade
from mosmac3.smac.initial_design import RandomInitialDesign
import numpy as np
import yaml
import pickle
import random
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import shutil
from pynisher import limit, MemoryLimitException
import traceback
import logging
import warnings

from mosmac3.smac.utils.configspace import get_config_hash
from raaml.resource_provider import AssembledResourceProvider, ResourceProvider, DummyResourceProvider, InferenceTimeProvider, AMDEnergyProvider
from raaml.config_spaces.gb_mlp_space import GBMLPPipelineFactory
from raaml.base_model_generation.pd_base_model_pool import BaseModelPool, base_model_pool_from_smac_history, load_base_model_pool_from_dir

from raaml.util.warning_filter import filter_warnings
from raaml.util.metrics import get_default_metric, RAAMLMetric
from raaml.util.map_method import map_bmg_method_to_unified_name, map_ensembling_method_to_unified_name
from raaml.util.split import RAAMLDataSplit
from raaml.util.affinity_manager import AffinityManager
from raaml.util.final_results import FinalResults
from raaml.util.model_paths import get_final_model_path, filter_refit_ens_models
from raaml.ensembling.methods.ges import GreedyEnsembleSelection
from raaml.ensembling.methods.single import base_model_pool_to_ensemble_pool
from raaml.ensembling.ensembling_predictions import EnsemblingPredictions, load_ensembling_predictions
from raaml.ensembling.scheduling.scheduler import LongestProcessingTimeScheduler, HighestThreadCountScheduler, HighestWorkloadScheduler
from raaml.util.raaml_progress import exception_to_status
from raaml.util.idle_powers import get_idle_power
from frequency.freq_api_amd import get_freq_values, SetFreqMgr


class ResourceAwareAutoMLPipeline():
    """
    A class to represent the Resource-Aware AutoML Pipeline, comprising base model generation, ensembling and resource management.
    """

    def __init__(self,
        identifier : str, 
        output_dir : str = "./raaml_output",
        base_model_gen : str = "mo-bo",
        n_jobs_optimization : int = 1,
        n_cores_target_function : int = 1,
        time_search_s : int = 120,
        trial_walltime_limit_s : int = None,
        trial_memory_limit_mb : int = 16 * 1024,
        seed :int = None,
        resource_providers : dict | None = {
            "inference_time": "default",
            "energy_consumption": "default",
        },
        n_cv : int = 5,
        stratified_cv : bool = True,
        include : dict = None,
        exclude :dict = None,
        bmg_kwargs : dict = None,
        logging_level : int = logging.INFO,
        overwrite=False,
        save_init_args=True
    ):
        """
        Initialize the Resource-Aware AutoML Pipeline.
        
        Parameters
        ----------
        identifier : str
            Unique identifier for the pipeline run. Used for naming output directories and logs.
        output_dir : str, default="./raaml_output"
            Root directory where all pipeline outputs will be saved.
        base_model_gen : str, default="mo-bo"
            Base model generation strategy. Options: "mo-bo" (Multi-objective Bayesian Optimization),
            "so-bo" (Single-objective Bayesian Optimization), "nsga2" (NSGA-II), "so-asha" (Single-objective ASHA),
            "mo-asha" (Multi-objective ASHA), "random" (Random Search), or "auto" (automatic selection).
        n_jobs_optimization : int, default=1
            Number of parallel workers for hyperparameter optimization. Must be > 0.
        n_cores_target_function : int, default=1
            Number of CPU cores to allocate per target function evaluation.
        time_search_s : int, default=120
            Total time budget for hyperparameter search in seconds.
        trial_walltime_limit_s : int, optional
            Maximum wall-clock time per trial in seconds. If None, defaults to 5% of time_search_s.
        trial_memory_limit_mb : int, default=16384 (16GB)
            Memory limit per trial in megabytes.
        seed : int, optional
            Random seed for reproducibility. If None, a random seed is generated from current time.
        resource_providers : dict | None, default={"inference_time": "default", "energy_consumption": "default"}
            Dictionary mapping resource names to ResourceProvider instances or "default"/"dummy" strings.
            Required for multi-objective optimization. Examples: "inference_time", "energy_consumption".
        n_cv : int, default=5
            Number of cross-validation folds for base model evaluation.
        stratified_cv : bool, default=True
            Whether to use stratified cross-validation (recommended for classification tasks).
        include : dict, optional
            Configuration space constraints to include specific components or hyperparameters.
        exclude : dict, optional
            Configuration space constraints to exclude specific components or hyperparameters.
        bmg_kwargs : dict, optional
            Additional keyword arguments for base model generation strategy (e.g., n_trials, pop_size, eta).
        logging_level : int, default=logging.INFO
            Python logging level for console output.
        overwrite : bool, default=False
            Whether to overwrite existing output files/directories.
        save_init_args : bool, default=True
            Whether to save initialization arguments to setup.yaml for later reloading.
            
        Raises
        ------
        NotImplementedError
            If base_model_gen is set to "auto".
        ValueError
            If n_jobs_optimization <= 0.
            If base_model_gen is not a recognized strategy.
            
        Notes
        -----
        The pipeline creates a directory structure at `{output_dir}/{identifier}/` containing:
        - setup.yaml: Saved initialization arguments
        - splits/: Cross-validation data splits
        - models/: Trained model artifacts
        - base_model_pool/: Serialized base model pool
        - ensemble_pools/: Serialized ensemble pools
        - smac_output/: SMAC3 optimization logs (for BO methods)
        
        Random seeds are automatically set for numpy and Python's random module.
        """
        
        
        
        self.init_args = {**locals()}
        
        self.identifier = identifier
        self.output_dir = output_dir
        self.base_model_gen = map_bmg_method_to_unified_name(base_model_gen)
        self.n_jobs_optimization = n_jobs_optimization
        self.n_cores_target_function = n_cores_target_function
        self.config_space = None
        self.seed = seed
        self.is_classification = None
        self.time_search_s = time_search_s
        self.trial_walltime_limit_s = trial_walltime_limit_s
        self.trial_memory_limit_mb = trial_memory_limit_mb
        self.resource_providers = resource_providers
        self.is_mo = self.base_model_gen in ["mo-bo", "asha", "nsga2"]
        self.n_cv = n_cv
        self.include = include
        self.exclude = exclude
        self.bmg_kwargs = bmg_kwargs if bmg_kwargs is not None else {}
        self.stratified = stratified_cv
        self.logging_level = logging_level
        self.overwrite = overwrite
        
        # Init logging
        logging.basicConfig(level=logging_level, force=True, format="[%(levelname)s][%(filename)s:%(lineno)d] %(message)s",)

        # Check validity of parameters
        if self.base_model_gen == "auto":
            raise NotImplementedError("Automatic selection of base model generation strategy is not yet implemented.")
        
        if self.n_jobs_optimization <= 0:
            raise ValueError("Number of cores for search must be greater than 0.")
        
        if self.trial_walltime_limit_s is None:
            self.trial_walltime_limit_s = int(0.05 * self.time_search_s)
            logging.info("Trial walltime limit not set. Setting to 5 percent of total search time.")
        
        # Create base directory
        self.base_dir = os.path.join(self.output_dir, self.identifier)
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)  
        
        # Initialize attributes
        self.base_model_pool = None
        self.ensembling_predictions = None
        self.final_results = None
        self.ensemble_pool = None
        self.filtered = False
        self.raaml_progress = None
        
        # Set random seed
        if self.seed is None:
            self.seed = time_ns() % 2**32
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.init_args["seed"] = self.seed
        
        # Save initialization arguments
        if save_init_args:
            self.check_if_exists_and_delete_for_overwrite(os.path.join(self.base_dir), "base directory", True)
            self._save_init_args()
    
        
    def _init_resource_providers(self):
        """
        Initialize resource providers for multi-objective optimization.
        
        Converts resource provider specifications (strings or instances) into initialized
        ResourceProvider objects. Supports "default" and "dummy" string specifications
        as well as custom ResourceProvider instances. Creates an AssembledResourceProvider
        that combines all individual providers.
        
        For multi-objective optimization, at least one resource provider must be specified.
        
        Raises
        ------
        ValueError
            If multi-objective optimization is enabled but no resource providers are given.
            If resource_providers is not a dictionary.
            If an unknown resource provider name is requested as "default".
            If a resource provider is neither a ResourceProvider instance, "default", nor "dummy".
        
        Notes
        -----
        Default resource providers:
        - "inference_time": InferenceTimeProvider
        - "energy_consumption": AMDEnergyProvider
        
        Sets self.resource_providers (dict) and self.assembled_resource_provider (AssembledResourceProvider).
        """
        
        # Check resource providers
        if self.is_mo and (self.resource_providers is None or len(self.resource_providers) == 0):
            raise ValueError("For multi-objective optimization, at least one resource provider must be given.")

        if self.resource_providers is not None and not isinstance(self.resource_providers, dict):
            raise ValueError("Resource providers must be given in a dictionary mapping the resource names to their providers.")
                    
        # Initialize resource providers                    
        for r_name, r in self.resource_providers.items():
            if isinstance(r, str) and r.lower() == "default":                
                if r_name == "inference_time":
                    self.resource_providers[r_name] = InferenceTimeProvider(r_name)
                elif r_name == "energy_consumption":
                    self.resource_providers[r_name] = AMDEnergyProvider(r_name)
                else:
                    raise ValueError(f"Default resource provider for {r_name} not implemented.")
            elif isinstance(r, str) and r.lower() == "dummy":
                self.resource_providers[r_name] = DummyResourceProvider(r_name)
            elif isinstance(r, ResourceProvider):
                self.resource_providers[r_name] = r
            else:
                raise ValueError(f"Resource provider for {r_name} must be a ResourceProvider or \"default\". Got {type(r)} instead.")              
         
        # Create assembled resource provider   
        self.assembled_resource_provider = AssembledResourceProvider(self.resource_providers)

    
    def _check_and_init_meta(self, meta, X, y):
        """
        Check if the meta-information is valid.

        Parameters
        ----------
        meta : dict
            Meta-information about the data.
        
        Raises
        ------
        ValueError
            If the meta-information is not valid.
        """
        
        if not isinstance(meta, dict):
            raise ValueError("Meta-information must be a dictionary.")
        
        if "feature_types" not in meta:
            raise ValueError("Meta-information must contain 'feature_types' key.")
        
        if "task_type" not in meta:
            raise ValueError("Meta-information must contain 'task_type' key.")
        
        self.is_classification = meta["task_type"].lower() == "classification"
        
        if self.is_classification and "n_classes" not in meta:
            logging.info("Meta-information does not contain 'n_classes' key. Extracting number of classes from the target variable.")
            self.n_classes = len(np.unique(y))
        elif self.is_classification:
            self.n_classes = meta["n_classes"]
            self.y_unique_labels = meta["classes"]
            
        self.task_type = meta["task_type"].lower()
        
        self.feature_types = meta["feature_types"]
        
        if "task_id" in meta:
            self.task_id = meta["task_id"]
            
    def _init_cores(self):        
        # Initialize affinity manager
        self.cores = list(os.sched_getaffinity(0))
        self.affinity_manager = AffinityManager(
            self.cores,
            self.n_cores_target_function,
            os.path.join(self.base_dir, "affinity.yaml")
        )
            
    def _save_init_args(self):
        # Save initialization arguments to setup.yaml
        self.check_if_exists_and_delete_for_overwrite(os.path.join(self.base_dir, "setup.yaml"), "initialization arguments")
        
        self.init_args.pop("self")
        self.init_args["seed"] = self.seed
        with open(f"{self.output_dir}/{self.identifier}/setup.yaml", "w") as f:
            yaml.dump(self.init_args, f, default_flow_style=False)
            
    
    def _pickle_ensemble_pool(self, name):
        """
        Save the ensemble pool to a file.
        """
        
        if self.ensemble_pool is None:
            raise ValueError("Ensemble pool is not initialized. Please run fit() first.")
        
        logging.debug("Pickling ensemble pool...")
        with open(os.path.join(self.output_dir, self.identifier, "ensemble_pools", f"ensemble_pool_{name}.pkl"), "wb") as f:
            pickle.dump(self.ensemble_pool, f)
            
    
    def load_ensemble_pool(self, name):
        """
        Load the ensemble pool from a file.
        """
        
        logging.debug("Loading ensemble pool...")
        if not os.path.exists(os.path.join(self.output_dir, self.identifier, "ensemble_pools", f"ensemble_pool_{name}.pkl")):
            return None
        with open(os.path.join(self.output_dir, self.identifier, "ensemble_pools", f"ensemble_pool_{name}.pkl"), "rb") as f:
            ensemble_pool = pickle.load(f)
        return ensemble_pool
            
    def set_raaml_progress(self, raaml_progress):
        self.raaml_progress = raaml_progress
    
            
    @staticmethod
    def loadRAAML(directory, identifier, seed = None):
        # Load a previously saved RAAML pipeline from the given directory and identifier.
        base_dir = os.path.join(directory, identifier)
        
        if not os.path.exists(base_dir):
            raise ValueError(f"Directory {base_dir} does not exist.")
        
        with open(os.path.join(base_dir, "setup.yaml"), "r") as f:
            init_args = yaml.safe_load(f)
            
        if seed is not None:
            init_args["seed"] = seed
        init_args["save_init_args"] = False
        
        return ResourceAwareAutoMLPipeline(**init_args)
    
    
    def check_if_exists_and_delete_for_overwrite(self, path, name, create_dir=False):
        # Check if path exists and delete it if overwrite is True.
        exists = os.path.exists(path)
        if exists and self.overwrite:
            logging.info(f"Overwriting existing {name}. ({path}).")
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
                os.makedirs(path)
        elif exists:
            raise ValueError(f"{name} already exists at {path}. Set overwrite to True to overwrite it or delete it manually.")
        elif create_dir:
            os.makedirs(path)
    
    def run_base_model_generation(
        self, 
        X, y,
        meta, 
        metric="default",
        config_space_provider_class=GBMLPPipelineFactory,
    ):
        """
        Execute the base model generation phase of the AutoML pipeline.
        
        This method orchestrates the entire base model generation process, including:
        - Data split creation for cross-validation
        - Metric initialization
        - Configuration space setup
        - Resource provider initialization
        - Execution of the selected base model generation strategy
        
        Parameters
        ----------
        X : array-like
            Input features with shape (n_samples, n_features).
        y : array-like
            Target variable with shape (n_samples,).
        meta : dict
            Meta-information about the dataset containing task type, feature types, etc.
        metric : str or RAAMLMetric, default="default"
            Optimization metric. If "default", selects metric based on task type.
        config_space_provider_class : class, default=GBMLPPipelineFactory
            Configuration space provider class for building the hyperparameter search space.
        
        Raises
        ------
        ValueError
            If meta-information is invalid or X and y have different sample counts.
        
        Notes
        -----
        - Creates output directories for splits, models, and base_model_pool
        - Logs total execution time
        - All configurations are evaluated using the specified base model generation strategy
        - Results are saved to disk for later loading/reuse
        """
        
        
        # Prepare output directories
        self.check_if_exists_and_delete_for_overwrite(os.path.join(self.base_dir, "splits"), "data splits", True)
        
        # Initialize components
        self._check_and_init_meta(meta, X, y)
        self._create_data_splits(X, y)
        self._init_metric(metric)
        self._init_configuration_space(X, y, config_space_provider_class)
        self._init_resource_providers()
        self._init_cores()
        
        # Check and prepare model and base model pool directories
        self.check_if_exists_and_delete_for_overwrite(os.path.join(self.base_dir, "models"), "models", True)
        self.check_if_exists_and_delete_for_overwrite(os.path.join(self.base_dir, "base_model_pool"), "base model pool", True)
        
        logging.info("Initialization for base model generation complete. Running base model generation.")

        # Run base model generation
        start = time_s()
        self._run_base_model_gen()
        logging.info(f"Base model generation completed in {time_s() - start} seconds.")
        
    def load_base_model_generation(
        self,
        X, y,
        meta,
        metric="default",
        config_space_provider_class=GBMLPPipelineFactory
    ):
        """
        Load a previously generated base model pool and reinitialize the pipeline.
        
        This method allows you to reload a base model pool from a previous AutoML run
        without re-running the expensive base model generation phase. It initializes
        all necessary components and loads the serialized base model pool from disk.
        
        Parameters
        ----------
        X : array-like
            Input features with shape (n_samples, n_features). Used to reinitialize
            configuration space and data splits.
        y : array-like
            Target variable with shape (n_samples,). Used to reinitialize configuration
            space and data splits.
        meta : dict
            Meta-information about the dataset
        metric : str or RAAMLMetric, default="default"
            Optimization metric for model evaluation. If "default", selects metric 
            based on task type.
        config_space_provider_class : class, default=GBMLPPipelineFactory
            Configuration space provider class for rebuilding the hyperparameter 
            search space. Must match the one used in the original run.
        
        Raises
        ------
        ValueError
            If meta-information is invalid.
            If X and y have different sample counts.
            If the base model pool directory does not exist.
            If the base model pool files are corrupted or incompatible.
        
        Notes
        -----
        - This method reuses the data splits saved during the original run.
        - The configuration space is rebuilt from scratch but should match the original.
        - Resource providers are reinitialized based on current pipeline configuration.
        - The base model pool must have been previously saved via run_base_model_generation().
        - To use this method, initialize the pipeline with the same parameters as the 
          original run (using loadRAAML() is recommended).
        """
        
        # Initialize components
        self._check_and_init_meta(meta, X, y)
        self._create_data_splits(X, y, load=True)
        self._init_metric(metric)
        self._init_configuration_space(X, y, config_space_provider_class)
        self._init_resource_providers()
        self._init_cores()
        
        # Load base model pool
        logging.info("Loading base model pool from folder.")
        self.base_model_pool = load_base_model_pool_from_dir(
            os.path.join(self.output_dir, self.identifier),
            self.config_space
        )
        
    
    def load_ensembling_predictions(self, method="cv"):
        """
        Load previously generated ensembling predictions from disk.
        
        This method loads ensembling predictions that were previously created and saved
        by the `create_ensembling_predictions()` method. It reinitializes the pipeline's
        ensembling predictions and resets the filter state.
        
        Parameters
        ----------
        method : str, default="cv"
            The method used to generate the ensembling predictions. This determines
            which saved predictions directory to load from.
            Options: "cv" (cross-validation predictions)
        
        Returns
        -------
        EnsemblingPredictions
            The loaded ensembling predictions object containing model predictions,
            metrics, and resource consumption data for all base models.
        
        Raises
        ------
        FileNotFoundError
            If the ensembling predictions directory does not exist for the given method.
        ValueError
            If the saved prediction files are corrupted or incompatible.
        
        Notes
        -----
        - Loading predictions resets the `filtered` flag to False, indicating that
          no filtering has been applied yet.
        - Ensembling predictions must have been previously created via
          `create_ensembling_predictions()` for this method to succeed.
        
        """
        
        # Load ensembling predictions
        self.ensembling_predictions = load_ensembling_predictions(
            self.base_dir, 
            method,
        )
        # Reset filtered flag
        self.filtered = False
        return self.ensembling_predictions
         
        
    def run_ensembling(
        self,
        method,
        name,
        scheduler = "lpt",
        do_allotment=True,
        ensembling_kwargs=None,
        filters=None,
        save_ensemble_pool=True
    ):
        """
        Execute the ensemble selection and creation phase of the AutoML pipeline.
        
        This method orchestrates the ensemble formation process, applying optional filters
        to the base model pool and then using the specified ensembling strategy to select
        and combine models. Supports multiple ensembling approaches including greedy selection,
        quality diversity optimization, and single model ensembles.
        
        Parameters
        ----------
        method : str
            Ensembling strategy to use. Options:
            - "single": Create ensemble with single models (dummy ensemble)
            - "ges": Greedy Ensemble Selection
            - "mo-ges": Multi-Objective Greedy Ensemble Selection
            - "qdo-es" or "*-qdo-es": Quality Diversity Optimization Ensemble Selection
              - "infer-qdo-es": QDO-ES optimizing for inference time
              - "energy-qdo-es": QDO-ES optimizing for energy consumption
              - "size-qdo-es": QDO-ES optimizing for ensemble size
              - "css-qdo-es": QDO-ES optimizing for config space similarity (default)
        name : str
            Unique name for the ensemble pool. Used for saving and loading ensemble artifacts.
        scheduler : str, default="lpt"
            Scheduling strategy for multi-objective resource allocation. Options:
            - "lpt": Longest Processing Time scheduler (default)
            - "hwl": Highest Workload scheduler
            - "htc": Highest Thread Count scheduler
        do_allotment : bool, default=True
            Whether to perform resource allotment/scheduling optimization. When True,
            allocates computing resources (cores, frequency) to ensemble components
            based on the selected scheduler.
        ensembling_kwargs : dict, optional
            Strategy-specific keyword arguments passed to the ensembling method.
            Common options:
            - "n_iterations" (int): Number of iterations for greedy/QDO selection
            - "batch_size" (int): Batch size for QDO-ES (default: 40)
            - "max_elites" (int): Maximum elites in QDO archive (default: 49)
            - "archive_type" (str): QDO archive type, "sliding" or other (default: "sliding")
            - "pruning_priority" (str): Pruning priority for allotment, "energy" or other (default: "energy")
            - "enable_reset" (bool): Enable reset in MO-GES (default: True)
            - "restriction_mode" (str): Restriction mode for MO-GES, "quadratic" or other (default: "quadratic")
            - "max_ensemble_size" (int): Maximum ensemble size for MO-GES (default: 20)
            - "abs_alc" (bool): Use absolute Average Loss Correlation in QDO-ES (default: True)
        filters : list of str or dict, optional
            List of filters to apply to base models before ensembling. Filters reduce
            the base model pool to a subset of high-quality models based on various criteria.
            If None, no filtering is applied. Examples: performance thresholds, diversity filters.
        save_ensemble_pool : bool, default=True
            Whether to serialize and save the resulting ensemble pool to disk.
            If True, ensemble pool is saved to {output_dir}/{identifier}/ensemble_pools/ensemble_pool_{name}.pkl
        
        Raises
        ------
        ValueError
            If method is not a recognized ensembling strategy.
            If no ensembling predictions are available (run create_ensembling_predictions first).
            If ensemble pool is empty (fewer than 2 valid base models remain after filtering).
            If scheduler is not recognized.
        
        Notes
        -----
        - Requires ensembling predictions to be created beforehand via create_ensembling_predictions().
        - Requires base model pool to be generated via run_base_model_generation().
        - Filtering is applied before ensembling if filters parameter is provided.
        - If do_allotment=True, resource scheduling is computed to optimize core and frequency allocation.
        - The ensemble pool can be later loaded via load_ensemble_pool().
        """
        
        start = time_s()
        
        if not os.path.exists(os.path.join(self.base_dir, "ensemble_pools")):
            os.makedirs(os.path.join(self.base_dir, "ensemble_pools"))
        
        method = map_ensembling_method_to_unified_name(method)
        
        if ensembling_kwargs is None:
            ensembling_kwargs = {}
        
        # Run filtering if filters are provided
        self._run_ensembling_filter(filters)
        
        logging.info("Filtering completed in {} seconds.".format(time_s() - start))
        start = time_s()
        
        # Get scheduler
        scheduler = self._get_scheduler(scheduler, ensembling_kwargs)
        
        # Run ensembling
        self._run_ensembling(
            method,
            name,
            ensembling_kwargs,
            save_ensemble_pool,
            do_allotment=do_allotment,
            scheduler=scheduler
        )
        
        logging.info("Ensembling completed in {} seconds.".format(time_s() - start))
       
        
    def refit(self):
        """
        Refit all base models in the pool on the full training dataset.
        
        This method retrains all configurations from the base model pool using the complete
        dataset (not cross-validation splits). This is typically done after base model generation
        and before ensemble creation to train final models on all available data.
        
        The refitting process:
        - Uses the full training data (X, y) stored in the pipeline
        - Applies resource limits (wall-time and memory) per model
        - Respects CPU core allocation via the affinity manager
        - Tracks progress if a progress handler is registered
        - Saves refitted model artifacts to disk
        
        Resource Limits
        ---------------
        Each refitted model is constrained by:
        - wall_time: trial_walltime_limit_s (seconds)
        - memory: trial_memory_limit_mb (megabytes)
        - cores: n_cores_target_function
        
        Raises
        ------
        ValueError
            If base_model_pool is not initialized.
            If X or y have not been set.
        
        Notes
        -----
        - This method should be called after run_base_model_generation() and before
          ensembling methods that require refitted models.
        - Refitted models are saved to {base_dir}/models/refit/ directory.
        """
        
        logging.info("Refitting base models...")
        start = time_s()
        
        # Run refit
        self.base_model_pool.refit_base_models(
            self.X, self.y,
            limit_args = {"wall_time" : (self.trial_walltime_limit_s, "s"), "memory" : (self.trial_memory_limit_mb, "MB")},
            load_fn=self._load_model_from_config,
            output_dir=self.base_dir,
            affinity_manager=self.affinity_manager,
            raaml_progress=self.raaml_progress,
            n_cores=self.n_cores_target_function
        )
        
        logging.info(f"Refit completed in {time_s() - start} seconds.")


    def _init_metric(self, metric):
        # Initialize the optimization metric.
        if isinstance(metric, str) and metric.lower() == "default":
            self.metric = get_default_metric(self.is_classification, self.n_classes if self.is_classification else None)
            logging.debug(
                f"Using default metric for classification with {self.n_classes} classes: {self.metric.name}" 
                if self.is_classification else f"Using default metric for regression: {self.metric.name}"
            )
        elif isinstance(metric, RAAMLMetric):
            self.metric = metric
            if self.metric.maximize:
                raise ValueError("Metric must be a minimization metric!")
            logging.debug(f"Using provided metric: {self.metric.name}")
        else:
            raise ValueError(f"Unknown score function: {metric}. Please provide a valid RAAMLMetric or 'default'.")
        
        # Set class labels for classification metrics
        if self.is_classification:
            self.metric.set_labels(self.y_unique_labels)
    
    
    def _create_data_splits(self, X, y, load=False):
        """
        Create or load cross-validation data splits for the pipeline.
        
        This method either creates new data splits or loads previously saved splits.
        The splits are used for cross-validation during base model generation and
        ensembling prediction creation.
        
        Parameters
        ----------
        X : array-like
            Input features with shape (n_samples, n_features).
        y : array-like
            Target variable with shape (n_samples,).
        load : bool, default=False
            If True, loads previously saved splits from disk.
            If False, creates new splits and saves them to disk.
        
        Raises
        ------
        ValueError
            If X and y have different numbers of samples.
        
        Notes
        -----
        - For classification tasks, uses stratified cross-validation by default
          to maintain class distribution across folds.
        - For regression tasks, uses regular cross-validation.
        - Creates additional budget splits (n_cv_steps) for methods like ASHA
          that require partial training on reduced data.
        - Splits are saved to {output_dir}/{identifier}/splits/ directory.
        
        Attributes Set
        --------------
        self.X : Full input features
        self.y : Full target variable
        self.data_split : RAAMLDataSplit object managing the splits
        self.X_train : Full training features (equivalent to X)
        self.y_train : Full training target (equivalent to y)
        """
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        
        self.X, self.y = X, y
        
        if load:
            # Load existing splits
            self.data_split = RAAMLDataSplit(X, y, load_folder=os.path.join(self.base_dir, "splits"), seed=self.seed)
            logging.debug(f"Loaded data splits from {os.path.join(self.base_dir, 'splits')}")
        else:
            # Create new splits
            self.data_split = RAAMLDataSplit(X, y, stratify=self.stratified if self.is_classification else False, n_cv=self.n_cv, n_budget_splits=self.bmg_kwargs.get("n_cv_steps", 3))
            self.data_split.save(os.path.join(self.output_dir, self.identifier, "splits"))
            logging.debug(f"Created data splits and saved splits to {os.path.join(self.output_dir, self.identifier, 'splits')}")

        # Set training data
        self.X_train, self.y_train = self.data_split.get_full_train_data()

    def _init_configuration_space(self, X, y, provider_class):
        """
        Initialize the configuration space.

        Parameters
        ----------
        X : The input data.
        y : The target variable.
        """
        
        # Initialize configuration space class
        self.config_space_provider = provider_class(
            X, y,
            {
                "task_type": self.task_type,
                "feature_types": self.feature_types,
            }, 
            self.include, self.exclude
        )
        
        # Get configuration space
        self.config_space = self.config_space_provider.get_config_space(self.seed)
    
    
    def _run_base_model_gen(self):
        self.base_model_pool = None

        if self.base_model_gen == "so-bo":
            self._run_SO_BO()
        else:
            raise ValueError(f"Unknown base model generation strategy: {self.base_model_gen}. ")

        self.base_model_pool.save(os.path.join(self.output_dir, self.identifier,"base_model_pool"))

        return self.base_model_pool


    def _run_ensembling_filter(self, filters):
        
        if self.filtered:
            # If already filtered, just load the ensembling predictions again to make sure they are not filtered
            self.load_ensembling_predictions()
        
        if filters is None or self.ensembling_predictions is None:
            return
        
        for filter in filters:
            # Apply each filter to the ensembling predictions sequentially
            self.ensembling_predictions.apply_filter(
                filter,
                self.base_model_pool
            )
            
        self.filtered = True
        

    def fit_models_for_ensembling_predictions(self, method="cv"):
        """
        Fit models for creating ensembling predictions.
        
        This method fits models for generating ensembling predictions based on the 
        specified method. Currently, only cross-validation ("cv") method is supported.
        Parameters
        ----------
        method : str, optional
            The method to use for fitting models for ensembling predictions. Default is "cv" (cross-validation).
            
        Raises
        ------
        NotImplementedError
            If the specified method is not supported.
        """
        
        # Prepare output directory
        self.check_if_exists_and_delete_for_overwrite(os.path.join(self.base_dir, "models", f"ens_{method}"), "models for ensembling predictions", True)
        
        logging.info("Fitting models for ensembling predictions...")
        start = time_s()
        
        # Fit models for ensembling predictions
        if method == "cv":
            # Iterate over all configurations in the base model pool
            for config_id, config in self.base_model_pool.get_configs(base_dir=self.base_dir, ret_ids=True, refit_success=False):
                try:
                    with limit(
                        self._fit_ensembling_folds,
                        wall_time=(self.trial_walltime_limit_s, "s"), 
                        memory=(self.trial_memory_limit_mb, "MB")
                    ) as limited_run_ensembling_folds:
                        try:
                            limited_run_ensembling_folds(config, config_id, method)
                        except Exception as e:
                            # Make sure all OOM errors are reported as such
                            if "can't allocate memory" in str(e):
                                logging.error(f"Out of memory error while running _target_fun with config: {get_config_hash(config)}")
                                raise MemoryLimitException("Catched OOM error in target function")
                            else:
                                raise e
                    self.update_progress("ens_cv_fit", "SUCCESS", None)
                except Exception as e:
                    short_exc = str(e).split("\n")[0]
                    logging.warning(f"Fitting models for ensembled predictions failed for config {config_id}: {short_exc}")
                    logging.debug(traceback.format_exc())
                    for i in range(self.n_cv):
                        if os.path.exists(os.path.join(self.base_dir, "models", f"ens_{method}", f"{config_id}_{i}.pkl")):
                            os.remove(os.path.join(self.base_dir, "models", f"ens_{method}", f"{config_id}_{i}.pkl"))
                    self.update_progress("ens_cv_fit", exception_to_status(e))
                    continue
        else:
            raise NotImplementedError("Only cross-validation ensembling predictions are supported for now.")
        
        logging.info("Fitting models for ensembling predictions completed in {} seconds.".format(time_s() - start))
            
            
    def create_ensembling_predictions(self, n_threads, frequency_mode="boost", method="cv", delete_after_pred=False):
        """
        Create ensembling predictions using the specified method.
        This method generates ensembling predictions for the base models in the pool
        based on the specified method (e.g., cross-validation).
        
        Parameters
        ----------
        n_threads : int
            Number of threads to use for prediction.
        frequency_mode : str, optional
            Frequency mode for prediction. Options are "boost" or "scaling". Default is "boost".
        method : str, optional
            The method to use for creating ensembling predictions. Default is "cv" (cross-validation).
        delete_after_pred : bool, optional
            Whether to delete the ensembled models after prediction to save space. Default is False.
        
        Raises
        ------
        ValueError
            If the specified method is not supported.
        """
        
        # Check if models for ensembling predictions exist
        exists = os.path.exists(os.path.join(self.base_dir, "models", f"ens_{method}"))
        config_ids = np.unique([int(f.split(".")[0].split("_")[0]) for f in os.listdir(os.path.join(self.base_dir, "models", f"ens_{method}"))]) if exists else []
                
        if len(config_ids) == 0:
            raise ValueError(f"Ensembling prediction models for method {method} do not exist. Please run fit_models_for_ensembling_predictions() or load_ensembling_predictions() first.")
        
        if not os.path.exists(os.path.join(self.base_dir, f"ensembling_predictions_{method}")):
            os.makedirs(os.path.join(self.base_dir, f"ensembling_predictions_{method}"))
        
        logging.info("Creating ensembling predictions...")
        start = time_s()
        
        # Initialize ensembling predictions object
        if self.ensembling_predictions is None:
            self.ensembling_predictions = EnsemblingPredictions(method, self.metric.name, self.assembled_resource_provider.get_names())
            
        # Ensure that predictions are not created twice and that incomplete predictions are removed
        config_ids = self.ensembling_predictions.filter_configs_incomplete(config_ids, n_threads, frequency_mode)
        self.ensembling_predictions.remove_config_ids(config_ids, n_threads, frequency_mode)
        
        if method == "cv":
            # Iterate over all configurations and create predictions
            for config_id in config_ids:
                try:
                    results = self._predict_ensembling_folds(
                        config_id,
                        method,
                        n_threads,
                        frequency_mode
                    )
                        
                    for i in range(len(results)):
                        self.ensembling_predictions.add_cv_fold_predictions(*(results[i]))
                            
                    self.update_progress("ens_cv_predict", "SUCCESS")
                            
                except Exception as e:
                    for i in range(self.n_cv):
                        if os.path.exists(os.path.join(self.base_dir, f"ensembling_predictions_{method}", f"{config_id}_{i}.npy")):
                            os.remove(os.path.join(self.base_dir, f"ensembling_predictions_{method}", f"{config_id}_{i}.npy"))
                    
                    short_exc = str(e).split("\n")[0]
                    logging.warning(f"Predicting ensembled predictions failed for config {config_id}: {short_exc}")
                    logging.debug(traceback.format_exc())
                    self.update_progress("ens_cv_predict", exception_to_status(e))
                
                # Optionally delete models after prediction
                if delete_after_pred:
                    refit_exists = os.path.exists(os.path.join(self.base_dir, "models", "refit", f"{config_id}.pkl"))
                    rand_index = np.random.randint(0, self.n_cv)
                    for i in range(self.n_cv):
                        if (
                            os.path.exists(os.path.join(self.base_dir, "models", f"ens_{method}", f"{config_id}_{i}.pkl")) and 
                            (refit_exists or i != rand_index)
                        ):
                            os.remove(os.path.join(self.base_dir, "models", f"ens_{method}", f"{config_id}_{i}.pkl"))   
                    if not refit_exists:
                        os.rename(
                            os.path.join(self.base_dir, "models", f"ens_{method}", f"{config_id}_{rand_index}.pkl"),
                            os.path.join(self.base_dir, "models", f"ens_{method}", f"{config_id}.pkl")
                        )            
        else:
            raise NotImplementedError("Only cross-validation ensembling predictions are supported for now.")
        
        logging.info("Predicting ensembling predictions completed in {} seconds.".format(time_s() - start))
        
        # Save ensembling predictions
        path = os.path.join(self.output_dir, self.identifier, f"ensembling_predictions_{method}")
        self.ensembling_predictions.save(path)
        logging.info(f"Ensembling predictions saved to {path}")

            
    def _predict_ensembling_folds(self, config_id, method, n_threads, frequency_mode):
        results_full = []
        for i in range(self.n_cv):
            path = os.path.join(self.base_dir, "models", f"ens_{method}", f"{config_id}_{i}.pkl")
            with open(path, "rb") as f:
                model = pickle.load(f)
                
            model.set_n_threads(n_threads)
            
            X_val, y_val = self.data_split.get_cv_val_data(i, ensembling=True)
            
            # Dummy predict
            with filter_warnings():
                model.predict(X_val)

            if frequency_mode == "boost":
                with filter_warnings():        
                    if hasattr(model, "predict_proba"):
                        with self.assembled_resource_provider:
                            predictions = model.predict_proba(X_val)
                    else:
                        with self.assembled_resource_provider:
                            predictions = model.predict(X_val)
                        
                resources = self.assembled_resource_provider.get_resource()
                
                path = os.path.join(self.base_dir, f"ensembling_predictions_{method}", f"{config_id}_{i}.npy")
                
                metric_val = self.metric(y_val, predictions)
                if n_threads == 1:
                    np.save(path, predictions)
                    
                discovery_time = self.base_model_pool.get_discovery_time(config_id)
                    
                objectives = {**resources, self.metric.name: metric_val}
                
                results_full.append((config_id, i, n_threads, "boost", path, objectives, discovery_time))
            elif frequency_mode == "scaling":
                for frequency in get_freq_values():
                    with SetFreqMgr(frequency):
                        with filter_warnings():        
                            if hasattr(model, "predict_proba"):
                                with self.assembled_resource_provider:
                                    predictions = model.predict_proba(X_val)
                            else:
                                with self.assembled_resource_provider:
                                    predictions = model.predict(X_val)
                                
                    resources = self.assembled_resource_provider.get_resource()
                    
                    path = os.path.join(self.base_dir, f"ensembling_predictions_{method}", f"{config_id}_{i}.npy")
                    if n_threads == 1:
                        np.save(path, predictions)
                    
                    metric_val = self.metric(y_val, predictions)
                        
                    discovery_time = self.base_model_pool.get_discovery_time(config_id)
                        
                    objectives = {**resources, self.metric.name: metric_val}
                    
                    results_full.append((config_id, i, n_threads, frequency, path, objectives, discovery_time))
                    
            else:
                raise ValueError(f"Unknown frequency mode: {frequency_mode}")
        
        return results_full
    
        
    def _fit_ensembling_folds(self, config, config_id, method):
        self.affinity_manager.reserve_and_set_cores(self.n_cores_target_function)
        for i in range(self.n_cv):
            X_train, y_train = self.data_split.get_cv_train_data(i, ensembling=True)

            model = self._load_model_from_config(config, np.random.randint(0, 1e9))
            model.set_n_threads(self.n_cores_target_function)
            if hasattr(model, "set_timeout"):
                model.set_timeout(0.7 * self.trial_walltime_limit_s / self.n_cv)
            
            with filter_warnings():
                model.fit(X_train, y_train)
            
            with open(os.path.join(self.base_dir, "models", f"ens_{method}", f"{config_id}_{i}.pkl"), "wb") as f:
                pickle.dump(model, f)
    
    
    
    def predict_final(self, model, X, y, config_id, n_threads=8, frequency_mode="boost"):
        """
        Create final predictions for a given model configuration.
        
        Parameters
        ----------
        model : RAAMLBasePipeline
            The trained model pipeline to use for predictions.
        X : array-like
            Input features for prediction.
        y : array-like
            True target values for evaluation.
        config_id : str
            Unique identifier for the model configuration.
        n_threads : int, optional
            Number of threads to use for prediction. Default is 8.
        frequency_mode : str, optional
            Frequency mode for prediction. Options are "boost" or "scaling". Default is "
            boost".
            
        Returns
        -------
        list of dict
            A list of dictionaries containing prediction results and resource usage.
        """
        
        
        results = []
        for i in range(self.n_cv): # Use same number as cross validation folds here, although not related to CV
            model.set_n_threads(n_threads)
            
            # Dummy predict to warm up
            with filter_warnings():
                model.predict(X)
            
            # Iterate over frequency settings and make predictions
            if frequency_mode == "boost":
                with filter_warnings():        
                    if hasattr(model, "predict_proba"):
                        with self.assembled_resource_provider:
                            predictions = model.predict_proba(X)
                    else:
                        with self.assembled_resource_provider:
                            predictions = model.predict(X)
                            
                resources = self.assembled_resource_provider.get_resource()
                
                metric_val = self.metric(y, predictions)
                path = os.path.join(self.base_dir, "final_predictions", f"{config_id}.npy")
                
                if i==0:
                    np.save(path, predictions)
                    
                res_dict = {**resources, self.metric.name: metric_val, "path": path, "frequency": "boost", "index": i}
                results.append(res_dict)
            elif frequency_mode=="scaling":
                for frequency in get_freq_values():
                    with SetFreqMgr(frequency):
                        with filter_warnings():        
                            if hasattr(model, "predict_proba"):
                                with self.assembled_resource_provider:
                                    predictions = model.predict_proba(X)
                            else:
                                with self.assembled_resource_provider:
                                    predictions = model.predict(X)
                                    
                    resources = self.assembled_resource_provider.get_resource()
                    
                    path = os.path.join(self.base_dir, "final_predictions", f"{config_id}.npy")
                    metric_val = self.metric(y, predictions)
                        
                    res_dict = {**resources, self.metric.name: metric_val, "path": path, "frequency": frequency, "index" : i}
                    results.append(res_dict)
                
            else:
                raise NotImplementedError(f"Frequency mode {frequency_mode} not implemented.")
                
        return results
    
    
    def create_final_predictions(self, X, y, n_cores=8, frequency_mode="boost"):
        """
        Create final predictions for all models.
        
        Parameters
        ----------
        X : array-like
            Input features for prediction.
        y : array-like
            True target values for evaluation.
        n_cores : int, optional
            Number of CPU cores to use for prediction. Default is 8.
        frequency_mode : str, optional
            Frequency mode for prediction. Options are "boost" or "scaling". Default is "
            boost".
        """
        
        # Initialize final results object
        if self.final_results is None:
            self.final_results = FinalResults([self.metric.name] + self.assembled_resource_provider.get_names())
            
        # Prepare output directory
        if not os.path.exists(os.path.join(self.base_dir, "final_predictions")):
            os.makedirs(os.path.join(self.base_dir, "final_predictions"))
                
        logging.info("Creating final predictions...")
        start = time_s()
        
        # Get all config ids from base model pool
        ids = self.base_model_pool.get_config_ids()
        # Only create final predictions for configs that survived either refit or ensembling
        ids = filter_refit_ens_models(self.base_dir, self.ensembling_predictions.method, ids)
        # Make sure we don't predict the same config twice
        ids = self.final_results.filter_configs_incomplete(ids, n_cores, frequency_mode)
        # Ensure that incomplete predictions are removed
        self.final_results.remove_config_ids(ids, n_cores, frequency_mode)
        
        # Iterate over all configurations and create final predictions
        for config_id in ids:
            path = get_final_model_path(self.base_dir, config_id, self.ensembling_predictions.method)
            with open(path, "rb") as f:
                pipeline = pickle.load(f)

            try:
                results = self.predict_final(pipeline, X, y, config_id, n_cores, frequency_mode)
                    
                for res in results:
                    path = res.pop("path")
                    index = res.pop("index")
                    frequency = res.pop("frequency")
                    self.final_results.add_results(config_id, index, n_cores, frequency, res, path)
                    
                self.update_progress("final_predict", "SUCCESS", None)
            except Exception as e:
                short_exc = str(e).split("\n")[0]
                logging.info(f"Final prediction of model {config_id} failed: {short_exc}.")
                logging.debug(traceback.format_exc())
                
                self.update_progress("final_predict", exception_to_status(e), None)
                continue
            
        logging.info("Creating final predictions completed in {} seconds.".format(time_s() - start))
        
        self.final_results.save(os.path.join(self.base_dir, "final_predictions"))
        logging.info(f"Final predictions saved to {os.path.join(self.base_dir, 'final_predictions')}")
        
    
    def load_final_predictions(self):
        self.final_results = FinalResults.load_results(os.path.join(self.base_dir, "final_predictions"))
    
    
    def _get_idle_power(self, frequencies):
        if len(frequencies) == 1 and "boost" in frequencies:
            return get_idle_power(boost=True)
        elif "boost" not in frequencies:
            return get_idle_power(boost=False)
        else:
            raise ValueError("Set of frequencies must not mix boost and non-boost mode!")
        
    
    def _get_scheduler(self, scheduling = "lpt", ensembling_kwargs = {}):
        if self.ensembling_predictions is None:
            return None
        
        # Compute total number of cores available
        n_cores_total = self.ensembling_predictions.results["n_threads"].max()
        frequencies = self.ensembling_predictions.results["frequency"].unique()
        
        # Get idle power for the frequency settings
        idle_power = self._get_idle_power(frequencies)
        pruning_priority = ensembling_kwargs.get("pruning_priority","energy")
        
        # Instantiate the scheduler
        if scheduling == "lpt":
            return LongestProcessingTimeScheduler(n_cores_total, pruning_priority, idle_power)
        elif scheduling == "hwl":
            return HighestWorkloadScheduler(n_cores_total, pruning_priority, idle_power)
        elif scheduling == "htc":
            return HighestThreadCountScheduler(n_cores_total, pruning_priority, idle_power)
        else:
            raise ValueError(f"Unknown scheduler {scheduling}. Choose one from ['lpt', 'hwl', 'htc']")
    
                
    def _run_ensembling(self, method, name, ensembling_kwargs, save_ensemble_pool, do_allotment, scheduler):
        # Check if ensembling predictions are available        
        if self.ensembling_predictions is None or len(self.ensembling_predictions.get_config_ids()) < 2:
            warnings.warn("Ensemble pool is empty. No ensembling will be performed.")
            self.ensemble_pool = None
            return 
        
        # Run ensembling based on the specified method
        if method == "single": # Dummy Ensemble with single models
            frequencies = self.ensembling_predictions.results["frequency"].unique().tolist()
            max_n_threads = self.ensembling_predictions.results["n_threads"].max()
            self.ensemble_pool = base_model_pool_to_ensemble_pool(
                self.base_model_pool,
                self.base_dir,
                self.metric,
                max_n_threads,
                frequencies,
                self.ensembling_predictions,
                scheduler
            )
        elif method.endswith("qdo-es"): # Quality Diversity Optimization Ensemble Selection     
            key = method.split("-")[0].lower()
            map = {"infer": "inference_time", "energy": "energy_consumption", "size": "ensemble_size"}
            self._run_qdo_ensemble_selection(ensembling_kwargs, do_allotment, map.get(key, "css"), scheduler)
        elif method in ["ges", "mo-ges"]: # Greedy Ensemble Selection
            self._run_default_ensembling(ensembling_kwargs, do_allotment, method, scheduler)
        else:
            raise ValueError(f"Unknown ensembling strategy: {method}. Choose from 'single' (Dummy Ensemble with single models), 'ges' (Greedy Ensemble Selection), 'qdo-es' (Quality Diversity Optimization Ensemble Selection), 'infer-qdo-es' (Inference Time Quality Diversity Optimization Ensemble Selection), 'energy-qdo-es' (Ensemble Energy Quality Diversity Optimization Ensemble Selection), 'size-qdo-es' (Ensemble Size Quality Diversity Optimization Ensemble Selection), or 'ra-stacking' (Resource-Aware Stacking).")
        
        # Set additional ensemble pool information
        self.ensemble_pool.set_resource_providers(self.assembled_resource_provider, self.resource_providers)
        self.ensemble_pool.set_final_results(self.final_results)
        if self.is_classification:
            self.ensemble_pool.set_y_labels(self.y_unique_labels)
        
        # Save ensemble pool if specified
        if save_ensemble_pool:
            self._pickle_ensemble_pool(name)
        
        
    def _load_model_from_config(self, config, seed=None):
        """
        Create the model based on the configuration.
        """
        
        if seed is None:
            seed = self.seed
        
        if self.config_space_provider is None:
            raise ValueError("Configuration space provider is not initialized.")
        
        return self.config_space_provider.load_pipeline(config, seed)      


    def _create_unique_model_identifier(self):
        """
        Create a unique identifier for the trial
        """
        
        exists = True
        while exists:
            rnd = np.random.RandomState(seed=time_ns() % 2**32)
            rand_int = rnd.randint(0, 1e9)
            exists = os.path.exists(os.path.join(self.output_dir, self.identifier, "models", "bmg", f"{rand_int}.pkl"))
        return rand_int
    
    
    def _run_cv_fold(self, model, X_train, y_train, X_val, y_val, results, config):
        with filter_warnings():
            model.fit(X_train, y_train)
            
            if self.is_classification:
                with self.assembled_resource_provider:
                    y_pred = model.predict_proba(X_val)
            else:
                with self.assembled_resource_provider:
                    y_pred = model.predict(X_val)
                y_pred = y_pred.squeeze()

        results[self.metric.name].append(self.metric(y_val, y_pred))
        
        for r_name, r_val in self.assembled_resource_provider.get_resource().items():
            results[r_name].append(r_val)
    

    def _target_fun_with_budget(self, config, seed, n_cv, time_limit):
        if n_cv == self.n_cv:
            return self._target_fun(config, seed)
        else:
            logging.debug(f"Running _target_fun_with_budget with config: {get_config_hash(config)} and budget {n_cv}")
            self.affinity_manager.reserve_and_set_cores()
            logging.debug(f"_target_fun_with_budget running with cores: {os.sched_getaffinity(0)}")  
        
            results = {n : [] for n in [self.metric.name] + self.assembled_resource_provider.get_names()}
            
            try:
                # Perform cross-validation                
                for i in range(n_cv):
                    model = self._load_model_from_config(config, np.random.randint(0, 1e9))
                    model.set_n_threads(self.n_cores_target_function)
                    if hasattr(model, "set_timeout"):
                        model.set_timeout(0.7 * time_limit / n_cv)
                        
                    X_train, y_train = self.data_split.get_cv_train_data(i, n_cv)
                    X_val, y_val = self.data_split.get_cv_val_data(i, n_cv)

                    self._run_cv_fold(model, X_train, y_train, X_val, y_val, results, config)
            except Exception as e:
                # Make sure all OOM errors are reported as such
                if "can't allocate memory" in str(e):
                    logging.error(f"Out of memory error while running _target_fun with config: {get_config_hash(config)}")
                    raise MemoryLimitException("Catched OOM error in target function")
                else:
                    raise e
                            
            # Aggregate results across cross-validation folds  
            ret_dict = {obj_name: np.mean(results[obj_name]) for obj_name in results.keys()}
            
            additional_info = {
                "results" : results,
            }
            
            logging.debug(f"Returning from _target_fun_with_budget with config: {get_config_hash(config)} and budget {n_cv}")
            return ret_dict, additional_info

    def _target_fun(self, config, seed):
        logging.debug(f"Running _target_fun with config: {get_config_hash(config)}")
        self.affinity_manager.reserve_and_set_cores()
        logging.debug(f"_target_fun running with cores: {os.sched_getaffinity(0)}")
        
        results = {n : [] for n in [self.metric.name] + self.assembled_resource_provider.get_names()}
        
        try:
            # Perform cross-validation                
            for i in range(self.n_cv):
                model = self._load_model_from_config(config, np.random.randint(0, 1e9))
                model.set_n_threads(self.n_cores_target_function)
                if hasattr(model, "set_timeout"):
                    model.set_timeout(0.7 * self.trial_walltime_limit_s / self.n_cv)
                
                X_train, y_train = self.data_split.get_cv_train_data(i)
                X_val, y_val = self.data_split.get_cv_val_data(i)

                self._run_cv_fold(model, X_train, y_train, X_val, y_val, results, config)
        except Exception as e:
            # Make sure all OOM errors are reported as such
            if "can't allocate memory" in str(e):
                logging.error(f"Out of memory error while running _target_fun with config: {get_config_hash(config)}")
                raise MemoryLimitException("Catched OOM error in target function")
            else:
                raise e
                    
        logging.debug("Results:")
        for obj_name in results.keys():
            logging.debug(f"{obj_name}: {results[obj_name]}")
            
        # Aggregate results across cross-validation folds
        ret_dict = {obj_name: np.mean(results[obj_name]) for obj_name in results.keys()}
        
        additional_info = {
            "results" : results
        }
          
        logging.debug(f"Returning from _target_fun with config: {get_config_hash(config)} and ret_dict: {ret_dict}")
        
        if self.base_model_gen == "so-bo":
            return ret_dict[self.metric.name], additional_info
        
        return ret_dict, additional_info
    
    
    def update_progress(self, step, status, objectives=None):
        # Update the RAAML progress tracker if it exists
        if self.raaml_progress:
            self.raaml_progress.update_progress(step, status, objectives)
        
    
    def _run_SO_BO(self):
        """
        Run the Single-objective Bayesian Optimization using SMAC3.
        """
        scenario = Scenario(
            configspace = self.config_space,
            name = self.identifier,
            output_directory=os.path.join(self.output_dir, self.identifier, "smac_output"),
            crash_cost = np.inf,
            deterministic=True,
            walltime_limit = self.time_search_s,
            trial_walltime_limit = self.trial_walltime_limit_s,
            trial_memory_limit = self.trial_memory_limit_mb * 1e6,
            n_trials = self.bmg_kwargs.get("n_trials", np.inf),
            seed=self.seed,
            n_workers=self.n_jobs_optimization,
            objectives = self.metric.name,
        )            
        
        smac = HPOFacade(
            scenario,
            target_function=self._target_fun,
            overwrite=self.overwrite,
            logging_level=self.logging_level,
            initial_design=RandomInitialDesign(scenario, n_configs = self.bmg_kwargs.get("n_configs_initial", 5)), 
            # callbacks=[CustomCallback(self.update_progress)],
        )

        start = time_s()
        
        smac.optimize()

        self.base_model_pool = base_model_pool_from_smac_history(
            smac.runhistory,
            multi_objective=False,
            identifier=self.identifier,
            cs=self.config_space,
            objective_names=[self.metric.name] + list(self.resource_providers.keys()),
            start_time=start
        )


    
            

    
        
    def _run_default_ensembling(self, ensembling_kwargs, do_allotment, method, scheduler):
        """
        Run GES or MO-GES ensembling.
        """
                
        if method == "ges":
            strategy = GreedyEnsembleSelection(
                n_iterations=ensembling_kwargs.get("n_iterations", 50),
                metric=self.metric,
                n_jobs=self.n_jobs_optimization,
                random_state=self.seed,
                use_best=True
            )
        self.ensemble_pool = strategy.ensemble_fit(
            self.data_split.get_ensembling_labels(),
            self.ensembling_predictions,
            self.base_model_pool,
            self.base_dir,
            do_allotment,
            scheduler
        )