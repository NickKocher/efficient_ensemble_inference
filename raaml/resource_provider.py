from abc import ABC, abstractmethod
import torch
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import yaml
import numpy as np
import time
import random
import logging

class ResourceProvider(ABC):
    """
    A class to measure/predict resource usage for some resource metric.
    """

    def __init__(self, metric_name, optimal_value, worst_value):
        """
        Initialize the ResourceProvider with an empty resource dictionary.
        """
        self.metric_name = metric_name
        
        if optimal_value is None or worst_value is None:
            raise ValueError("Both optimal_value and worst_value must be provided.")
        
        self.optimal_value = optimal_value
        self.worst_value = worst_value
        
        
    @abstractmethod
    def get_resource(self):
        """
        Get the resource value for a given configuration.
        
        Parameters
        ----------
        config : dict
            The configuration for which to get the resource value.
        
        Returns
        -------
        float
            The resource value for the given configuration.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @abstractmethod
    def get_range(self):
        """
        Get the range of the resource metric.
        
        Returns
        -------
        tuple
            A tuple containing the minimum and maximum values of the resource metric.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
           
class DummyResourceProvider(ResourceProvider):
    """
    A class to measure/predict resource usage for some resource metric using random values.
    """

    def __init__(self, metric_name, min=0, max=1):
        """
        Initialize the DummyResourceProvider with a dummy value.
        
        Parameters
        ----------
        metric_name : str
            The name of the resource metric.
        min : float, optional
            The minimum value for the resource metric (default is 0).
        max : float, optional
            The maximum value for the resource metric (default is 1).
        """
        super().__init__(metric_name, min, max)
        self.min = min
        self.max = max
        
        
    def get_resource(self):
        return random.uniform(self.min, self.max)
    
    def get_range(self):
        """
        Get the range of the resource metric.
        
        Returns
        -------
        tuple
            A tuple containing the minimum and maximum values of the resource metric.
        """
        return (self.min, self.max * 100) # Use 100 * max to account for ensembles
        

class ContextManagerResourceProvider(ResourceProvider):
    """
    A class to measure resource usage for some resource metric using a context manager.
    """

    def __init__(self, metric_name, optimal_value, worst_value):
        """
        Initialize the ContextManagerResourceProvider with a context manager and a metric name.
        
        Parameters
        ----------
        metric_name : str
            The name of the resource metric.
        """
        super().__init__(metric_name, optimal_value, worst_value)
        self.resource_val = None
    
    
    @abstractmethod
    def __enter__(self):
        """
        Enter the runtime context related to this object. To be overridden by subclasses.
        
        Returns
        -------
        self
            The instance of the class.
        """
        return self
    
    
    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object. To be overridden by subclasses.
        
        Parameters
        ----------
        exc_type : type
            The exception type.
        exc_value : Exception
            The exception value.
        traceback : TracebackType
            The traceback object.
        
        Returns
        -------
        """
        return False
    
    
    def get_resource(self):
        return self.resource_val
    
    def get_range(self):
        """
        Get the range of the resource metric.
        
        Returns
        -------
        tuple
            A tuple containing the minimum and maximum values of the resource metric.
        """
        return (self.optimal_value, self.worst_value)


class InferenceTimeProvider(ContextManagerResourceProvider):
    def __init__(self, metric_name):
        super().__init__(metric_name, 0.0, np.inf)
        
    def __enter__(self):
        self.time_start = time.perf_counter()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.resource_val = time.perf_counter() - self.time_start

class AMDEnergyProvider(DummyResourceProvider):
    pass
# class AMDEnergyProvider(ContextManagerResourceProvider):
#     def __init__(self, metric_name):
        
#         # try:
#         #     self._read_energies()
#         # except Exception as e:
#         #     raise ValueError(f"Could not read energy from /sys/class/hwmon/hwmon3/energy_input: {e}. Make sure you have access to /sys/class/hwmon/hwmon3/energy<core_id>_input.")
        
#         super().__init__(metric_name, 0.0, np.inf)
        
#     def _read_energy(self, core):
#         with open(f"/sys/class/hwmon/hwmon3/energy{core+1}_input", "r") as f:
#             energy = float(f.read()) / 1e6
#         return energy

#     def _read_energies(self):
#         cores = sorted(list(os.sched_getaffinity(0)))
#         return np.array([self._read_energy(core) for core in cores])
        
#     def __enter__(self):
#         self.energy_start = self._read_energies()
        
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.energy_end = self._read_energies()
#         self.resource_val = np.sum(self.energy_end - self.energy_start)

       
        
class AssembledResourceProvider(ContextManagerResourceProvider):
    """
    A class to assemble multiple resource providers into a single resource provider.
    """

    def __init__(self, resource_providers):
        """
        Initialize the AssembledResourceProvider with a list of resource providers and an empty resource dictionary.
        
        Parameters
        ----------
        metric_name : str
            The name of the resource metric.
        resource_providers : list
            A list of ResourceProvider objects.
        """
        
        super().__init__(
            "_".join(resource_providers.keys()),
            [rp.optimal_value for rp in resource_providers.values()],
            [rp.worst_value for rp in resource_providers.values()]
        )
        self.resource_providers = resource_providers
        
        
    def __enter__(self):
        """
        Enter the runtime context related to this object.
        
        Returns
        -------
        self
            The instance of the class.
        """
        
        for provider in self.resource_providers.values():
            if isinstance(provider, ContextManagerResourceProvider):
                provider.__enter__()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object.
        
        Parameters
        ----------
        exc_type : type
            The exception type.
        exc_value : Exception
            The exception value.
        traceback : TracebackType
            The traceback object.
        
        Returns
        -------
        bool
            False to propagate exceptions, True to suppress them.
        """
                
        for provider in self.resource_providers.values():
            if isinstance(provider, ContextManagerResourceProvider):
                provider.__exit__(exc_type, exc_value, traceback)
                
                
    def get_names(self):
        """
        Get the names of the resource providers.
        
        Returns
        -------
        list
            A list of names of the resource providers.
        """
        return list(self.resource_providers.keys())
    
    def get_resource(self):
        return {name: provider.get_resource() for name, provider in self.resource_providers.items()}
    
    def __len__(self):
        """
        Get the number of resource providers.
        
        Returns
        -------
        int
            The number of resource providers.
        """
        return len(self.resource_providers)
    
    def get_range(self):
        return {name : provider.get_range() for name, provider in self.resource_providers.items()}