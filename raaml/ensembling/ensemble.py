import numpy as np
import pickle
import pandas as pd
from contextlib import nullcontext

class Ensemble:
    """
    Class representing an ensemble of models.
    
    Attributes
    ----------
    config : dict
        Configuration of the ensemble.
    model_path : str
        Path to the model.
    weight_matrix : list
        Weight matrix for the ensemble.
    """
    
    def __init__(self, configs, model_paths, weight_vector, resource_provider, y_labels):

        self.weight_vector = []
        self.model_paths = []
        self.configs = []
        self.resource_provider = resource_provider
        self.y_labels = y_labels
        
        for weight, model_path, config in zip(weight_vector, model_paths, configs):
            if abs(weight) < 1e-12:
                continue
            self.weight_vector.append(weight)
            self.model_paths.append(model_path)
            self.configs.append(config)
        
        self.loaded = False
        self.models = None
        
    def set_models(self, models):
        """
        Set the models for the ensemble.
        
        Parameters
        ----------
        models : list
            List of models to set.
        """
        self.models = models
        self.loaded = True

    def load_models(self):
        """
        Load the models from the specified paths.
        
        Returns
        -------
        list
            List of loaded models.
        """
        if self.loaded:
            print("Models already loaded. (INSIDE)")
            return
        
        models = []
        for path in self.model_paths:
            with open(path, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
        self.set_models(models)
        
    def predict(self, X):
        """
        Make predictions using the ensemble of models.
        
        Parameters
        ----------
        X : array-like
            Input data for making predictions.
        
        Returns
        -------
        array-like
            Predictions from the ensemble.
        """
        from raaml.util.timer_display import TimerDisplay
        
        
        if not self.loaded:
            self.load_models()
        
        predictions = []
        resource_usage = np.zeros((len(self.models), len(self.resource_provider)))
        
        for i, (model, weight) in enumerate(zip(self.models, self.weight_vector)):
            if hasattr(model, 'predict_proba'):
                with self.resource_provider:
                    pred = model.predict_proba(X)
                pred = model.rearrange_probas(pred, self.y_labels)
                predictions.append(pred * weight)
                
                res = self.resource_provider.get_resource()
                for j, key in enumerate(self.resource_provider.get_names()):
                    resource_usage[i,j] = res[key]
            else:
                with self.resource_provider:
                    pred = model.predict(X)
                predictions.append(pred * weight)
                
                res = self.resource_provider.get_resource()
                for j, key in enumerate(self.resource_provider.get_names()):
                    resource_usage[i,j] = res[key]
        
        predictions = np.array(predictions)
        
        final_pred = np.sum(predictions, axis=0) / np.sum(self.weight_vector)
        if len(final_pred.shape) == 2:
            final_pred = final_pred.argmax(axis=1)
        final_res = resource_usage.sum(axis=0)
        final_res = {key: final_res[j] for j, key in enumerate(self.resource_provider.get_names())}
        
        return final_pred, final_res
    