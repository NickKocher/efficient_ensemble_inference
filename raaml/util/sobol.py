import ConfigSpace as CS
import numpy as np
from scipy.stats import qmc
import ConfigSpace.hyperparameters as CSH
import warnings
import random
from ConfigSpace.util import deactivate_inactive_hyperparameters, ForbiddenValueError

def transform_continuous_designs(
    design: np.ndarray, origin: str, configspace: CS.ConfigurationSpace
) -> list[CS.Configuration]:
    """Transforms the continuous designs into a discrete list of configurations.

    Parameters
    ----------
    design : np.ndarray
        Array of hyperparameters originating from the initial design strategy.
    origin : str | None, defaults to None
        Label for a configuration where it originated from.
    configspace : ConfigurationSpace

    Returns
    -------
    configs : list[Configuration]
        Continuous transformed configs.
    """
    params = configspace.get_hyperparameters()
    for idx, param in enumerate(params):
        if isinstance(param, CSH.IntegerHyperparameter):
            design[:, idx] = param._inverse_transform(param._transform(design[:, idx]))
        elif isinstance(param, CSH.NumericalHyperparameter):
            continue
        elif isinstance(param, CSH.Constant):
            design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
            design_[:, :idx] = design[:, :idx]
            design_[:, idx + 1 :] = design[:, idx:]
            design = design_
        elif isinstance(param, CSH.CategoricalHyperparameter):
            v_design = design[:, idx]
            v_design[v_design == 1] = 1 - 10**-10
            design[:, idx] = np.array(v_design * len(param.choices), dtype=int)
        elif isinstance(param, CSH.OrdinalHyperparameter):
            v_design = design[:, idx]
            v_design[v_design == 1] = 1 - 10**-10
            design[:, idx] = np.array(v_design * len(param.sequence), dtype=int)
        else:
            raise ValueError("Hyperparameter not supported when transforming a continuous design.")

    configs = []
    for vector in design:
        try:
            conf = deactivate_inactive_hyperparameters(
                configuration=None, configuration_space=configspace, vector=vector
            )
        except ForbiddenValueError:
            continue

        conf.origin = origin
        configs.append(conf)

    return configs




def sobol_sampling(cs, n_samples, seed):
    params = list(cs.values())

    constants = 0
    for p in params:
        if isinstance(p, CSH.Constant):
            constants += 1

    dim = len(params) - constants
    sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=seed)

    configs = []
    n_samples_power_of_2 = 2**(int(np.floor(np.log2(n_samples))) + 3) # Adjust to a higher power of 2 to compensate for the possibly high number of invalid configurations
    while len(configs) < n_samples:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = sobol_gen.random(n_samples_power_of_2)
            configs.extend(transform_continuous_designs(design=sobol, origin="sobol_sampling", configspace=cs))
        
    return configs[:n_samples]