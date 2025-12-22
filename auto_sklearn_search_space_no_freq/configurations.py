import json
import warnings
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    IntegerHyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import ForbiddenValueError, deactivate_inactive_hyperparameters
from auto_sklearn_search_space_no_freq.algorithm_util import spaces
from sklearn.utils.multiclass import type_of_target
from scipy import sparse
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats.qmc import Sobol
from auto_sklearn_search_space_no_freq.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
)

classification_task_mapping = {
    "multilabel-indicator": MULTILABEL_CLASSIFICATION,
    "multiclass": MULTICLASS_CLASSIFICATION,
    "binary": BINARY_CLASSIFICATION,
}
regression_task_mapping = {
    "continuous-multioutput": MULTIOUTPUT_REGRESSION,
    "continuous": REGRESSION,
    "multiclass": REGRESSION,
}


# copied from smac
def transform_continuous_designs(
    design: np.ndarray, origin: str, configspace: ConfigurationSpace
) -> list[Configuration]:
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
        if isinstance(param, IntegerHyperparameter):
            design[:, idx] = param._inverse_transform(param._transform(design[:, idx]))
        elif isinstance(param, NumericalHyperparameter):
            continue
        elif isinstance(param, Constant):
            design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
            design_[:, :idx] = design[:, :idx]
            design_[:, idx + 1 :] = design[:, idx:]
            design = design_
        elif isinstance(param, CategoricalHyperparameter):
            v_design = design[:, idx]
            v_design[v_design == 1] = 1 - 10**-10
            design[:, idx] = np.array(v_design * len(param.choices), dtype=int)
        elif isinstance(param, OrdinalHyperparameter):
            v_design = design[:, idx]
            v_design[v_design == 1] = 1 - 10**-10
            design[:, idx] = np.array(v_design * len(param.sequence), dtype=int)
        else:
            raise ValueError(
                "Hyperparameter not supported when transforming a continuous design."
            )

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


def load_algorithms(filename="algorithms"):
    f = open(filename, "r")
    algorithm_dict = json.load(f)
    f.close()
    return algorithm_dict


def save_algorithms(algorithm_dict={}, filename="algorithms.txt"):
    f = open(filename, "w")
    json.dump(algorithm_dict, f)
    f.close()
    return algorithm_dict


def get_algorithm_conf_from_ID(algID, algorithm_dict):
    return algorithm_dict["algID"]


def load_algorithm_with_hyperparameters(
    task, feat, config, dataset_properties, seed, include, exclude, init_params=None
):
    model = spaces[task](
        feat_type=feat,
        config=config,
        dataset_properties=dataset_properties,
        random_state=seed,
        include=include,
        exclude=exclude,
        init_params=init_params,
    )
    return model


def sobol_sequence(configspace, size=1, progress=0) -> list[Configuration]:
    params = configspace.get_hyperparameters()
    og_size = size
    og_progress_size = og_size + progress
    constants = 0
    for p in params:
        if isinstance(p, Constant):
            constants += 1
    dim = len(params) - constants
    sobol_gen = Sobol(d=dim, scramble=True, seed=0)
    configs = []
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = sobol_gen.random(og_progress_size)

        configs = configs + transform_continuous_designs(
            design=sobol, origin="Initial Design: Sobol", configspace=configspace
        )
        print(len(configs))
        if len(configs) >= og_progress_size:
            return configs[0:og_progress_size]


def latin_hypercube(configspace, size=1, progress=0) -> list[Configuration]:
    params = configspace.get_hyperparameters()
    og_size = size
    og_progress_size = og_size + progress
    constants = 0
    for p in params:
        if isinstance(p, Constant):
            constants += 1
    size = (size + progress) * 20
    while True:
        lhd = LatinHypercube(d=len(params) - constants, seed=0).random(n=size)
        configs = transform_continuous_designs(
            design=lhd,
            origin="Initial Design: Latin Hypercube",
            configspace=configspace,
        )
        if len(configs) >= og_progress_size:
            return configs[0:og_progress_size]
        else:
            print(size)
            size = size * 2


def create_configuration_space(task, feat, X, y, include=None, exclude=None):
    target_type = type_of_target(y)
    # analysis of data
    if sparse.issparse(X):
        is_sparse = 1
        # has_missing = np.all(
        #     np.isfinite(cast(sparse.csr_matrix, X).data)
        #     )
    else:
        is_sparse = 0
        # if hasattr(X, "iloc"):
        #     has_missing = cast(pd.DataFrame, X).isnull().values.any()
        # else:
        #     has_missing = np.all(np.isfinite(X))
    if task == "regression":
        task_type = regression_task_mapping[target_type]
        multioutput = False
        if task_type == MULTIOUTPUT_REGRESSION:
            multioutput = True
        dataset_properties = {"multioutput": multioutput, "sparse": is_sparse}
    else:
        multilabel = False
        multiclass = False
        task_type = classification_task_mapping[target_type]
        if task_type == MULTILABEL_CLASSIFICATION:
            multilabel = True
        if task_type == MULTICLASS_CLASSIFICATION:
            multiclass = True
        if task_type == BINARY_CLASSIFICATION:
            pass
        dataset_properties = {
            "multilabel": multilabel,
            "multiclass": multiclass,
            "sparse": is_sparse,
        }

    cs = spaces[task](
        feat_type=feat,
        dataset_properties=dataset_properties,
        include=include,
        exclude=exclude,
        random_state=None,
    )._get_hyperparameter_search_space(feat, include, exclude, dataset_properties)

    return cs, dataset_properties
