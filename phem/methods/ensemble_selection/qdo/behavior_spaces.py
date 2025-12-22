"""Selection of Pre-defined behavior spaces."""

from __future__ import annotations

from phem.methods.ensemble_selection.qdo.behavior_functions.basic import get_loss_correlation_behavior_fun


def get_bs_configspace_similarity_and_loss_correlation(is_classification, abs=True):
    # "bs_configspace_similarity_and_loss_correlation"
    from phem.methods.ensemble_selection.qdo.behavior_functions.basic import get_loss_correlation_behavior_fun
    from phem.methods.ensemble_selection.qdo.behavior_functions.implicit_diversity_metrics import (
        ConfigSpaceGowerSimilarity
    )
    from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace
    
    return BehaviorSpace([ConfigSpaceGowerSimilarity, get_loss_correlation_behavior_fun(is_classification, abs)])


def get_bs_ensemble_size_and_loss_correlation(is_classification, n_models=50, abs=True):
    # "bs_configspace_similarity_and_loss_correlation"
    from phem.methods.ensemble_selection.qdo.behavior_functions.basic import (
        get_ensemble_size_behavior_fun,
        get_loss_correlation_behavior_fun,
    )
    from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace

    return BehaviorSpace([get_ensemble_size_behavior_fun(n_models), get_loss_correlation_behavior_fun(is_classification, abs)])
