from __future__ import annotations

from functools import partial

from phem.base_utils.diversity_metrics import get_loss_correlation_metric
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorFunction


def ensemble_size(weights) -> float:
    return sum(weights != 0)


def get_ensemble_size_behavior_fun(max_models):
    return BehaviorFunction(ensemble_size, ["weights"], (0, max_models), "none", name="Ensemble Size")

def get_loss_correlation_behavior_fun(is_classification, abs=True):
    corr = get_loss_correlation_metric(is_classification, abs=abs)
    range = (0, 1) if abs else (-1, 1)
    pred_format = "proba" if is_classification else "raw"
    
    return BehaviorFunction(
        partial(corr, checks=False),
        ["y_true", "Y_pred_base_models"],
        range, 
        pred_format,
        name=corr.name + "(Lower is more Diverse)",
    )