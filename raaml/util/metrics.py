from collections.abc import Callable
from sklearn.metrics import log_loss, root_mean_squared_error, accuracy_score
from phem.base_utils.custom_metrics.roc_auc import roc_auc_score
from phem.base_utils.metrics import AbstractMetric
import numpy as np
from sklearn.utils.validation import _check_y, check_array
import warnings


def min_wrapper_roc_auc(y_true, y_pred, labels=None):
    """Wrapper for roc_auc_score to return 1 - score."""
    return 1 - roc_auc_score(y_true, y_pred, labels=labels)


def min_wrapper_accuracy(y_true, y_pred, labels=None):
    return 1 - accuracy_score(y_true, y_pred)


def log_loss_wrapper(y_true, y_pred, labels=None):
    # Normalize predictions to sum to 1, as log loss checks are very restrictive (i.e. tolerances are small)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)    
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return log_loss(y_true, y_pred, labels=labels)


def get_default_metric(is_classification, n_classes = None):
    if is_classification:
        if n_classes is None:
            raise ValueError("Number of classes must be provided for classification tasks.")
        if n_classes == 2:
            return make_raaml_metric(
                metric_func=min_wrapper_roc_auc,
                metric_name="1 - ROC AUC",
                maximize=False,
                classification=True,
                always_transform_conf_to_pred=True,
                optimum_value=0,
                max_value=1,
                requires_confidences=True,
            )
        else:
            return make_raaml_metric(
                metric_func=log_loss_wrapper,
                metric_name="Log Loss",
                maximize=False,
                classification=True,
                always_transform_conf_to_pred=False,
                optimum_value=0,
                max_value=np.inf,
                requires_confidences=True,
            )
    else:
        return make_raaml_metric(
            metric_func=root_mean_squared_error,
            metric_name="RMSE",
            maximize=False,
            classification=False,
            always_transform_conf_to_pred=True,
            requires_confidences=False,
            optimum_value=0,
            max_value=np.inf,
        )

def get_accuracy_score():
    return make_raaml_metric(
        metric_func=min_wrapper_accuracy,
        metric_name="1- Accuracy",
        maximize=False,
        classification=True,
        always_transform_conf_to_pred=True,
        optimum_value=0,
        max_value=1,
        requires_confidences=False,
        only_positive_class=False
    )



# -- Metric Utils
def make_raaml_metric(
    metric_func: Callable,
    metric_name: str,
    maximize: bool,
    classification: bool,
    always_transform_conf_to_pred: bool,
    optimum_value: int,
    max_value: int,
    pos_label: int = 1,
    requires_confidences: bool = False,
    only_positive_class: bool = False
):
    """Make a metric that has additional information.

    Parameters
    ----------
    metric_func: Callable
        The metric function to call.
        We expect it to be metric_func(y_true, y_pred) with y_pred potentially being
        probabilities instead of classes.
    metric_name: str
        Name of the metric
    maximize: bool
        Whether to maximize the metric or not
    classification: bool
        If it is a classification metric or not
    always_transform_conf_to_pred: bool
        Set to True if the metric can not handle confidences and only accepts predictions (only for classification)
    optimum_value: int
        The maximal value the metric can reach (used to compute the loss).
    max_value: int
        The maximum value the metric can reach (used for hypervolume computation).
    pos_label: int, default=1
        Index of the label used as positive label (relevant only for binary classification metrics)
    requires_confidences: bool, default=False
        If the metric requires confidences.
    only_positive_class: bool, default=False
        Only relevant if requires_confidences is True. If only_positive_class is true, only the positive class
        values are passed. This is only needed for binary classification. Ignored if always_transform_conf_to_pred is
        True.
    """
    return RAAMLMetric(
        metric_func,
        metric_name,
        maximize,
        classification,
        always_transform_conf_to_pred,
        optimum_value,
        max_value,
        pos_label,
        requires_confidences,
        only_positive_class
    )


class RAAMLMetric(AbstractMetric):
    
    def __init__(
        self,
        metric,
        name,
        maximize,
        classification,
        transform_conf_to_pred,
        optimum_value,
        max_value,
        pos_label,
        requires_confidences,
        only_positive_class
    ):
        super().__init__(
            metric=metric,
            name=name,
            maximize=maximize,
            classification=classification,
            transform_conf_to_pred=transform_conf_to_pred,
            optimum_value=optimum_value,
            pos_label=pos_label,
            requires_confidences=requires_confidences,
            only_positive_class=only_positive_class,
        )
        self.max_value = max_value
        self.labels = None
          
    def __eq__(self, other):
        # For simplicity, we only compare by name
        return self.name == other.name
    
    
    
        
    def set_labels(self, labels):
        self.labels = labels    
        
    def __call__(self, y_true, y_pred, to_loss = False, checks=True):
        
        if self.labels is None and self.classification:
            raise ValueError("Labels must be set before calling the metric.")
                
        return super().__call__(y_true, y_pred, to_loss, checks, labels=self.labels)