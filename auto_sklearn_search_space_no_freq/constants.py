# -*- encoding: utf-8 -*-
import numpy as np

BINARY_CLASSIFICATION = 1
MULTICLASS_CLASSIFICATION = 2
MULTILABEL_CLASSIFICATION = 3
REGRESSION = 4
MULTIOUTPUT_REGRESSION = 5

REGRESSION_TASKS = [REGRESSION, MULTIOUTPUT_REGRESSION]
CLASSIFICATION_TASKS = [
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
]

TASK_TYPES = REGRESSION_TASKS + CLASSIFICATION_TASKS

TASK_TYPES_TO_STRING = {
    BINARY_CLASSIFICATION: "binary.classification",
    MULTICLASS_CLASSIFICATION: "multiclass.classification",
    MULTILABEL_CLASSIFICATION: "multilabel.classification",
    REGRESSION: "regression",
    MULTIOUTPUT_REGRESSION: "multioutput.regression",
}

STRING_TO_TASK_TYPES = {
    "binary.classification": BINARY_CLASSIFICATION,
    "multiclass.classification": MULTICLASS_CLASSIFICATION,
    "multilabel.classification": MULTILABEL_CLASSIFICATION,
    "regression": REGRESSION,
    "multioutput.regression": MULTIOUTPUT_REGRESSION,
}

MAX_THREADS = 64

def get_numeric_task_type(task_type: str, y):
    """
    Maps a string task type to a numeric task type as used in the constants.

    Parameters
    ----------
    task_type: str
        String task type, either "classification" or "regression"
    y: array-like
        Targets/labels of the data

    Returns
    -------
    int
        Numeric task type, either BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION or REGRESSION
    """
    if task_type == "classification":
        if len(np.unique(y)) == 2:
            task_type = BINARY_CLASSIFICATION
        else:
            task_type = MULTICLASS_CLASSIFICATION
    else:
        task_type = REGRESSION
    return task_type
