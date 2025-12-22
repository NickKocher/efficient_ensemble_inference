import openml
openml.config.set_root_cache_directory('/storage/work/paessens/.openml_cache')

import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import gc
from collections import Counter
import psutil
import pandas  as pd
from pympler import muppy, summary
import logging

from raaml.util.warning_filter import filter_warnings
from raaml.util.split import split_indices


def get_resources(task_id):
    # 16GB: 87
    # 32GB: 6
    # 64GB: 11
    
    mapping = {
        233213: 64,
        233214: 32,
        317614: 64,
        359929: 64,
        10090: 32,
        168909: 32,
        189355: 32,
        189356: 64,
        359976: 64,
        359986: 64,
        359988: 64,
        359989: 64,
        359994: 32,
        360112: 64,
        360113: 32,
        360114: 64,
        360975: 64,
    }
    
    if task_id not in mapping:
        return 16, 1
    
    mem = mapping[task_id]
    return mem, int(mem / 16)


def load_openml_data_new(task_id):
    logging.info(f"Loading AMLB task: {task_id}")
    # Load the task and dataset, get train-test split
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    features = dataset.features
    target = dataset.default_target_attribute
    
    del dataset
    gc.collect()
    
    target_feature = [f for f in features.values() if f.name == target][0]
    task_type = "classification" if target_feature.data_type == "nominal" else "regression"
    feature_data_types = [f for f in features.values() if f.name in X.columns]
        
    if any(f.data_type == "date" or f.data_type == "string" for f in feature_data_types):
        raise ValueError("Date and text features are not supported yet.")
        
    map = {"nominal": "categorical", "numeric": "numerical"}
    feature_types = {f.name : map[f.data_type] for i, f in enumerate(feature_data_types) }
    
    #  Memory optimization for X 
    for col, f in feature_types.items():
        if f == "categorical":
            # Convert categorical columns to categorical
            X[col] = X[col].astype("category")
        elif f == "numerical":
            X[col] = X[col].astype("float32")
        
        gc.collect()
    
    meta = {
        "task_id": task_id,
        "task_type": task_type,
        "feature_types": feature_types,
    }
    
    #  Handle target variable 
    if task_type == "classification":
        # Convert target to categorical
        y = pd.Series(y, dtype="category", name=target)
        
        meta["classes"] = np.sort(y.cat.categories.tolist())
        meta["n_classes"] = y.cat.categories.size
    else:
        y = y.astype("float32")
        
    gc.collect()
        
    train_indices, test_indices = split_indices(len(X), train_size=0.8, stratify=y if meta["task_type"] == "classification" else None, seed=0)
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    
    del X
    gc.collect()
    
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    del y, train_indices, test_indices
    gc.collect()
    
    # Select numerical feature names
    logging.info("Applying numerical imputation...")
    num_cols = [col for col, t in feature_types.items() if t == "numerical"]
    if num_cols:
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        # Fit-transform train
        X_train_num = X_train[num_cols].to_numpy(dtype="float32")
        X_train_num = imputer.fit_transform(X_train_num)
        X_train[num_cols] = X_train_num
        del X_train_num
        gc.collect()
        
        # Transform test
        X_test_num = X_test[num_cols].to_numpy(dtype="float32")
        X_test_num = imputer.transform(X_test_num)
        X_test[num_cols] = X_test_num
        del X_test_num
        gc.collect()
        
    gc.collect()
            
    return X_train, X_test, y_train, y_test, meta



def load_openml_data_splitted_and_imputed(task_id):
    
    X, y, meta, _ = load_openml_data(task_id)
    
    train_indices, test_indices = split_indices(len(X), train_size=0.8, stratify=y if meta["task_type"] == "classification" else None, seed=0)
    
    X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    del X, y, train_indices, test_indices
    gc.collect()
    
    cat_ind = [i for i in meta["feature_types"].keys() if meta["feature_types"][i] == "categorical"]
    num_ind = [i for i in meta["feature_types"].keys() if meta["feature_types"][i] == "numerical"]
    
    if len(num_ind) > 0:
        imputer = SimpleImputer(strategy="mean")
        X_train[:,num_ind] = imputer.fit_transform(X_train[:, num_ind])
        X_test[:, num_ind] = imputer.transform(X_test[:, num_ind])
                
    return X_train, X_test, y_train, y_test, meta



def load_openml_data(task_id, task_type=None, label_encoder=True):
    # Load the task and dataset, get train-test split
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    train_indices, _ = task.get_train_test_split_indices()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    features = dataset.features
    
    target_feature = [f for f in features.values() if f.name == dataset.default_target_attribute][0]
    
    if task_type is None:
        task_type = "classification" if target_feature.data_type == "nominal" else "regression"
        
    feature_data_types_unsorted = [f for f in features.values() if f.name in X.columns]
    feature_data_types = sorted(feature_data_types_unsorted, key=lambda f: list(X.columns).index(f.name))
        
    if any(f.data_type == "date" for f in feature_data_types):
        raise ValueError("Date features are not supported yet. Please remove them from the dataset.")
        
    map = {"nominal": "categorical", "numeric": "numerical", "string": "text"}
    feature_types = {i : map[f.data_type] for i, f in enumerate(feature_data_types) }
    
    meta = {
        "task_id": task_id,
        "task_type": task_type,
        "feature_types": feature_types,
    }
        
    X = np.array(X)
    y = np.array(y)
    
    if task_type == "classification":
        if label_encoder:
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        meta["classes"] = np.unique(y)
        meta["n_classes"] = len(meta["classes"])
        
    return X, y, meta, train_indices


def load_openml_data_subsampled(task_id):
    # Load full data
    X, y, meta, _ = load_openml_data(task_id, label_encoder=False)
        
    # Apply (stratified) subsampling
    if X.shape[0] > 45000:
        X, _, y, _ = train_test_split(X, y, train_size=45000, stratify=y if meta["task_type"] == "classification" else None, random_state=0)
        
    if meta["task_type"] == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        meta["classes"] = np.unique(y)
        meta["n_classes"] = len(meta["classes"])
    
    # Apply univariate feature selection
    if X.shape[1] > 500:
        feat_types = meta["feature_types"]
        
        # Apply Ordinal encoding temporarily
        if "categorical" in feat_types.values():
            cat_indices = [i for i in feat_types.keys() if feat_types[i] == "categorical"]
            X_num = X.copy()
            X_num[:,cat_indices] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X_num[:,cat_indices])
        else:
            X_num = X.copy()
            
        # Impute missing values temporarily
        imputer = SimpleImputer(strategy="mean")
        X_num = imputer.fit_transform(X_num)
        
        # Select max. 500 features using univariate feature selection
        with filter_warnings():
            selector = SelectKBest(k=500, score_func=f_classif if meta["task_type"] == "classification" else f_regression)
            selector.fit_transform(X_num, y)
        
        del X_num
        gc.collect()
                
        selected_idx = selector.get_support(indices=True)
        
        X = X[:, selected_idx]
        meta["feature_types"] = {i: feat_types[idx] for i, idx in enumerate(selected_idx)}
        
    # Apply 80/20 train-test split
    if meta["task_type"] == "classification":
        counts = Counter(y)
        rare_classes = [cls for cls, cnt in counts.items() if cnt < 2]

        if rare_classes:
            # indices of rare samples (all go to train set)
            rare_idx = np.isin(y, rare_classes)
            X_rare, y_rare = X[rare_idx], y[rare_idx]
            X_rest, y_rest = X[~rare_idx], y[~rare_idx]

            # stratified split only on classes with >=2 samples
            X_train, X_test, y_train, y_test = train_test_split(
                X_rest, y_rest, train_size=0.8, stratify=y_rest, random_state=0
            )

            # add the rare samples back to train set
            X_train = np.vstack([X_train, X_rare])
            y_train = np.hstack([y_train, y_rare])
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, stratify=y, random_state=0
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=0
        )
        
    del X, y, _
    gc.collect()
        
    return X_train, X_test, y_train, y_test, meta
    


def load_amlb_tasks(task_type=None, cache_path=None, n_cpu_cores=None):
    if task_type is None:
        reg = load_amlb_tasks(task_type="regression", cache_path=cache_path)
        clf = load_amlb_tasks(task_type="classification", cache_path=cache_path)
        
        if n_cpu_cores is not None:
            reg = [id for id in reg if get_resources(id)[1] == n_cpu_cores]
            clf = [id for id in clf if get_resources(id)[1] == n_cpu_cores]

        return [("regression", r) for r in reg] + [("classification", c) for c in clf]

    if cache_path is None:
        cache_path = f"./.cache/amlb_tasks_{task_type}.yaml"
        
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            tasks = yaml.safe_load(f)
        return tasks
    
    if task_type == "classification":
        suite = openml.study.get_suite(271)
    elif task_type == "regression":
        suite = openml.study.get_suite(269)
    else:
        raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")
    
    tasks = suite.tasks
    
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))
    
    with open(cache_path, 'w') as f:
        yaml.dump(tasks, f)
    
    return tasks


def get_amlb_meta():
    with open("./raaml/util/amlb_meta/regression_meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
        
    with open("./raaml/util/amlb_meta/classification_meta.yaml", "r") as f:
        meta.update(yaml.safe_load(f))
        
    task_ids = list(meta.keys())
    
    assert len(task_ids) == 104, f"Expected 104 AMLB tasks, but found {len(task_ids)}"
    data = []
    for task_id in task_ids:
        data.append({"task_id" : task_id, **meta[task_id]})
    return pd.DataFrame(data)


def zero_based_index_to_amlb_index(index, ret_task_type=False, n_cpu_cores=None):
    tasks = load_amlb_tasks(n_cpu_cores=n_cpu_cores)
    
    if ret_task_type:
        return tasks[index][1], tasks[index][0]

    return tasks[index][1]

def amlb_index_to_zero_based_index(index, n_cpu_cores=None):
    tasks = load_amlb_tasks(n_cpu_cores=n_cpu_cores)
    for i, (_, task_id) in enumerate(tasks):
        if task_id == index:
            return i
    raise ValueError(f"Task ID {index} not found.")