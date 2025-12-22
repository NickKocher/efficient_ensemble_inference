import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from matplotlib.pyplot import ylabel
from pytabkit.models.sklearn.sklearn_interfaces import (
    RealMLP_TD_Classifier, RealMLP_TD_Regressor, 
    TabM_D_Classifier, TabM_D_Regressor,
    XGB_TD_Classifier, XGB_TD_Regressor,
    CatBoost_TD_Classifier, CatBoost_TD_Regressor
)
import numpy as np
from raaml.config_spaces.config_space_factory import PipelineFactory
import threadpoolctl
import pandas as pd
import time




class GBMLPPipelineFactory(PipelineFactory):
    
    def __init__(self, X, y, meta, include=None, exclude=None):
        self.config_space = get_config_space(include, exclude, meta)
        super().__init__(X, y, meta, include, exclude)
        
    def get_config_space(self, seed=None):
        self.config_space.seed(seed)
        return self.config_space
    
    def load_pipeline(self, config, seed):
        return load_model(self.meta["task_type"] == "classification", config, seed)
    

def apply_hybrid_balancing(X: pd.DataFrame, y: pd.Series):
    # Find class counts
    cls_count = len(y) // len(y.unique())
    class_counts = y.value_counts()
        
    indices = []
    for cls, count in class_counts.items():
        cls_idx = y[y == cls].index
        if count < cls_count:
            # Upsample
            sampled_idx = np.random.choice(cls_idx, size=cls_count, replace=True)
        else:
            # Downsample
            sampled_idx = np.random.choice(cls_idx, size=cls_count, replace=False)
        indices.append(sampled_idx)
    
    # Concatenate all sampled indices
    balanced_idx = np.concatenate(indices)
    np.random.shuffle(balanced_idx)  # shuffle to mix classes
        
    # Return balanced data
    return X.loc[balanced_idx].reset_index(drop=True), y.loc[balanced_idx].reset_index(drop=True)
    
    
def apply_upsampling(X: pd.DataFrame, y: pd.Series):
    # Find class counts
    class_counts = y.value_counts()
    max_count = class_counts.max()
        
    indices = []
    for cls, count in class_counts.items():
        cls_idx = y[y == cls].index
        # Sample with replacement to reach max_count
        sampled_idx = np.random.choice(cls_idx, size=max_count, replace=True)
        indices.append(sampled_idx)
    
    # Concatenate all sampled indices
    balanced_idx = np.concatenate(indices)
    np.random.shuffle(balanced_idx)  # shuffle to mix classes
        
    # Return balanced data
    return X.loc[balanced_idx].reset_index(drop=True), y.loc[balanced_idx].reset_index(drop=True)


def apply_downsampling(X: pd.DataFrame, y: pd.Series):   
    # Find class counts
    class_counts = y.value_counts()
    min_count = class_counts.min()
        
    indices = []
    for cls, count in class_counts.items():
        cls_idx = y[y == cls].index
        # Sample without replacement to reach min_count
        sampled_idx = np.random.choice(cls_idx, size=min_count, replace=False)
        indices.append(sampled_idx)
    
    # Concatenate all sampled indices
    balanced_idx = np.concatenate(indices)
    np.random.shuffle(balanced_idx)  # shuffle to mix classes
        
    # Return balanced data
    return X.loc[balanced_idx].reset_index(drop=True), y.loc[balanced_idx].reset_index(drop=True)
   
   
def map_balancing(kwargs):
    balancing_str = kwargs.pop("balancing")
    return {
        "upsampling" : apply_upsampling,
        "downsampling" : apply_downsampling,
        "hybrid" : apply_hybrid_balancing,
    }.get(balancing_str, None)

class MultiThreadingRealMLPClassifier(RealMLP_TD_Classifier):
    
    def __init__(self, **kwargs):
        self.balancing_fn_ = map_balancing(kwargs)
        super().__init__(**kwargs)
    
    def set_n_threads(self, n):
        self.n_threads_ = n
        
    def set_timeout(self, timeout):
        self.timeout_ = timeout
    
    def fit(self, X, y):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            start = time.time()
            if self.balancing_fn_ is not None:
                X, y = self.balancing_fn_(X, y)
            end = time.time()
            seconds_balancing = int(np.ceil(end - start))
            return super().fit(X, y, time_to_fit_in_seconds=(self.timeout_ - seconds_balancing) if hasattr(self, "timeout_") else None)      
        
    def predict(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict(X)
    
    def predict_proba(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict_proba(X)
    
class MultiThreadingRealMLPRegressor(RealMLP_TD_Regressor):
    
    def set_n_threads(self, n):
        self.n_threads_ = n
    
    def set_timeout(self, timeout):
        self.timeout_ = timeout
    
    def fit(self, X, y):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().fit(X, y, time_to_fit_in_seconds=self.timeout_ if hasattr(self, "timeout_") else None)      
        
    def predict(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict(X)
        
        
class MultiThreadingTabMClassifier(TabM_D_Classifier):
    
    def __init__(self, **kwargs):
        self.balancing_fn_ = map_balancing(kwargs)
        super().__init__(**kwargs)
        
    def set_n_threads(self, n):
        self.n_threads_ = n
    
    def fit(self, X, y):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            if self.balancing_fn_ is not None:
                X, y = self.balancing_fn_(X, y)
            return super().fit(X, y)      
        
    def predict(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict(X)
    
    def predict_proba(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict_proba(X)
        
        
class MultiThreadingTabMRegressor(TabM_D_Regressor):
    
    def set_n_threads(self, n):
        self.n_threads_ = n
    
    def fit(self, X, y):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().fit(X, y)      
        
    def predict(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict(X)
        
        
class MultiThreadingXGBClassifier(XGB_TD_Classifier):
    
    def __init__(self, **kwargs):
        self.balancing_fn_ = map_balancing(kwargs)
        super().__init__(**kwargs)
    
    def set_n_threads(self, n):
        self.n_threads_ = n
    
    def fit(self, X, y):    
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            if self.balancing_fn_ is not None:
                X, y = self.balancing_fn_(X, y)    
            return super().fit(X, y)      
        
    def predict(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict(X)
    
    def predict_proba(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict_proba(X)
        
class MultiThreadingXGBRegressor(XGB_TD_Regressor):
    
    def set_n_threads(self, n):
        self.n_threads_ = n
    
    def fit(self, X, y):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().fit(X, y)      
        
    def predict(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict(X)
        
class MultiThreadingCatBoostClassifier(CatBoost_TD_Classifier):
    
    def set_n_threads(self, n):
        self.n_threads_ = n
    
    def fit(self, X, y):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().fit(X, y)      
        
    def predict(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict(X)
    
    def predict_proba(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict_proba(X)
        
class MultiThreadingCatBoostRegressor(CatBoost_TD_Regressor):
    
    def set_n_threads(self, n):
        self.n_threads_ = n
    
    def fit(self, X, y):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().fit(X, y)      
        
    def predict(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict(X)

    def predict_proba(self, X):
        with threadpoolctl.threadpool_limits(limits=self.n_threads_):
            return super().predict_proba(X)
        

        

def get_config_space(include, exclude, meta):
    space = CS.ConfigurationSpace()

    if include is not None and exclude is not None:
        raise ValueError("Cannot specify both include and exclude")
         
    models = set(["realmlp", "tabm", "xgboost"])
    if include is not None:
        models = models.intersection(include)
    if exclude is not None:
        models = models.difference(exclude)
        
    models = list(models)
    default = [m for m in ["xgboost" "realmlp", "tabm"] if m in models][0]
    
    
    model_hp = CSH.CategoricalHyperparameter(
        name="model",
        choices=models,
        default_value=default,
    )
    space.add(model_hp)
    
    
    if meta["task_type"] == "classification":
        balancing_hp = CSH.CategoricalHyperparameter(
            name="balancing",
            choices=["upsampling", "downsampling", "hybrid", "none"],
            default_value="downsampling",
        )
        space.add(balancing_hp)
    
    
    if "realmlp" in models:
        space.add_configuration_space(
            prefix = "realmlp",
            configuration_space = get_config_space_real_mlp(),
            parent_hyperparameter = {"parent" : model_hp, "value": "realmlp"},
        )
    
    if "xgboost" in models:
        space.add_configuration_space(
            prefix = "xgboost",
            configuration_space = get_xgboost_config_space(),
            parent_hyperparameter = {"parent" : model_hp, "value": "xgboost"},
        )
    
    if "tabm" in models:
        space.add_configuration_space(
            prefix = "tabm",
            configuration_space = get_tabm_config_space(),
            parent_hyperparameter = {"parent" : model_hp, "value": "tabm"},
        )
    
    """if "catboost" in models:
        space.add_configuration_space(
            prefix = "catboost",
            configuration_space = get_catboost_config_space(),
            parent_hyperparameter = {"parent" : model_hp, "value": "catboost"},
        )"""
    
    return space

def load_model(classification, config, seed, verbosity=0):
    clf_map = {"realmlp" : MultiThreadingRealMLPClassifier, "xgboost" : MultiThreadingXGBClassifier, "tabm" : MultiThreadingTabMClassifier, "catboost" : MultiThreadingCatBoostClassifier}
    reg_map = {"realmlp" : MultiThreadingRealMLPRegressor, "xgboost" : MultiThreadingXGBRegressor, "tabm" : MultiThreadingTabMRegressor, "catboost" : MultiThreadingCatBoostRegressor}
    
    params = dict(config)
    model_type = str(params.pop("model"))
    
    if model_type == "realmlp":
        wd = params.pop("realmlp:wd")
        params["wd"] = params.pop("realmlp:wd_low") if wd == "low" else params.pop("realmlp:wd_high")
        
        ls_eps = params.pop("realmlp:ls_eps")
        params["ls_eps"] = 0.0 if ls_eps == "zero" else params.pop("realmlp:ls_eps_pos")
        
        n_hidden = params.pop("realmlp:n_hidden")
        n_layers = params.pop("realmlp:n_layers")
        params["hidden_sizes"] = [n_hidden] * n_layers
        
        params["batch_size"] = 256
        
    elif model_type == "xgboost":   
        reg_lambda = params.pop("xgboost:reg_lambda")
        params["reg_lambda"] = 0.0 if reg_lambda == "zero" else params.pop("xgboost:reg_lambda_pos")
        
        reg_alpha = params.pop("xgboost:reg_alpha")
        params["alpha"] = 0.0 if reg_alpha == "zero" else params.pop("xgboost:reg_alpha_pos")
        
        reg_gamma = params.pop("xgboost:reg_gamma")
        params["gamma"] = 0.0 if reg_gamma == "zero" else params.pop("xgboost:reg_gamma_pos")
        
        
    elif model_type == "tabm":
        wd = params.pop("tabm:wd")
        params["weight_decay"] = 0.0 if wd == "zero" else params.pop("tabm:wd_pos")
        
        params.update({
            "batch_size" : 256,
            "patience": 16,
            "allow_amp" : True,
            "arch_type" : "tabm-mini",
            "tabm_k" : 32,
            "gradient_clipping_norm" : 1.0,
            "share_training_batches" : False,
            "num_emb_type" : "pwl",
        })

        
    params.update({
        "random_state" : seed,
        "verbosity" : verbosity,
    })
        
    params = {k.replace(f"{model_type}:", "") : v for k, v in params.items()}
        
    if classification:
        est = clf_map[model_type](**params)
    else:
        est = reg_map[model_type](**params)

    return est
    

    
    
def get_config_space_real_mlp():    
    space_real_mlp = CS.ConfigurationSpace()
    
    numb_emb_type = CSH.CategoricalHyperparameter(
        name="num_emb_type",
        choices=["none", "pbld", "pl", "plr"],
        default_value="none",
    )
    
    add_front_scale = CSH.CategoricalHyperparameter(
        name="add_front_scale",
        choices=[True, False],
        weights=[0.6, 0.4],
        default_value=True,
    )
    
    n_hidden = CSH.UniformIntegerHyperparameter(
        name="n_hidden",
        lower=64,
        upper=512,
        default_value=256,
        log=True,
    )
    
    n_layers = CSH.UniformIntegerHyperparameter(
        name="n_layers",
        lower=1,
        upper=5,
        default_value=3,
    )
    
    lr = CSH.UniformFloatHyperparameter(
        name="lr",
        lower=1e-2,
        upper=5e-1,
        default_value=1e-1,
        log=True,
    )
    
    p_drop = CSH.UniformFloatHyperparameter(
        name="p_drop",
        lower=0.0,
        upper=0.6,
        default_value=0.3,
    )
    
    wd = CSH.CategoricalHyperparameter(
        name="wd",
        choices=["low", "high"],
        default_value="low",
    )
    wd_low = CSH.UniformFloatHyperparameter(
        name="wd_low",
        lower=0,
        upper=1e-3,
        default_value=0,
    )
    wd_high = CSH.UniformFloatHyperparameter(
        name="wd_high",
        lower=1e-3,
        upper=1e-1,
        default_value=1e-3,
        log=True
    )
    wd_low_cond = CS.EqualsCondition(wd_low, wd, "low", )
    wd_high_cond = CS.EqualsCondition(wd_high, wd, "high", )
    
    plr_sigma = CSH.UniformFloatHyperparameter(
        name="plr_sigma",
        lower=1e-2,
        upper=1e2,
        default_value=1e-1,
        log=True,
    )
    
    act = CSH.CategoricalHyperparameter(
        name="act",
        choices=["relu", "selu", "mish", "silu", "gelu"],
        default_value="relu",
    )
    
    use_parametric_act = CSH.CategoricalHyperparameter(
        name="use_parametric_act",
        choices=[False, True],
        default_value=False,
    )
    
    p_drop_sched = CSH.CategoricalHyperparameter(
        name="p_drop_sched",
        choices=["flat_cos", "constant"],
        default_value="flat_cos",
    )
    
    wd_sched = CSH.CategoricalHyperparameter(
        name="wd_sched",
        choices=["flat_cos", "constant"],
        default_value="flat_cos",
    )
    
    ls_eps = CSH.CategoricalHyperparameter(
        name="ls_eps",
        choices=["zero", "positive"],
        default_value="zero",
    )
    ls_eps_pos = CSH.UniformFloatHyperparameter(
        name="ls_eps_pos",
        lower=0.0,
        upper=0.2,
        default_value=0.0,
    )
    ls_eps_cond = CS.EqualsCondition(ls_eps_pos, ls_eps, "positive")
    
    lr_sched = CSH.CategoricalHyperparameter(
        name="lr_sched",
        choices=["coslog4", "cos"],
        default_value="cos",
    )
    
    sq_mom = CSH.UniformFloatHyperparameter(
        name="sq_mom",
        lower=0.9,
        upper=0.999,
        default_value=0.9,
        log=True
    )
    
    plr_lr_factor = CSH.UniformFloatHyperparameter(
        name="plr_lr_factor",
        lower=3e-2,
        upper=3e-1,
        default_value=3e-2,
    )
    
    space_real_mlp.add([
        numb_emb_type, add_front_scale, n_hidden, n_layers, lr, p_drop, wd, wd_low, wd_high, wd_low_cond, wd_high_cond, plr_sigma, act, use_parametric_act, p_drop_sched, wd_sched, ls_eps, ls_eps_pos, ls_eps_cond, lr_sched, sq_mom, plr_lr_factor
    ])
    
    return space_real_mlp

def get_xgboost_config_space():
    xgb_space = CS.ConfigurationSpace()
    
    n_estimators = CSH.UniformIntegerHyperparameter(
        name="n_estimators",
        lower=2,
        upper=2000,
        default_value=1000,
        log=True
    )
    max_depth = CSH.UniformIntegerHyperparameter(
        name="max_depth",
        lower=2,
        upper=10,
        default_value=3,
    )
    subsample = CSH.UniformFloatHyperparameter(
        name="subsample",
        lower=1e-3,
        upper=1,
        default_value=1,
    )
    colsample_bytree = CSH.UniformFloatHyperparameter(
        name="colsample_bytree",
        lower=0.1,
        upper=1,
        default_value=1,
    )
    colsample_bylevel = CSH.UniformFloatHyperparameter(
        name="colsample_bylevel",
        lower=0.1,
        upper=1,
        default_value=1,
    )
    min_child_weight = CSH.UniformFloatHyperparameter(
        name="min_child_weight",
        lower=np.exp(-16),
        upper=np.exp(5),
        default_value=1,
        log=True
    )
    reg_alpha = CSH.CategoricalHyperparameter(
        name="reg_alpha",
        choices=["zero", "positive"],
        default_value="zero",
    )
    reg_alpha_pos = CSH.UniformFloatHyperparameter(
        name="reg_alpha_pos",
        lower=np.exp(-16),
        upper=np.exp(2),
        default_value=1,
        log=True
    )
    reg_alpha_cond = CS.EqualsCondition(reg_alpha_pos, reg_alpha, "positive")
    
    reg_lambda = CSH.CategoricalHyperparameter(
        name="reg_lambda",
        choices=["zero", "positive"],
        default_value="zero",
    )
    reg_lambda_pos = CSH.UniformFloatHyperparameter(
        name="reg_lambda_pos",
        lower=np.exp(-16),
        upper=np.exp(2),
        default_value=1,
        log=True
    )
    reg_lambda_cond = CS.EqualsCondition(reg_lambda_pos, reg_lambda, "positive")
    
    reg_gamma = CSH.CategoricalHyperparameter(
        name="reg_gamma",
        choices=["zero", "positive"],
        default_value="zero",
    )
    reg_gamma_pos = CSH.UniformFloatHyperparameter(
        name="reg_gamma_pos",
        lower=np.exp(-16),
        upper=np.exp(2),
        default_value=1,
        log=True
    )
    reg_gamma_cond = CS.EqualsCondition(reg_gamma_pos, reg_gamma, "positive")
    
    xgb_space.add([
        n_estimators,
        max_depth,
        subsample,
        colsample_bytree,
        colsample_bylevel,
        min_child_weight,
        reg_alpha,
        reg_alpha_pos,
        reg_alpha_cond,
        reg_lambda,
        reg_lambda_pos,
        reg_lambda_cond,
        reg_gamma,
        reg_gamma_pos,
        reg_gamma_cond,
    ])
    
    return xgb_space

def get_tabm_config_space():
    tabm_space = CS.ConfigurationSpace()
    
    lr = CSH.UniformFloatHyperparameter(
        name="lr",
        lower=1e-4,
        upper=3e-3,
        default_value=1e-3,
        log=True
    )
    
    wd = CSH.CategoricalHyperparameter(
        name="wd",
        choices=["zero", "positive"],
        default_value="zero",
    )
    wd_pos = CSH.UniformFloatHyperparameter(
        name="wd_pos",
        lower=1e-4,
        upper=1e-1,
        default_value=1e-3,
        log=True
    )
    wd_cond = CS.EqualsCondition(wd_pos, wd, "positive")
    
    n_blocks = CSH.UniformIntegerHyperparameter(
        name="n_blocks",
        lower=1,
        upper=5,
        default_value=3,
    )
    
    d_block = CSH.UniformIntegerHyperparameter(
        name="d_block",
        lower=32,
        upper=512,
        default_value=128,
    )
    
    dropout = CSH.UniformFloatHyperparameter(
        name="dropout",
        lower=0.0,
        upper=0.5,
        default_value=0.0,
    )
    
    d_embedding = CSH.UniformIntegerHyperparameter(
        name="d_embedding",
        lower=8,
        upper=32,
        default_value=16,
    )
    
    num_emb_n_bins = CSH.UniformIntegerHyperparameter(
        name="num_emb_n_bins",
        lower=2,
        upper=128,
        default_value=32,
    )
    
    tabm_space.add([
        lr,
        wd,
        wd_pos,
        wd_cond,
        n_blocks,
        d_block,
        dropout,
        d_embedding,
        num_emb_n_bins,
    ])
    
    return tabm_space



def get_catboost_config_space():
    cs = CS.ConfigurationSpace()
    
    n_estimators = CSH.UniformIntegerHyperparameter(
        name="n_estimators",
        lower=2,
        upper=2000,
        default_value=100,
        log=True,
    )
    
    learning_rate = CSH.UniformFloatHyperparameter(
        name="lr",
        lower=np.exp(-5),
        upper=1,
        default_value=np.exp(-5),
        log=True
    )
    
    random_strength = CSH.UniformIntegerHyperparameter(
        name="random_strength",
        lower=1,
        upper=20,
        default_value=1,
    )
    
    one_hot_max_size = CSH.UniformIntegerHyperparameter(
        name="one_hot_max_size",
        lower=0,
        upper=25,
        default_value=1,
    )
    
    l2_leaf_reg = CSH.UniformFloatHyperparameter(
        name="l2_leaf_reg",
        lower=1,
        upper=10,
        default_value=1,
        log=True
    )
    
    bagging_temperature = CSH.UniformFloatHyperparameter(
        name="bagging_temperature",
        lower=0,
        upper=1,
        default_value=0,
    )
    
    subsample = CSH.UniformFloatHyperparameter(
        name="subsample",
        lower=1e-3,
        upper=1,
        default_value=1
    )
    
    max_depth = CSH.UniformIntegerHyperparameter(
        name="max_depth",
        lower=1,
        upper=10,
        default_value=1,
    )
    
    colsample_bylevel = CSH.UniformFloatHyperparameter(
        name="colsample_bylevel",
        lower=0.1,
        upper=1,
        default_value=1,
    )
    
    leaf_estimation_iterations = CSH.UniformIntegerHyperparameter(
        name="leaf_estimation_iterations",
        lower=1,
        upper=10,
        default_value=1,
    )
    
    cs.add([
        n_estimators,
        learning_rate,
        random_strength,
        one_hot_max_size,
        l2_leaf_reg,
        bagging_temperature,
        subsample,
        max_depth,
        colsample_bylevel,
        leaf_estimation_iterations
    ])
    
    return cs