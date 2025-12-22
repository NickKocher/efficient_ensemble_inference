import os

def get_final_model_paths(base_dir, config_ids, method):
    ret = []
    for config_id in config_ids:
        if os.path.exists(os.path.join(base_dir, "models", "refit", f"{config_id}.pkl")):
            ret.append(os.path.join(base_dir, "models", "refit", f"{config_id}.pkl"))
        else:
            ret.append(os.path.join(base_dir, "models", f"ens_{method}", f"{config_id}.pkl"))
    return ret

def filter_refit_ens_models(base_dir, method, config_ids):
    ret = []
    for config_id in config_ids:
        if (
            os.path.exists(os.path.join(base_dir, "models", "refit", f"{config_id}.pkl")) or
            os.path.exists(os.path.join(base_dir, "models", f"ens_{method}", f"{config_id}.pkl"))
        ):
            ret.append(config_id)
    return ret

def get_final_model_path(base_dir, config_id, method):
    if os.path.exists(os.path.join(base_dir, "models", "refit", f"{config_id}.pkl")):
        return os.path.join(base_dir, "models", "refit", f"{config_id}.pkl")
    else:
        return os.path.join(base_dir, "models", f"ens_{method}", f"{config_id}.pkl")