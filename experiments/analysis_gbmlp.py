from contextlib import nullcontext
import numpy as np
import time
import os
from pynisher import limit, TimeoutException, MemoryLimitException
import traceback
import pickle
import argparse
import pandas as pd
import tqdm
import pyRAPL

from raaml.util.warning_filter import filter_warnings
from raaml.util.openml_data import load_openml_data_new, zero_based_index_to_amlb_index, get_resources
from raaml.config_spaces.gb_mlp_space import GBMLPPipelineFactory
from raaml.util.progress_summary import ProgressSummary
from raaml.util.split import split_indices, create_cv_splits
from raaml.util.metrics import get_default_metric
from frequency.freq_api_amd import get_freq_values, SetFreqMgr
from frequency.freq_api import SetFreq, get_overall_min_freq, get_overall_max_freq


def fit(model, X, y, path, timeout, n_cores):   
    model.set_n_threads(n_cores)
    
    if hasattr(model, "set_timeout"):
        model.set_timeout(timeout * 0.7)
    
    with filter_warnings():
        model.fit(X, y)
            
    with open(path, "wb") as f:
        pickle.dump(model, f)


def train_models(task_id, X_train, y_train, meta, timeout, memout, N, seed, n_cores):
    n_success = 0
    n_error = 0
    n_timeout = 0
    n_memout = 0
    
    raaml_progress = ProgressSummary("train_gbmlp_analysis", amlb_tasks=True)
    
    fac = GBMLPPipelineFactory(X_train, y_train, meta)
    cs = fac.get_config_space(seed)
    
    print(cs.random)
    
    limited_fit = limit(fit, wall_time=(timeout, "s"), memory=(memout, "GB"))
    
    if not os.path.exists(f"./analysis/trained_gbmlp/{task_id}"):
        os.makedirs(f"./analysis/trained_gbmlp/{task_id}")
        
    configs_df = pd.DataFrame(columns=["config_id"] + list(cs.keys()) + ["status"])
            
    i = 0
    run_times = []
    while n_success < N:
        config = cs.sample_configuration()
        model = fac.load_pipeline(config, seed)
                
        try:
            print("Model type:", config["model"])
            
            start = time.perf_counter()
            limited_fit(model, X_train, y_train, f"./analysis/trained_gbmlp/{task_id}/{i}.pkl", timeout, n_cores)
            end = time.perf_counter()
            
            print("Fitting time:", end - start)
            
            run_times.append(end - start)
            
            configs_df.loc[len(configs_df)] = [i] + list(config.get_array()) + ["success"]
                
            n_success += 1
        except Exception as e:
            if isinstance(e, TimeoutException):
                n_timeout += 1
                print("Timeout!")
                configs_df.loc[len(configs_df)] = [i] + list(config.get_array()) + ["timeout"]
            elif isinstance(e, MemoryLimitException):
                n_memout += 1
                print("Memory out!")
                configs_df.loc[len(configs_df)] = [i] + list(config.get_array()) + ["memout"]
            else:
                n_error += 1
                print("Error!")
                print(traceback.format_exc())
                configs_df.loc[len(configs_df)] = [i] + list(config.get_array()) + ["error"]
                
        configs_df.to_csv(f"./analysis/trained_gbmlp/{task_id}/configs.csv", index=False)
        i += 1
        
        avg_runtime = (np.mean(run_times) * n_success + timeout * n_timeout) / (n_success + n_timeout) if n_success + n_timeout > 0 else "-"
        raaml_progress.set_progress(
            task_id, 
            float(n_success) / N, 
            meta_dict={"n_success": n_success, "n_error": n_error, "n_timeout": n_timeout, "n_memout": n_memout, "avg_succes_time": np.mean(run_times), "avg_time": avg_runtime},
            print_progress=True
        )


def _read_energy( core):
    with open(f"/sys/class/hwmon/hwmon3/energy{core+1}_input", "r") as f:
        energy = float(f.read()) / 1e6
    return energy

def _read_energies(cores):
    return np.array([_read_energy(core) for core in cores])

def predict_models(task_id, X_test, data, is_classification, n_cores, N, freq_scaling=False, platform="amd"):
    affinity = sorted(list(os.sched_getaffinity(0)))
    files = [filename for filename in os.listdir(f"./analysis/trained_gbmlp/{task_id}") if filename.endswith(".pkl")]
    n_counter = 0
    for filename in tqdm.tqdm(files):
        config_id = filename.split(".")[0]
        
        # Load model
        with open(f"./analysis/trained_gbmlp/{task_id}/{filename}", "rb") as f:
            try:
                model = pickle.load(f)
            except Exception as e:
                print(f"Error loading model {filename}: {e}")
                continue
            
        print("Model: ", type(model))
                    
        model.set_n_threads(n_cores)
        
        # Dummy predict
        with filter_warnings():
            model.predict(X_test)
        
        if not freq_scaling:
            with filter_warnings():
                if is_classification:
                    start_energy = _read_energies(affinity)
                    start = time.perf_counter()
                    model.predict_proba(X_test)
                    end = time.perf_counter()
                    end_energy = _read_energies(affinity)
                else:
                    start_energy = _read_energies(affinity)
                    start = time.perf_counter()
                    model.predict(X_test)
                    end = time.perf_counter()
                    end_energy = _read_energies(affinity)
                        
            #print("Prediction with ", n_cores, "threads:", end - start, "energy:", np.sum(end_energy - start_energy))
                    
            data.append({"model_type" : type(model), "config_id" : config_id, "n_threads": n_cores, "inference_time": end - start, "energy_consumption": np.sum(end_energy - start_energy)})
        elif platform == "amd":
            for freq in get_freq_values():
                with filter_warnings():
                    if is_classification:
                        start_energy = _read_energies(affinity)
                        start = time.perf_counter()
                        with SetFreqMgr(freq):
                            model.predict_proba(X_test)
                        end = time.perf_counter()
                        end_energy = _read_energies(affinity)
                    else:
                        start_energy = _read_energies(affinity)
                        start = time.perf_counter()
                        with SetFreqMgr(freq):
                            model.predict(X_test)
                        end = time.perf_counter()
                        end_energy = _read_energies(affinity)
                                        
                data.append({"model_type" : type(model), "config_id" : config_id, "n_threads": n_cores, "frequency" : freq, "inference_time": end - start, "energy_consumption": np.sum(end_energy - start_energy)})
        elif platform == "intel":
            pyRAPL.setup()
            measurement = pyRAPL.Measurement("cpu_energy")
            
            for freq in list(np.linspace(get_overall_min_freq(), get_overall_max_freq(), 10)):
                e1, e2, counter = -1, -1, 0
                while e1 < 0 or e2 < 0 and counter < 5:
                    with filter_warnings():
                        if is_classification:
                            measurement.begin()
                            start = time.perf_counter()
                            with SetFreq(freq, cores="all"):
                                model.predict_proba(X_test)
                            end = time.perf_counter()
                            measurement.end()
                        else:
                            measurement.begin()
                            start = time.perf_counter()
                            with SetFreq(freq, cores="all"):
                                model.predict(X_test)
                            end = time.perf_counter()
                            measurement.end()
                    e1, e2 = measurement.result.pkg[0], measurement.result.pkg[1]
                    counter += 1
                
                if e1 > 0 and e2 > 0:
                    idle_energy = 153.1569 * (end - start)
                    energy = (e1 + e2) / 1e6 - idle_energy
                    data.append({"model_type" : type(model), "config_id" : config_id, "n_threads": n_cores, "frequency" : int(freq), "inference_time": end - start, "energy_consumption": energy})
        else:
            raise ValueError("Unknown platform: {}".format(platform))
        n_counter += 1
        if n_counter >= N:
            break
            

def consistency_analysis(task_id, full_time_sec, timeout, memout, n_cores, seed):
    X_train, _, y_train, _, meta = load_openml_data_new(task_id)
    progress = ProgressSummary("consistency_gbmlp", amlb_tasks=True) 
    fac = GBMLPPipelineFactory(X_train, y_train, meta)
    cs = fac.get_config_space(seed)
    
    train_indices, test_indices = split_indices(len(X_train), train_size=0.8, stratify=y_train if meta["task_type"] == "classification" else None, seed=seed)
    X_test, y_test = X_train.iloc[test_indices], y_train.iloc[test_indices]
    X_train, y_train = X_train.iloc[train_indices], y_train.iloc[train_indices]
    
    cv_splits = create_cv_splits(len(X_train), 5, stratify=y_train if meta["task_type"] == "classification" else None, seed=seed) 
    
    limited_fit = limit(fit, wall_time=(timeout, "s"), memory=(memout, "GB"))
    limited_fit5 = limit(fit, wall_time=(timeout * 5, "s"), memory=(memout, "GB"))

    
    metric = get_default_metric(meta["task_type"] == "classification", meta["n_classes"] if meta["task_type"] == "classification" else None)
    metric.set_labels(meta["classes"] if meta["task_type"] == "classification" else None)
    
    data = pd.DataFrame(columns=["config_id", "inference_type", "index", "inference_time", "energy_consumption", metric.name, "pred_path", "y_true_path"])
    configs_df = pd.DataFrame(columns=["config_id"] + list(cs.keys()))
    
    affinity = list(os.sched_getaffinity(0))
    
    if not os.path.exists(f"./analysis/data/gbmlp_consistency/{task_id}"):
        os.makedirs(f"./analysis/data/gbmlp_consistency/{task_id}")
    
    # Save y_true
    np.save(f"./analysis/data/gbmlp_consistency/{task_id}/y_true.npy", y_test)
    for i in range(5):
        np.save(f"./analysis/data/gbmlp_consistency/{task_id}/y_true_{i}.npy", y_train.iloc[cv_splits[i]])
    
    n_errors = 0
    config_id = 0
    start_time = time.time()
    while time.time() - start_time < full_time_sec:
        data_arr = []
        
        config = cs.sample_configuration()
        print("Running model: ", config["model"])
        
        error = False
        
        try:
            # Run refit
            model = fac.load_pipeline(config, seed)
            
            limited_fit5(model, X_train, y_train, f"./analysis/data/gbmlp_consistency/{task_id}/{config_id}.pkl", timeout * 5, n_cores)
            with open(f"./analysis/data/gbmlp_consistency/{task_id}/{config_id}.pkl", "rb") as f:
                model = pickle.load(f)
                
            model.set_n_threads(n_cores)
            if hasattr(model, "set_timeout"):
                model.set_timeout(timeout * 5 * 0.7)
            
            with filter_warnings():
                # Dummy predict
                model.predict(X_test)
                
                for i in range(5):
                    if meta["task_type"] == "classification":
                        start_energy = _read_energies(affinity)
                        start = time.perf_counter()
                        predictions = model.predict_proba(X_test)
                        end = time.perf_counter()
                        end_energy = _read_energies(affinity)
                    elif meta["task_type"] == "regression":
                        start_energy = _read_energies(affinity)
                        start = time.perf_counter()
                        predictions = model.predict(X_test)
                        end = time.perf_counter()
                        end_energy = _read_energies(affinity)
                    metric_val = metric(y_test, predictions)
                    energy = np.sum(end_energy - start_energy)
                    inf_time = end - start
                    
                    np.save(f"./analysis/data/gbmlp_consistency/{task_id}/{config_id}_{i}.npy", predictions)
                    data_arr.append([config_id, "refit", i, inf_time, energy, metric_val, f"./analysis/data/gbmlp_consistency/{task_id}/{config_id}_{i}.npy", f"./analysis/data/gbmlp_consistency/{task_id}/y_true.npy"])

            # Run CV
            for i in range(5):
                train_indices_mask = np.ones(len(X_train), dtype=bool)
                train_indices_mask[cv_splits[i]] = False
                X_train_cv, y_train_cv = X_train.iloc[train_indices_mask], y_train.iloc[train_indices_mask]
                X_test_cv, y_test_cv = X_train.iloc[cv_splits[i]], y_train.iloc[cv_splits[i]]
                
                
                model = fac.load_pipeline(config, seed)
                
                limited_fit(model, X_train_cv, y_train_cv, f"./analysis/data/gbmlp_consistency/{task_id}/{config_id}_{i}.pkl", timeout, n_cores)
                with open(f"./analysis/data/gbmlp_consistency/{task_id}/{config_id}_{i}.pkl", "rb") as f:
                    model = pickle.load(f)
                    
                model.set_n_threads(n_cores)                
                
                with filter_warnings():
                    # Dummy predict
                    model.predict(X_test_cv)
                    
                    if meta["task_type"] == "classification":
                        start_energy = _read_energies(affinity)
                        start = time.perf_counter()
                        predictions = model.predict_proba(X_test_cv)
                        end = time.perf_counter()
                        end_energy = _read_energies(affinity)
                    else:
                        start_energy = _read_energies(affinity)
                        start = time.perf_counter()
                        predictions = model.predict(X_test_cv)
                        end = time.perf_counter()
                        end_energy = _read_energies(affinity)
                        
                    metric_val = metric(y_test_cv, predictions)
                    energy = np.sum(end_energy - start_energy)
                    inf_time = end - start
                    
                    np.save(f"./analysis/data/gbmlp_consistency/{task_id}/cv_{config_id}_{i}.npy", predictions)
                    data_arr.append([config_id, "cv", i, inf_time, energy, metric_val, f"./analysis/data/gbmlp_consistency/cv_{task_id}/{config_id}_{i}.npy", f"./analysis/data/gbmlp_consistency/{task_id}/y_true_{i}.npy"])
        except Exception as e:
            print(f"Error occurred while running limited fit: {e}")
            print(traceback.format_exc())
            error = True
            n_errors += 1
        
        if not error:
            configs_df.loc[len(configs_df)] = [config_id] + list(config.get_array())
            data_df = pd.DataFrame(data_arr, columns=["config_id", "type", "fold", "inf_time", "energy", "metric", "predictions", "y_true"])
            data = pd.concat([data, data_df])
        
        progress.set_progress(task_id, (time.time() - start_time) / full_time_sec, {"n_success": config_id + 1 - n_errors, "n_errors": n_errors}, True)
        configs_df.to_csv(f"./analysis/data/gbmlp_consistency/{task_id}/configs.csv", index=False)
        data.to_csv(f"./analysis/data/gbmlp_consistency/{task_id}/data.csv", index=False)
        config_id += 1

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_zb", default=0, type=int)
    parser.add_argument("--phase", default="train", type=str, choices=["train", "predict", "predict_fs", "consistency", "predict_fs_intel"])
    parser.add_argument("--N", default=100, type=int)
    parser.add_argument("--timeout", default=300, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--n_cores", default=1, type=int)
    parser.add_argument("--cons_time", default=8*3600, type=int)
    args = parser.parse_args()
    
    if args.seed < 0:
        seed = time.time_ns() % 2**32
    else:
        seed = args.seed
    
    np.random.seed(seed)
    
    if args.phase == "train":    
        task_id = zero_based_index_to_amlb_index(args.task_zb, n_cpu_cores=args.n_cores)
        X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)
        mem, n_cores = get_resources(task_id)
        
        assert len(os.sched_getaffinity(0)) == n_cores, "Expecting {} threads for training".format(n_cores)
        assert n_cores == args.n_cores, "Expecting  {} threads for training".format(n_cores)
        
        train_models(task_id, X_train, y_train, meta, args.timeout, mem, args.N, seed, args.n_cores)
    elif args.phase == "predict":
        task_id = zero_based_index_to_amlb_index(args.task_zb)
        X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)
        
        assert len(os.sched_getaffinity(0)) == args.n_cores, "Only {} threads allowed for prediction".format(args.n_cores)
        
        data = []
        predict_models(task_id, X_test, data, meta["task_type"] == "classification", args.n_cores, args.N)
        data = pd.DataFrame(data)
        
        data["config_id"] = data["config_id"].astype(int)
        data["n_threads"] = data["n_threads"].astype(int)
        
        data_path = f"./analysis/data/mt_gb_mlp/{task_id}.csv"
        
        if os.path.exists(data_path):
            data = pd.concat([pd.read_csv(data_path), data])
            data.sort_values(["config_id", "n_threads"], inplace=True)
            data = data.reset_index(drop=True)
        else:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        data.to_csv(data_path, index=False)
    elif args.phase == "predict_fs":    
        task_id = zero_based_index_to_amlb_index(args.task_zb)
        X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)
        
        assert len(os.sched_getaffinity(0)) == args.n_cores, "Only {} threads allowed for prediction".format(args.n_cores)
        
        data = []
        predict_models(task_id, X_test, data, meta["task_type"] == "classification", args.n_cores, args.N, freq_scaling=True)
        data = pd.DataFrame(data)
        
        data["config_id"] = data["config_id"].astype(int)
        data["n_threads"] = data["n_threads"].astype(int)
        data["frequency"] = data["frequency"].astype(int)
        
        data_path = f"./analysis/data/mt_gb_mlp_fs/{task_id}.csv"
        
        if os.path.exists(data_path):
            data = pd.concat([pd.read_csv(data_path), data])
            data.sort_values(["config_id", "n_threads", "frequency"], inplace=True)
            data = data.reset_index(drop=True)
        else:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        data.to_csv(data_path, index=False)
    elif args.phase == "predict_fs_intel":
        task_id = zero_based_index_to_amlb_index(args.task_zb)
        X_train, X_test, y_train, y_test, meta = load_openml_data_new(task_id)
        
        assert len(os.sched_getaffinity(0)) == args.n_cores, "Only {} threads allowed for prediction".format(args.n_cores)
        
        data = []
        predict_models(task_id, X_test, data, meta["task_type"] == "classification", args.n_cores, args.N, freq_scaling=True, platform="intel")
        data = pd.DataFrame(data)
        
        data["config_id"] = data["config_id"].astype(int)
        data["n_threads"] = data["n_threads"].astype(int)
        
        data_path = f"./analysis/data/mt_gb_mlp_fs_intel/{task_id}.csv"
        
        if os.path.exists(data_path):
            data = pd.concat([pd.read_csv(data_path), data])
            data.sort_values(["config_id", "n_threads", "frequency"], inplace=True)
            data = data.reset_index(drop=True)
        else:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        data.to_csv(data_path, index=False)
    elif args.phase == "consistency":
        task_id = zero_based_index_to_amlb_index(args.task_zb, n_cpu_cores=args.n_cores)
        mem, n_cores = get_resources(task_id)
        
        assert n_cores == args.n_cores, "Expecting {} threads for consistency analysis".format(n_cores)
        assert len(os.sched_getaffinity(0)) == n_cores, "Expecting {} threads for consistency analysis".format(n_cores)
        
        consistency_analysis(task_id, args.cons_time, args.timeout, mem, n_cores, seed)
