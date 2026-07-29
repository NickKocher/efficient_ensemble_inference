import pandas as pd
import os
import tqdm
import yaml
import argparse
import json

from raaml.util.openml_data import load_amlb_tasks

tasks = [t[1] for t in load_amlb_tasks()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="base", help="Type of experiment: base or extended")
    parser.add_argument("--time", type=int, default=3600, help="Time budget for the experiment")
    args = parser.parse_args()

    base_folder = f"./ensemble_pool_results/{args.exp_type}/"

    base_model_pool_results = []
    ensemble_pool_results = []
    base_model_pools_meta = []
    ensemble_pools_meta = []

    pool_index = 0

    for task in tqdm.tqdm(tasks):
        for seed in range(1,6):
            for method in ["so-smac"]:
                dir = os.path.join(base_folder, str(task), str(seed), method)
                if os.path.exists(os.path.join(dir, "meta_results.yaml")):
                    subfiles = os.listdir(dir)
                    
                    with open(os.path.join(dir, "meta_results.yaml"), "r") as f:
                        meta_results = yaml.safe_load(f)

                    n_models_indicator = f"base_28_ges_nd_50_silo_time_{args.time}" if args.exp_type == "base_time" else "base_28_ges_nd_50_silo"
                    more_than_50_models = n_models_indicator in meta_results
                    
                    for f in meta_results.keys():
                        if not os.path.exists(os.path.join(dir, f"{f}.csv")):
                            continue
                        
                        df = pd.read_csv(os.path.join(dir, f"{f}.csv"), engine='pyarrow')
                        try:
                            df_ens = pd.read_csv(os.path.join(dir, f"{f}_results_ens.csv"), engine='pyarrow')
                        except:
                            df_ens = pd.DataFrame()

                        assert len(df) == len(df_ens), "Length mismatch between pool results and ensemble results"

                        try:
                            df_inf = pd.read_csv(os.path.join(dir, f"{f}_inference_times.csv"), engine='pyarrow')
                        except:
                            df_inf = pd.DataFrame()

                        if not df_inf.empty:
                            assert len(df) == len(df_inf), "Length mismatch between pool results and inference times"

                        df["task_id"] = task
                        df["seed"] = seed
                        df["bmg_method"] = method
                        df["pool_index"] = pool_index

                        df[df_ens.columns] = df_ens

                        if not df_inf.empty:
                            # stored as JSON strings by extract_ensemble_inference_times/
                            # collect_ensemble_inference_times.py so they round-trip through CSV
                            df["inference_times"] = df_inf["inference_times"].apply(json.loads)
                        
                        if "single" in f:
                            freq = "boost" if "boost" in f else "no-boost"
                            time = f.split("_")[-1]
                            
                            df["frequency"] = freq
                            df["time"] = time
                            base_model_pool_results.append(df)
                            
                            meta = meta_results[f]
                            meta["pool_index"] = pool_index
                            
                            base_model_pools_meta.append(meta)
                        else:
                            freq = "boost" if "boost" in f else "no-boost"
                            ensembling_method = f.split("_")[2]
                            filter = "_".join(f.split("_")[3:])
                            
                            if filter == "top_50" and not more_than_50_models:
                                filter = "none"
                            
                            df["frequency"] = freq
                            df["es_method"] = ensembling_method
                            df["filter"] = filter
                            ensemble_pool_results.append(df)           
                            
                            meta = meta_results[f]
                            meta["pool_index"] = pool_index
                            
                            ensemble_pools_meta.append(meta)
                            
                            
                        pool_index += 1
                else:
                    print(f"Missing {dir}")
                    

    ensemble_pool_results = pd.concat(ensemble_pool_results)
    ensemble_pool_meta = pd.DataFrame(ensemble_pools_meta)

    if args.exp_type == "base":
        base_model_pool_results = pd.concat(base_model_pool_results)
        base_model_pool_meta = pd.DataFrame(base_model_pools_meta)
        base_model_pool_results.to_csv("./test/collect/base_model_pools.csv", index=False)
        ensemble_pool_results.to_csv("./test/collect/base_ensemble_pools.csv", index=False)

        base_model_pool_meta.to_csv("./test/collect/base_model_pools_meta.csv", index=False)
        ensemble_pool_meta.to_csv("./test/collect/base_ensemble_pools_meta.csv", index=False)
    elif args.exp_type == "base_time":
        ensemble_pool_results.to_csv(f"./test/collect/base_time_{args.time}_ensemble_pools.csv", index=False)
        ensemble_pool_meta.to_csv(f"./test/collect/base_time_{args.time}_ensemble_pools_meta.csv", index=False)