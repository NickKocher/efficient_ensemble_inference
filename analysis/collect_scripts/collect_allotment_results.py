import pandas as pd
import os
import tqdm
import yaml
import argparse

from raaml.util.openml_data import load_amlb_tasks

tasks = [t[1] for t in load_amlb_tasks()]


##############################################
### BASE RESULTS #############################
##############################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="parallel", help="Type of experiment: base or extended")
    args = parser.parse_args()

    base_folder = f"./ensemble_pool_results/{args.exp_type}/"

    alloted_pools = []
    alloted_meta = []

    pool_index = 0

    for task in tqdm.tqdm(tasks):
        for seed in range(1,6):
            for method in ["random", "so-smac", "mo-smac", "nsga2", "so-asha", "mo-asha"]:
                dir = os.path.join(base_folder, str(task), str(seed), method)
                if os.path.exists(os.path.join(dir, "meta_results.yaml")):
                    subfiles = os.listdir(dir)
                    
                    with open(os.path.join(dir, "meta_results.yaml"), "r") as f:
                        meta_results = yaml.load(f, Loader=yaml.FullLoader)
                    
                    for f in meta_results.keys():
                        if not os.path.exists(os.path.join(dir, f"{f}.csv")):
                            continue
                        
                        if "bmp" in f or "nds" in f:
                            df = pd.read_csv(os.path.join(dir, f"{f}.csv"))
                            
                            df["task_id"] = task
                            df["seed"] = seed
                            df["bmg_method"] = method
                            df["pool_index"] = pool_index
                            
                            if args.exp_type == "parallel" or args.exp_type == "frequency_scaling+parallel":
                                if "bmp" in f:
                                    ensembling_method = "base_model_pool"
                                else:
                                    ensembling_method = f.split("|")[0].split("_")[-1]
                                rest = f.split(".")[0].split("_")[-3:]
                                allotment_method, scheduler, n_cores = rest
                                n_cores = int(n_cores)
                                df["n_cores"] = n_cores
                                df["scheduler"] = scheduler
                                df["allotment_method"] = allotment_method
                                df["ensembling_method"] = ensembling_method
                            elif args.exp_type == "frequency_scaling":
                                if "bmp" in f:
                                    ensembling_method = "base_model_pool"
                                else:
                                    ensembling_method = f.split("|")[0].split("_")[-1]
                                allotment_method = f.split("_")[-1].split(".")[0]
                                df["ensembling_method"] = ensembling_method
                                df["allotment_method"] = allotment_method
                                
                            meta = meta_results[f]
                            meta["pool_index"] = pool_index
                            
                            pool_index += 1
                            
                            alloted_pools.append(df)
                            alloted_meta.append(meta)
                else:
                    print(f"Missing {dir}")
    
    alloted_pools_df = pd.concat(alloted_pools)
    alloted_meta_df = pd.DataFrame(alloted_meta)

    alloted_pools_df.to_csv(f"./test/collect/{args.exp_type}_alloted_eps.csv", index=False)
    alloted_meta_df.to_csv(f"./test/collect/{args.exp_type}_alloted_meta.csv", index=False)