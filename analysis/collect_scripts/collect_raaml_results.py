import pandas as pd
import os
import tqdm
import numpy as np

from raaml.util.openml_data import load_amlb_tasks

def collect_raaml_results(file_path: str) -> pd.DataFrame:
    data = []
    
    dirs = os.listdir(file_path)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(file_path, d)) and d.isdigit()]
    
    tasks = [t[1] for t in load_amlb_tasks()]
    
    for task_id in tqdm.tqdm(tasks):
        if not os.path.exists(os.path.join(file_path, str(task_id))):
            raise FileNotFoundError(f"Directory {os.path.join(file_path, str(task_id))} not found.")
        
        for identifier in os.listdir(os.path.join(file_path, str(task_id))):
            method = identifier.split("_")[0]
            seed = int(identifier.split("_")[-1])
            
            bmp_path = os.path.join(file_path, str(task_id), identifier, "base_model_pool", "results.csv")
            ensembling_path = os.path.join(file_path, str(task_id), identifier, "ensembling_predictions_cv", "ensembling_predictions.csv")
            final_path = os.path.join(file_path, str(task_id), identifier, "final_predictions", "final_results.csv")
            bmp_configs_path = os.path.join(file_path, str(task_id), identifier, "base_model_pool", "configs.csv")
            
            bmp_results = pd.read_csv(bmp_path) if os.path.exists(bmp_path) else pd.DataFrame(columns=["config_id", "time", "status"])
            bmp_configs = pd.read_csv(bmp_configs_path) if os.path.exists(bmp_configs_path) else pd.DataFrame(columns=["config_id", "realmlp:act", "tabm:d_block"])
            ensembling_results = pd.read_csv(ensembling_path) if os.path.exists(ensembling_path) else pd.DataFrame(columns=["config_id", "frequency", "n_threads"])
            final_results = pd.read_csv(final_path) if os.path.exists(final_path) else pd.DataFrame(columns=["config_id", "frequency", "n_cores"])
            
            bmp_times = np.unique(bmp_results["time"])
            
            for config_id in bmp_results['config_id'].unique():
                bmp_config = bmp_results[bmp_results['config_id'] == config_id]
                ensembling_config = ensembling_results[ensembling_results['config_id'] == config_id]
                final_config = final_results[final_results['config_id'] == config_id]
                
                bmp_configuration = bmp_configs[bmp_configs['config_id'] == config_id]
                if not np.isnan(bmp_configuration['realmlp:act'].values[0]):
                    model = "RealMLP"
                elif not np.isnan(bmp_configuration['tabm:d_block'].values[0]):
                    model = "TabM"
                else:
                    model = "XGBoost"
                
                cols = ["RMSE", "1 - ROC AUC", "Log Loss", "inference_time", "energy_consumption"]
                cols = [col for col in cols if col in bmp_config.columns]

                bmp_results_dict = {f"bmp_{col}": bmp_config[col].mean() for col in cols}
                bmp_results_dict["status"] = bmp_config["status"].values[0]
                time_config = bmp_config["time"].values[0]
                times_before = bmp_times[bmp_times < time_config]
                time_last = times_before.max() if len(times_before) > 0 else 0
                
                ens_grouped = ensembling_config.groupby(["frequency", "n_threads"])
                final_grouped = final_config.groupby(["frequency", "n_cores"])

                pred_metric = cols[0]

                if len(ensembling_config) > 0:
                    bmp_results_dict[f"ensembling_{pred_metric}"] = ensembling_config[pred_metric].values[0]
                    for (freq, n_cores), ensembling_subconfig in ens_grouped:
                        for col in ["inference_time", "energy_consumption"]:
                            bmp_results_dict[f"ensembling_{col}_freq_{freq}_cores_{n_cores}"] = ensembling_subconfig[col].mean()

                if len(final_config) > 0:
                    bmp_results_dict[f"final_{pred_metric}"] = final_config[pred_metric].values[0]
                    for (freq, n_cores), final_subconfig in final_grouped:
                        for col in ["inference_time", "energy_consumption"]:
                            bmp_results_dict[f"final_{col}_freq_{freq}_cores_{n_cores}"] = final_subconfig[col].mean()

                row = {
                    "task_id": task_id,
                    "method": method,
                    "seed": seed,
                    "config_id": config_id,
                    "model": model,
                    "time_5fcv": time_config - time_last,
                    **bmp_results_dict,
                }
                
                data.append(row)
                
                
                """for type, df in zip(["bmp", "ensembling", "final"], [bmp_config, ensembling_config, final_config]):
                    if type == "bmp":
                        row = {
                            "task_id": task_id,
                            "method": method,
                            "seed": seed,
                            "config_id": config_id,
                            "type": type,
                            "model": model,
                            "n_threads": None,
                            "frequency": None,
                            "status": df["status"].values[0]
                        }
                        for col in cols:
                            row[col] = df[col].mean()
                            row[f"{col}_variation_coefficient"] = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                        data.append(row)
                    else:
                        n_threads_col = "n_threads" if "n_threads" in df.columns else "n_cores"
                        for n_threads in df[n_threads_col].unique():
                            for freq in df['frequency'].unique():
                                df_n_freq = df[(df[n_threads_col] == n_threads) & (df['frequency'] == freq)]
                                if not df_n_freq.empty:
                                    row = {
                                        "task_id": task_id,
                                        "method": method,
                                        "seed": seed,
                                        "config_id": config_id,
                                        "type": type,
                                        "model": model,
                                        "n_threads": n_threads,
                                        "frequency": freq
                                    }
                                    for col in cols:
                                        row[col] = df_n_freq[col].mean()
                                        row[f"{col}_variation_coefficient"] = df_n_freq[col].std() / df_n_freq[col].mean() if df_n_freq[col].mean() != 0 else 0
                                    data.append(row)"""
                
    return pd.DataFrame(data)


if __name__ == "__main__":
    results_df = collect_raaml_results("./raaml_output_gbmlp")
    results_df.to_csv("./test/collect/raaml_results_gbmlp.csv", index=False)