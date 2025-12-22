import os
import json
import pandas as pd
import argparse
import time
import numpy as np

from raaml.util.openml_data import load_amlb_tasks, amlb_index_to_zero_based_index


# Convert seconds to d-hh:mm:ss format
def format_time(seconds):
    if np.isnan(seconds):
        return "unknown"
    days = int(seconds // (24 * 3600))
    remaining = seconds % (24 * 3600)
    hours = int(remaining // 3600)
    remaining = remaining % 3600
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    return f"{days}d-{hours:02d}:{minutes:02d}:{seconds:02d}"

def sec_from_time_string(time_string):
    """Convert a time string in the format d-hh:mm:ss to seconds."""
    if time_string == "unknown":
        return 0
    split1 = time_string.split("d-")
    days = int(split1[0])
    split2 = split1[1].split(":")
    hours, minutes, seconds = map(int, split2)
    return days * 24 * 3600 + hours * 3600 + minutes * 60 + seconds

class ProgressSummary:
    def __init__(self, name: str, dir: str = "./progress/", amlb_tasks=False):
        self.name = name
        self.dir = dir
        self.subdir = os.path.join(dir, name)
        os.makedirs(self.subdir, exist_ok=True)
        if amlb_tasks:
            tasks = load_amlb_tasks()
            self.tasks = [t[1] for t in tasks]
        else:
            self.tasks = None
            
    
    def get_progress_history(self, task_id):
        if not os.path.exists(os.path.join(self.subdir, f"{task_id}_progress.json")):
            return []
        
        with open(os.path.join(self.subdir, f"{task_id}_progress.json"), "r") as f:
            print(f"Loading progress for task {task_id} from {os.path.join(self.subdir, f'{task_id}_progress.json')}")
            data = json.load(f)
            if "__hist__" not in data:
                return []
            return data["__hist__"]
    

    def set_progress(self, task_id, progress_value, meta_dict=None, print_progress=False):
        if meta_dict is None:
            meta_dict = {}
            
        hist = self.get_progress_history(task_id)
        hist.append((time.time(), progress_value))
        if len(hist) > 50:
            hist.pop(0)

        with open(os.path.join(self.subdir, f"{task_id}_progress.json"), "w") as f:
            json.dump({"progress": progress_value, "__hist__" : hist, **meta_dict}, f)
            
        if print_progress:
            print(f"Progress for task {task_id}: {progress_value * 100:.2f}%")


    def print_progress(self, print_sum=False):
        task_infos = []
        
        for json_file in os.listdir(self.subdir):
            if json_file.endswith("_progress.json"):
                with open(os.path.join(self.subdir, json_file), "r") as f:
                    progress_data = json.load(f)
                    # Collect data
                    task_id = "_".join(json_file.split('_')[:-1])
                    task_info = {'task_id': task_id}
                    task_info.update({k: v for k, v in progress_data.items() if k != "__hist__"})
                    
                    hist = progress_data["__hist__"]
                    if len(hist) >= 2:
                        hist = hist[-2:]
                        hist = [hist[0]] + [hist[i+1] for i in range(len(hist)-1) if (hist[i+1][1] - hist[i][1]) > 0]
                            
                        full_runtime_estimation = np.mean([(hist[i+1][0] - hist[i][0]) / (hist[i+1][1] - hist[i][1] + 1e-12) for i in range(len(hist) - 1)])
                        remaining_runtime_estimation = full_runtime_estimation * (1 - progress_data["progress"])
                    else:
                        full_runtime_estimation = np.nan
                        remaining_runtime_estimation = np.nan
                    
                    task_info.update({"est. runtime": full_runtime_estimation, "est. rem. runtime": remaining_runtime_estimation})
                    
                    task_infos.append(task_info)

        df = pd.DataFrame(task_infos)
        
        df["est. runtime"] = df['est. runtime'].apply(format_time)
        df["est. rem. runtime"] = df["est. rem. runtime"].apply(format_time)
        
        df = df.sort_values(by=["progress", "task_id"], ascending=False)
        
        # Print total # of tasks and missing tasks
        print(f"Total: {len(df)} entries")
        if self.tasks is not None:
            missing = set(self.tasks) - set(df['task_id'].astype(int).tolist())
            if len(missing) > 0:
                print("Missing tasks:", missing)
            df["task_id_zb"] = df["task_id"].astype(int).apply(amlb_index_to_zero_based_index)
            df = df[["task_id_zb", "task_id"] + [col for col in df.columns if col not in ["task_id_zb", "task_id"]]]

        
        with pd.option_context('display.max_rows', None):
            print(df)
            
        print(f"Total: {len(df)} entries")
            
        if print_sum:
            total_runtime = sum(df["est. runtime"].apply(sec_from_time_string))
            total_remaining = sum(df["est. rem. runtime"].apply(sec_from_time_string))
            print(f"\nTotal estimated runtime: {format_time(total_runtime)}")
            print(f"Total remaining runtime: {format_time(total_remaining)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print progress of training jobs")
    parser.add_argument("--progress_dir", type=str, default="./progress/", help="Path to the progress directory")
    parser.add_argument("--name", type=str, required=True, help="Name of the progress summary")
    parser.add_argument("--amlb", action="store_true", help="Whether the tasks are AMLB tasks")
    parser.add_argument("--sum", action="store_true", help="Whether to print the sum of estimated runtime")
    args = parser.parse_args()

    # Create a ProgressSummary instance
    progress = ProgressSummary(name=args.name, dir=args.progress_dir, amlb_tasks=args.amlb)
    progress.print_progress(print_sum=args.sum)