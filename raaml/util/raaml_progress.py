from enum import auto
from json import load
import os
import sqlite3
import time
import numpy as np
from tabulate import tabulate
from pynisher import MemoryLimitException, TimeoutException
from raaml.util.openml_data import get_resources, load_amlb_tasks, amlb_index_to_zero_based_index


def exception_to_status(exception):
    if isinstance(exception, TimeoutException):
        return "TIMEOUT"
    elif isinstance(exception, MemoryLimitException):
        return "MEMOUT"
    else:
        return "CRASHED"


class RAAMLProgressSummary:
    VALID_STATUSES = {"SUCCESS", "CRASHED", "TIMEOUT", "MEMOUT"}

    def __init__(self, identifier, name, progress_db_dir="./progress/"):
        self.identifier = identifier
        self.base_dir = os.path.join(progress_db_dir, name)
        self.db_path = os.path.join(self.base_dir, f"{identifier}.db")

        os.makedirs(self.base_dir, exist_ok=True)
        if not os.path.exists(self.db_path):
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                step TEXT PRIMARY KEY,
                timestamp REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS status_counts (
                step TEXT,
                status TEXT,
                count INTEGER,
                PRIMARY KEY(step, status)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS objectives (
                step TEXT,
                objective_name TEXT,
                running_avg REAL,
                count INTEGER,
                min_val REAL,
                max_val REAL,
                PRIMARY KEY(step, objective_name)
            )
        """)
        conn.commit()
        conn.close()

    def update_progress(self, step, status, objective_values=None):
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of {self.VALID_STATUSES}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update timestamp
        cursor.execute("""
            INSERT INTO progress(step, timestamp)
            VALUES (?, ?)
            ON CONFLICT(step) DO UPDATE SET timestamp=excluded.timestamp
        """, (step, time.time()))

        # Update status counts
        cursor.execute("""
            INSERT INTO status_counts(step, status, count)
            VALUES (?, ?, 1)
            ON CONFLICT(step, status) DO UPDATE SET count = count + 1
        """, (step, status))

        # Update objectives
        if objective_values:
            for obj_name, val in objective_values.items():
                if not np.isfinite(val):
                    continue
                cursor.execute("""
                    SELECT running_avg, count, min_val, max_val
                    FROM objectives
                    WHERE step=? AND objective_name=?
                """, (step, obj_name))
                row = cursor.fetchone()
                if row:
                    running_avg, count, min_val, max_val = row
                    new_count = count + 1
                    new_avg = (running_avg * count + val) / new_count
                    new_min = min(min_val, val)
                    new_max = max(max_val, val)
                    cursor.execute("""
                        UPDATE objectives
                        SET running_avg=?, count=?, min_val=?, max_val=?
                        WHERE step=? AND objective_name=?
                    """, (new_avg, new_count, new_min, new_max, step, obj_name))
                else:
                    cursor.execute("""
                        INSERT INTO objectives(step, objective_name, running_avg, count, min_val, max_val)
                        VALUES (?, ?, ?, 1, ?, ?)
                    """, (step, obj_name, val, val, val))

        conn.commit()
        conn.close()
        
    def reset(self):
        """Delete all progress data for this identifier."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            self._init_db()
            
            
            
    @staticmethod
    def print_all_results(progress_db_dir, name, step_filter=None, seed=None, n_cpus=None):
        from itertools import product
        from collections import defaultdict

        base_dir = os.path.join(progress_db_dir, name)
        if not os.path.exists(base_dir):
            print(f"No directory found at {base_dir}")
            return

        db_files = [f for f in os.listdir(base_dir) if f.endswith(".db")]
        if not db_files:
            print("No databases found.")
            return

        # --- containers for final table and discovered objective names ---
        table = []
        obj_names_set = set()

        # --- per-step issue maps (step -> list of problematic ids + info) ---
        step_low_success_map = defaultdict(list)   # success < 5 (including 0 or missing)
        step_high_crash_map = defaultdict(list)    # success / total < 0.5
        step_set = set()

        # all identifiers (DB files that exist)
        identifiers = []
        identifiers_with_progress = []

        # --- build expected identifiers and index lookup for bmg setup ---
        expected_identifiers = []
        methods = ["random", "so-smac", "mo-smac", "so-asha", "mo-asha", "nsga2"]
        task_ids = [t[1] for t in load_amlb_tasks(n_cpu_cores=n_cpus)]
        
        for m in methods:
            for t in task_ids:
                expected_identifiers.append(f"{t}_{m}_{seed}")



        # --- first pass: read every DB and collect per-identifier state (status, objectives, progress) ---
        per_id_data = {}

        for db_file in db_files:
            identifier = db_file.replace(".db", "")
            identifiers.append(identifier)

            conn = sqlite3.connect(os.path.join(base_dir, db_file))
            cursor = conn.cursor()

            # progress rows (step, timestamp)
            cursor.execute("SELECT step, timestamp FROM progress")
            progress_rows = cursor.fetchall()
            if step_filter is not None:
                progress_rows = [row for row in progress_rows if row[0] == step_filter]
                
            if len(progress_rows) > 0:
                identifiers_with_progress.append(identifier)

            # status counts (step, status, count)
            cursor.execute("SELECT step, status, count FROM status_counts")
            raw_status_rows = cursor.fetchall()
            status_dict = {}
            for step, status, count in raw_status_rows:
                if step_filter is not None and step != step_filter:
                    continue
                status_dict.setdefault(step, {s: 0 for s in RAAMLProgressSummary.VALID_STATUSES})
                status_dict[step][status] = count

            # objectives (step, objective_name, running_avg, min_val, max_val)
            cursor.execute("SELECT step, objective_name, running_avg, min_val, max_val FROM objectives")
            raw_obj_rows = cursor.fetchall()
            obj_dict = {}
            for step, obj_name, avg, min_val, max_val in raw_obj_rows:
                if step_filter is not None and step != step_filter:
                    continue
                obj_dict.setdefault(step, {})[obj_name] = (avg, min_val, max_val)
                obj_names_set.add(obj_name)
                step_set.add(step)

            per_id_data[identifier] = {
                "progress_rows": progress_rows,
                "status_dict": status_dict,
                "obj_dict": obj_dict,
            }

            conn.close()

        # --- also include steps from expected identifiers even if DB is missing ---
        missing = set(expected_identifiers) - set(identifiers_with_progress)
        if expected_identifiers:
            for eid in expected_identifiers:
                if eid not in per_id_data:
                    per_id_data[eid] = {
                        "progress_rows": [],
                        "status_dict": {},
                        "obj_dict": {},
                    }

        # --- build main table and detect issues ---
        sorted_obj_names = sorted(obj_names_set)
        
        large_deviations = {}

        for identifier in sorted(per_id_data.keys()):
            data = per_id_data[identifier]
            status_dict = data["status_dict"]
            obj_dict = data["obj_dict"]

            # steps seen for this identifier
            seen_steps = set(status_dict.keys()) | {row[0] for row in data["progress_rows"]}

            for step in seen_steps:
                step_set.add(step)
                counts = status_dict.get(step, {s: 0 for s in RAAMLProgressSummary.VALID_STATUSES})
                total = sum(counts.values())
                success = counts.get("SUCCESS", 0)

                if success < 5:
                    step_low_success_map[step].append(
                        (identifier, success, total)
                    )
                if total > 0:
                    success_rate = success / total
                    if success_rate < 0.4:
                        step_high_crash_map[step].append(
                            (identifier, success, total)
                        )

            # --- add table rows for visualization (only for real DBs) ---
            if identifier in identifiers:
                task_id = int(identifier.split("_")[0])
                if task_id not in task_ids:
                    continue
                
                for step, ts in data["progress_rows"]:
                    total_status = sum(status_dict.get(step, {}).values()) or 1
                    id_display = f"{identifier}"
                    row = [id_display, step, time.strftime("%d/%m %H:%M:%S", time.localtime(ts))]

                    for s in ["SUCCESS", "CRASHED", "TIMEOUT", "MEMOUT"]:
                        abs_count = status_dict.get(step, {}).get(s, 0)
                        perc = (abs_count / total_status) * 100
                        row.append(f"{abs_count} ({perc:.1f}%)")
                        
                    autogluon_result, n_classes = EXPECTED_RESULTS_RANGE[task_id]
                    row.append(f"{autogluon_result:.2e}")

                    # objectives: print min-max in scientific notation, no avg
                    for obj_name in sorted_obj_names:
                        if obj_name in obj_dict.get(step, {}):
                            _, min_val, max_val = obj_dict[step][obj_name]
                            try:
                                row.append(f"{min_val:.2e}")
                            except Exception:
                                row.append(f"{min_val}")
                                
                            if step == "bmg" and obj_name == objective_from_n_classes(n_classes):
                                deviation = abs(min_val - autogluon_result) / (autogluon_result + 1e-12)
                                if deviation > 0.25 and not abs(min_val - autogluon_result) < 1e-2:
                                    large_deviations[identifier] = (min_val, autogluon_result, n_classes)
                        else:
                            row.append("")

                    table.append(row)

        # --- print main results table ---
        table.sort(key=lambda x: (x[0].split("]")[0], x[1]))
        headers = ["id (index)", "Step", "Last Update", "SUCCESS", "CRASHED", "TIMEOUT", "MEMOUT", "AG Result"] + sorted_obj_names
        print(tabulate(table, headers=headers, tablefmt="github", disable_numparse=True))
        
        print("Total: ", len(table), "entries")

        # --- print step-wise summary of issues ---
        print("\n### Step-wise Summary of Issues")

        steps_to_report = sorted(step_set)

        def _sort_key(entry):
            _, idx, *_ = entry
            return (idx if isinstance(idx, int) else float("inf"), str(entry[0]))

        for step in steps_to_report:
            low_list = step_low_success_map.get(step, [])
            crash_list = step_high_crash_map.get(step, [])

            if not low_list and not crash_list and len(missing) == 0 and len(large_deviations) == 0:
                continue

            print(f"\nStep: {step}")

            if low_list:
                print("  Few successes (<5 SUCCESS):")
                if len(low_list) > 200:
                    print("     (too many to list)")
                else:
                    for identifier, success, total in sorted(low_list, key=_sort_key):
                        print(f"     {identifier} — SUCCESS={success}, total={total}")

            if crash_list:
                print("  High crash rate (<40% success):")
                for identifier, success, total in sorted(crash_list, key=_sort_key):
                    pct = (success / total * 100) if total > 0 else 0.0
                    print(f"     {identifier} — SUCCESS={success}/{total} ({pct:.1f}%)")
            
            if len(missing) > 0:
                print(f"  Missing ({len(missing)}):")
                if len(missing) > 100:
                    print("     (too many to list)")
                else:
                    for identifier in sorted(missing):
                        index = identifier_to_index(identifier)
                        print(f"{identifier} ({index})", end=", ")
                    
            if len(large_deviations) > 0:
                print()
                print(f"  Tasks with large deviations from expected results ({len(large_deviations)}):")
                tasks = {}
                for identifier, (val, ag_val, n_classes) in sorted(large_deviations.items(), key=lambda x: x[1][0]):
                    index = identifier_to_index(identifier)
                    #print(f"       {identifier} ({index}): {val:.3e} vs {ag_val:.3e}")
                    tasks[int(identifier.split("_")[0])] = (val, ag_val)
                for task, (val, ag_val) in tasks.items():
                    print(f"  Task {task}: {val:.3e} vs {ag_val:.3e}")

def identifier_to_index(identifier):
    task_id, method, seed = identifier.split("_")
    method_index = ["random", "so-smac", "mo-smac", "so-asha", "mo-asha", "nsga2"].index(method)
    
    _, n_cpus = get_resources(int(task_id))
    tasks = load_amlb_tasks(n_cpu_cores=n_cpus)
    tasks = [t[1] for t in tasks]
    
    cpu_res_name = {1: "low", 2: "medium", 4: "high"}[n_cpus]
    
    task_id_index = tasks.index(int(task_id))
    
    return f"{task_id_index * 6 + method_index} ({cpu_res_name})"



EXPECTED_RESULTS_RANGE = {
    146818: (0.05900000000000005, 2),
    146820: (0.0050000000000000044, 2),
    167120: (0.469, 2),
    168350: (0.03200000000000003, 2),
    168757: (0.20399999999999996, 2),
    168868: (0.007000000000000006, 2),
    168911: (0.11399999999999999, 2),
    189354: (0.268, 2),
    189356: (0.21799999999999997, 2),
    189922: (0.009000000000000008, 2),
    190137: (0.06699999999999995, 2),
    190392: (0.05500000000000005, 2),
    190410: (0.122, 2),
    190411: (0.07899999999999996, 2),
    190412: (0.127, 2),
    359955: (0.242, 2),
    359956: (0.05800000000000005, 2),
    359958: (0.04800000000000004, 2),
    359962: (0.16000000000000003, 2),
    359965: (0.0, 2),
    359966: (0.015000000000000013, 2),
    359967: (0.11399999999999999, 2),
    359968: (0.07799999999999996, 2),
    359971: (0.0020000000000000018, 2),
    359972: (0.010000000000000009, 2),
    359973: (0.17400000000000004, 2),
    359975: (0.0040000000000000036, 2),
    359979: (0.09799999999999998, 2),
    359980: (0.0030000000000000027, 2),
    359982: (0.05900000000000005, 2),
    359983: (0.06799999999999995, 2),
    359988: (0.08599999999999997, 2),
    359989: (0.0, 2),
    359990: (0.01100000000000001, 2),
    359991: (0.20899999999999996, 2),
    359992: (0.29000000000000004, 2),
    359994: (0.30300000000000005, 2),
    360113: (0.357, 2),
    360114: (0.16200000000000003, 2),
    360975: (0.09199999999999997, 2),
    3945: (0.15100000000000002, 2),
    10090: (0.695, 50),
    168784: (0.464, 7),
    168909: (0.014, 5),
    168910: (0.683, 7),
    189355: (0.266, 355),
    190146: (0.312, 4),
    2073: (1.003, 10),
    211979: (0.65, 4),
    211986: (0.831, 3),
    359953: (0.211, 20),
    359954: (0.654, 5),
    359957: (0.126, 9),
    359959: (0.927, 3),
    359960: (0.002, 4),
    359961: (0.071, 10),
    359963: (0.052, 7),
    359964: (0.106, 3),
    359969: (1.039, 6),
    359970: (0.668, 5),
    359974: (0.698, 7),
    359976: (0.221, 10),
    359977: (0.295, 3),
    359981: (0.012, 3),
    359984: (2.467, 100),
    359985: (0.672, 10),
    359986: (1.304, 10),
    359987: (0.0, 7),
    359993: (0.559, 3),
    360112: (0.001, 23),
    7593: (0.057, 7),
    167210: (21.0, 0),
    233211: (500.0, 0),
    233212: (1900.0, 0),
    233213: (150.0, 0),
    233214: (6800000.0, 0),
    233215: (8.6, 0),
    317614: (8.4, 0),
    359929: (29.0, 0),
    359930: (0.19, 0),
    359931: (0.67, 0),
    359932: (12.0, 0),
    359933: (0.095, 0),
    359934: (0.84, 0),
    359935: (0.57, 0),
    359936: (0.0018, 0),
    359937: (3400.0, 0),
    359938: (23000.0, 0),
    359939: (0.028, 0),
    359940: (0.028, 0),
    359941: (3300000.0, 0),
    359942: (0.14, 0),
    359943: (1.5, 0),
    359944: (2.1, 0),
    359945: (0.13, 0),
    359946: (2.6, 0),
    359948: (890.0, 0),
    359949: (110000.0, 0),
    359950: (2.9, 0),
    359951: (25000.0, 0),
    359952: (28000.0, 0),
    360932: (0.71, 0),
    360933: (0.69, 0),
    360945: (21000.0, 0),
}

def objective_from_n_classes(n_classes):
    if n_classes == 0:
        return "Log Loss"
    elif n_classes == 2:
        return "1 - ROC AUC"
    else:
        return "RMSE"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print progress of jobs")
    parser.add_argument("--progress_dir", type=str, default="./progress/", help="Path to the progress directory")
    parser.add_argument("--name", type=str, required=True, help="Name of the progress summary")
    parser.add_argument("--step_filter", type=str, default=None, help="Filter by step")
    parser.add_argument("--seed", type=int, default=1, help="seed to check for")
    parser.add_argument("--n_cpus", type=int, default=None, help="Number of CPUs that is assigned to the corresponding task")
    args = parser.parse_args()

    RAAMLProgressSummary.print_all_results(args.progress_dir, args.name, args.step_filter, args.seed, args.n_cpus)
