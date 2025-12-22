from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import time


@dataclass
class Task:
    config_id: int
    n_cores: int
    frequency: str | int
    inference_time: float
    energy: float


class ListScheduler(ABC):
    
    
    def __init__(self, n_cores_total, pruning_objective="energy", idle_power=None):
        self.pruning_objective = pruning_objective
        self.n_cores_total = n_cores_total
        
        if idle_power is None:
            raise NotImplementedError("idle_power must be specified")
        self.idle_power = idle_power
        
        
    def __eq__(self, other):
        
        if not isinstance(other, ListScheduler):
            return False
            
        attributes_match = (
            self.pruning_objective == other.pruning_objective and
            np.abs(self.idle_power - other.idle_power) < 1e-12 and 
            self.n_cores_total == other.n_cores_total 
        )
        
        types_match = type(self) == type(other)
        
        return attributes_match and types_match
        
        
    @abstractmethod
    def sort_tasks(self, task_list: list[Task]) -> list[int]:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def sort_tasks_by_order(self, task_list: list[Task], config_id_order: list[int]):
        return sorted(task_list, key=lambda t: config_id_order.index(t.config_id))
    
    
    def prune(self, task_list: list[Task]) -> list[Task]:
        grouped = defaultdict(list)
        for t in task_list:
            grouped[t.config_id].append(t)
        pruned = []
        for tasks in grouped.values():
            if self.pruning_objective == "energy":
                best = min(tasks, key=lambda t: t.energy)
            else:
                best = min(tasks, key=lambda t: t.inference_time)
            pruned.append(best)
        return pruned
    

    def plot_schedule(self, schedule: list[dict], total_n_cores: int):
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab20(np.linspace(0, 1, len(schedule)))

        # Build per-core timeline from the schedule
        core_intervals = [[] for _ in range(total_n_cores)]
        for entry in schedule:
            for c in entry["cores"]:
                core_intervals[c].append((entry["start"], entry["end"], entry["task_id"]))

        # Find the total makespan
        makespan = max(entry["end"] for entry in schedule)

        # Sort intervals for each core and compute gaps (including end gaps)
        per_core_gaps = [[] for _ in range(total_n_cores)]
        for core in range(total_n_cores):
            intervals = sorted(core_intervals[core], key=lambda x: x[0])
            last_end = 0.0
            for start, end, _ in intervals:
                if start > last_end:
                    per_core_gaps[core].append((last_end, start))
                last_end = end
            # Add final gap until makespan
            if last_end < makespan:
                per_core_gaps[core].append((last_end, makespan))

        # Plot task bars
        for i, entry in enumerate(schedule):
            start, end = entry["start"], entry["end"]
            for core in entry["cores"]:
                ax.barh(core, end - start, left=start,
                        color=colors[i], edgecolor="black", alpha=0.85)
                ax.text((start + end) / 2, core, f"T{entry['task_id']}",
                        ha="center", va="center", fontsize=8, color="white")

        # Plot idle (gap) bars and label them with duration
        for core, gaps in enumerate(per_core_gaps):
            for start, end in gaps:
                gap_len = end - start
                if gap_len > 0:
                    ax.barh(core, gap_len, left=start,
                            color="lightgray", edgecolor="none", alpha=0.6)
                    ax.text((start + end) / 2, core, f"{gap_len:.1f}",
                            ha="center", va="center", fontsize=7, color="black", fontstyle="italic")

        # Label axes and appearance
        ax.set_xlabel("Time")
        ax.set_ylabel("Core ID")
        ax.set_yticks(range(total_n_cores))
        ax.set_yticklabels([f"Core {i}" for i in range(total_n_cores)])
        ax.set_title("Task Scheduling Gantt Chart (Gray = Idle Time, Label = Gap Duration)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()


    def _get_first_fit(self, task_list: list[Task], n_cores_free: int) -> Task:
        for task in task_list:
            if task.n_cores <= n_cores_free:
                return task
        return None


    def compute_offline_resources(
        self,
        task_list: list[Task],
        return_schedule : bool = False
    ):
        task_list = list(task_list)
        task_list = self.prune(task_list)
        task_list = self.sort_tasks(task_list)
        
        return self.compute_online_resources(
            task_list,
            [t.config_id for t in task_list],
            return_schedule=return_schedule,
            prune=False,
        )

        
    def compute_online_resources(
        self,
        task_list: list[Task],
        config_id_order: list[int],
        return_schedule : bool = False,
        prune : bool = True
    ):
        task_list = list(task_list)
        if prune:
            task_list = self.prune(task_list)
        task_list = self.sort_tasks_by_order(task_list, config_id_order)
        task_list_full = list(task_list)
                
        next_free_time = np.zeros(self.n_cores_total)
        cur_time = 0.0
        gap_time = 0.0

        if return_schedule:
            schedule = []

        while len(task_list) > 0:
            cores_free = np.where(next_free_time <= cur_time)[0]
            task = self._get_first_fit(task_list, len(cores_free))

            # No task fits now — advance time to the next core release
            if task is None:
                new_cur_time = np.min([t for t in next_free_time if t > cur_time])
                idle_duration = new_cur_time - cur_time
                gap_time += idle_duration * len(cores_free)  # idle cores contribute to total gap
                cur_time = new_cur_time
                continue

            # Assign the task to the first available cores
            task_list.remove(task)
            assigned = cores_free[:task.n_cores]
            start_time = cur_time
            end_time = cur_time + task.inference_time

            # Update cores’ availability
            next_free_time[assigned] = end_time

            if return_schedule:
                # Record for plotting
                schedule.append({
                    "task_id": task.config_id,
                    "cores": assigned,
                    "start": start_time,
                    "end": end_time
                })

            # Advance time to next event
            cur_time = max(np.min(next_free_time), cur_time)

        # All tasks done
        final_time = np.max(next_free_time)

        # Add remaining idle time after the last task for each core
        end_gaps = final_time - next_free_time
        end_gap_total = np.sum(end_gaps)
        gap_time += end_gap_total
        avg_gap = gap_time / self.n_cores_total
        
        energy = gap_time * self.idle_power + np.sum(task.energy for task in task_list_full)
        
        order = [t.config_id for t in task_list_full]

        if return_schedule:
            return final_time, energy, avg_gap, order, schedule
                
        return final_time, energy, avg_gap, order


class LongestProcessingTimeScheduler(ListScheduler):
    def sort_tasks(self, task_list: list[Task]) -> list[Task]:
        return sorted(task_list, key=lambda t: t.inference_time, reverse=True)


class HighestWorkloadScheduler(ListScheduler):
    def sort_tasks(self, task_list: list[Task]) -> list[Task]:
        return sorted(task_list, key=lambda t: t.inference_time * t.n_cores, reverse=True)


class HighestThreadCountScheduler(ListScheduler):
    def sort_tasks(self, task_list: list[Task]) -> list[Task]:
        return sorted(task_list, key=lambda t: t.n_cores, reverse=True)
    
    
class RandomScheduler(ListScheduler):
    def sort_tasks(self, task_list: list[Task]) -> list[Task]:
        return list(np.random.permutation(task_list))