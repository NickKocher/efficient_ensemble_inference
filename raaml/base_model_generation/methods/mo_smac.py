import numpy as np

from mosmac3.smac.intensifier.mixins import intermediate_update, intermediate_decision
from mosmac3.smac.intensifier.multi_objective_intensifier import MOIntensifier
from mosmac3.smac import Callback


class NewIntensifier(
    intermediate_decision.NewCostDominatesOldCost,
    intermediate_update.ClosestIncumbentComparison,
    MOIntensifier,
):
    pass


class CustomCallback(Callback):
    
    def __init__(self, update_fun):
        self.update_fun = update_fun
    

    def on_tell_end(self, smbo, info, value) -> bool | None:
        status = value.status.name
        if status == "MEMORYOUT":
            status = "MEMOUT"
            
        objective_values = None
        if "results" in value.additional_info:
            results = value.additional_info["results"]
            objective_values = {obj_name: np.mean(results[obj_name]) for obj_name in results.keys()}
            
        self.update_fun("bmg", status, objective_values)
    