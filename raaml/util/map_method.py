import numpy as np

def map_bmg_method_to_unified_name(method: str) -> str:
    """
    Maps a method name to its unified name.

    Args:
        method (str): The method name to map.

    Returns:
        str: The unified name of the method.
    """
        
    method = method.upper().replace("_", "-").replace(" ", "-")
    
    method_mapping = {
        "BO": "so-bo",
        "SO-BO": "so-bo",
        "SOBO": "so-bo",
        "SO-SMAC": "so-bo",
        "SOSMAC": "so-bo",
        "LOAD": "load",
        "LOAD-BASE-MODEL-POOL": "load",
        "LOAD-BMP": "load",
        "LOAD-FROM-FILE": "load",
        "LOADFROMFILE": "load",
        "LOAD-BASE-MODEL-POOL-FROM-FILE": "load",
        "NSGA-II": "nsga2",
        "NSGAII": "nsga2",
        "NSGA2": "nsga2",
        "NSGA-2": "nsga2",
    }
    
    if method not in method_mapping:
        raise ValueError(f"Unknown base model generation method: {method}. Choose from {list(np.unique(method_mapping.values()))}")
    
    return method_mapping.get(method, None)

def map_ensembling_method_to_unified_name(method: str) -> str:
    """
    Maps an ensembling method name to its unified name.

    Args:
        method (str): The ensembling method name to map.

    Returns:
        str: The unified name of the ensembling method.
    """
    
    if method is None:
        return None
    
    method = method.upper().replace("_", "-").replace(" ", "-")
    
    method_mapping = {
        "SINGLE": "single",
        "SINGLE-ENSEMBLE": "single",
        "SINGLE-ENSEMBLING": "single",
        "GES": "ges",
        "GES-ENSEMBLE": "ges",
        "GES-ENSEMBLING": "ges",
        "GREEDY-ENSEMBLE": "ges",
        "GREEDY-ENSEMBLING": "ges",
        "GREEDY-ENSEMBLE-SELECTION": "ges",
        "GREEDY": "ges",
    }
    
    if method not in method_mapping:
        raise ValueError(f"Unknown ensembling method: {method}. Choose one from {list(np.unique(list(method_mapping.values())))}.")
    
    return method_mapping.get(method, None)