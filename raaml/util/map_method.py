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
        "MO-BO": "mo-bo",
        "MOBO": "mo-bo",
        "MO-SMAC": "mo-bo",
        "MOSMAC": "mo-bo",
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
        "SO-ASHA": "so-asha",
        "ASHA": "so-asha",
        "MO-ASHA": "mo-asha",
        "MOASHA": "mo-asha",
        "NSGA-II": "nsga2",
        "NSGAII": "nsga2",
        "NSGA2": "nsga2",
        "NSGA-2": "nsga2",
        "RANDOM": "random",
        "RANDOM-SEARCH": "random",      
        "RANDOMSEARCH": "random", 
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
        "QDO-ES": "qdo-es",
        "QDO-ENSEMBLE": "qdo-es",
        "QDO-ENSEMBLING": "qdo-es",
        "QDO-ENSEMBLE-SELECTION": "qdo-es",
        "QDO": "qdo-es",
        "QUALITY-DIVERSITY-OPTIMIZATION": "qdo-es",
        "QUALITY-DIVERSITY": "qdo-es",
        "QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLE-SELECTION": "qdo-es",
        "QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLING": "qdo-es",
        "QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLE": "qdo-es",
        "INFER-QDO-ES": "infer-qdo-es",
        "INFER-QDO-ENSEMBLE": "infer-qdo-es",
        "INFER-QDO-ENSEMBLING": "infer-qdo-es",
        "INFER-QDO-ENSEMBLE-SELECTION": "infer-qdo-es",
        "INFER-QDO": "infer-qdo-es",
        "INFER-QUALITY-DIVERSITY-OPTIMIZATION": "infer-qdo-es",
        "INFER-QUALITY-DIVERSITY": "infer-qdo-es",
        "INFER-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLE-SELECTION": "infer-qdo-es",
        "INFER-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLING": "infer-qdo-es",
        "INFER-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLE": "infer-qdo-es",
        "SIZE-QDO-ES": "size-qdo-es",
        "SIZE-QDO-ENSEMBLE": "size-qdo-es",
        "SIZE-QDO-ENSEMBLING": "size-qdo-es",
        "SIZE-QDO-ENSEMBLE-SELECTION": "size-qdo-es",
        "SIZE-QDO": "size-qdo-es",
        "SIZE-QUALITY-DIVERSITY-OPTIMIZATION": "size-qdo-es",
        "SIZE-QUALITY-DIVERSITY": "size-qdo-es",
        "SIZE-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLE-SELECTION": "size-qdo-es",
        "SIZE-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLING": "size-qdo-es",
        "SIZE-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLE": "size-qdo-es",
        "ENERGY-QDO-ES": "energy-qdo-es",
        "ENERGY-QDO-ENSEMBLE": "energy-qdo-es",
        "ENERGY-QDO-ENSEMBLING": "energy-qdo-es",
        "ENERGY-QDO-ENSEMBLE-SELECTION": "energy-qdo-es",
        "ENERGY-QDO": "energy-qdo-es",
        "ENERGY-QUALITY-DIVERSITY-OPTIMIZATION": "energy-qdo-es",
        "ENERGY-QUALITY-DIVERSITY": "energy-qdo-es",
        "ENERGY-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLE-SELECTION": "energy-qdo-es",
        "ENERGY-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLING": "energy-qdo-es",
        "ENERGY-QUALITY-DIVERSITY-OPTIMIZATION-ENSEMBLE": "energy-qdo-es",
        "STACKING": "ra-stacking",
        "STACK": "ra-stacking",
        "STACK-ENSEMBLE": "ra-stacking",
        "STACK-ENSEMBLING": "ra-stacking",
        "STACK-ENSEMBLE-SELECTION": "ra-stacking",
        "RA-STACKING": "ra-stacking",
        "RESOURCE-AWARE-STACKING": "ra-stacking",
        "RESOURCE-AWARE-STACK": "ra-stacking",
        "RESOURCE-AWARE-STACK-ENSEMBLE": "ra-stacking",
        "RESOURCE-AWARE-STACK-ENSEMBLING": "ra-stacking",
        "RESOURCE-AWARE-STACK-ENSEMBLE-SELECTION": "ra-stacking",
        "DUO-BRUTEFORCE": "duo-bruteforce",
        "SYSTEMATIC": "systematic",
        "MO-GES" : "mo-ges"
    }
    
    if method not in method_mapping:
        raise ValueError(f"Unknown ensembling method: {method}. Choose one from {list(np.unique(list(method_mapping.values())))}.")
    
    return method_mapping.get(method, None)