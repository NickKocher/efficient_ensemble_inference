import numpy as np
import pandas as pd

def filter_pareto_front(df_results):
    arr = df_results.values
    is_pareto = np.ones(arr.shape[0], dtype=bool)
    for i, row in enumerate(arr):
        is_pareto[i] = not np.any(np.all(arr < row, axis=1))
    return df_results[is_pareto].copy()


def is_pareto_optimal(df):
    arr = df.values
    is_pareto = np.ones(arr.shape[0], dtype=bool)
    
    for i, row in enumerate(arr):
        # Any other point that is no worse in all objectives and strictly better in at least one
        dominates = np.any(
            np.all(arr <= row, axis=1) & np.any(arr < row, axis=1)
        )
        if dominates:
            is_pareto[i] = False

    return is_pareto


def dominates(p, q):
    """Check if solution p dominates solution q"""
    return all(p <= q) and any(p < q)

def non_dominated_sort(df):
    """Perform non-dominated sorting on the DataFrame."""
    num_points = len(df)
    S = [[] for _ in range(num_points)]  # Solutions dominated by i
    n = [0 for _ in range(num_points)]   # Number of solutions dominating i
    rank = [0 for _ in range(num_points)]

    fronts = [[]]

    for p in range(num_points):
        for q in range(num_points):
            if p == q:
                continue
            if dominates(df.iloc[p], df.iloc[q]):
                S[p].append(q)
            elif dominates(df.iloc[q], df.iloc[p]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Remove empty last front
    if not fronts[-1]:
        fronts.pop()

    return fronts, rank


def crowding_distance(front, df):
    """Calculate crowding distance for all individuals in a front."""
    distance = np.zeros(len(front))
    if len(front) == 0:
        return distance

    num_objectives = df.shape[1]
    for i in range(num_objectives):
       
        obj_values = df.iloc[front, i].values
        sorted_indices = np.argsort(obj_values)
        sorted_front = [front[i] for i in sorted_indices]
        min_val = obj_values.min()
        max_val = obj_values.max()

        distance[sorted_indices[0]] = distance[sorted_indices[-1]] = np.inf

        for j in range(1, len(front) - 1):
            if max_val == min_val:
                distance[sorted_indices[j]] = np.inf
            else:
                prev_obj = df.iloc[sorted_front[j - 1], i]
                next_obj = df.iloc[sorted_front[j + 1], i]
                distance[sorted_indices[j]] += (next_obj - prev_obj) / (max_val - min_val)

    return distance


def nsga2_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a DataFrame using NSGA-II selection mechanism.
    Each row is a configuration, each column is an objective (to be minimized).
    """
    fronts, rank = non_dominated_sort(df)
    df['rank'] = rank
    df['crowding_distance'] = 0.0

    for front in fronts:
        distances = crowding_distance(front, df.iloc[:, :-2])
        df.loc[df.index[front], 'crowding_distance'] = distances
        
    return df

