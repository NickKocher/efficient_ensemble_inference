from bisect import bisect_left
import numpy as np
import pandas as pd
import lzma
import json
import os
import re
from os import listdir

ESU = 10000
SCALING_FACTOR = 1 / 2**ESU
ONEMILLION = 1000000
ENERGYTOJOULE = 15 / ONEMILLION


# Code adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns index from closest value to myNumber.

    If two numbers are equally close, return index from the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0
    if pos == len(myList):
        return len(myList) - 1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos - 1


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def calculate_pareto_set(df):
    """
    Calculate the Pareto-optimal set of algorithms for each dataset (column).

    Parameters:
    - df: Pandas DataFrame where each cell contains [energy, loss].

    Returns:
    - A dictionary with dataset names as keys and lists of Pareto-optimal algorithms as values.
    """
    pareto_sets = {}
    pareto_points = {}
    for dataset in df.columns:
        # Extract loss and time for all algorithms for the current dataset
        points = np.array(df[dataset].tolist())

        # Identify Pareto-optimal algorithms
        optimal_mask = is_pareto_efficient_simple(points)
        pareto_algorithms = df.index[optimal_mask].tolist()

        pareto_sets[dataset] = pareto_algorithms
        pareto_points[dataset] = points[optimal_mask]

    return pareto_sets, pareto_points


def create_metadata_from_smac(
    path_to_dataset_measurements: str,
    cores: int,
    dataset_id: int,
    outertest: bool = False,
    validate: bool = False,
    runtime: int = 24,
    get_predictions: bool = False,
    meta_method: str = "metafeatures",
    obj: str = "mo",
    meta_source: str = "mo",
):
    cores_socket = cores + 2
    print(dataset_id, flush=True)
    # currently supports one seed
    seeds = [42]
    folds = 5
    for seed in seeds:
        log_path = get_smac_associated_logs(
            dataset_id=dataset_id,
            cores_socket=cores_socket,
            seed=seed,
            validate=validate,
            runtime=runtime,
            meta_method=meta_method,
            meta_source=meta_source,
            obj=obj,
        )

        # print(log_path)
        measurements = listdir(log_path)
        for index in range(len(measurements)):
            measurements[index] = re.split("_", measurements[index])[0]
        measurements = list(set(measurements))

        # sort measurements?
        # np.sort(measurements)

        id_to_metric = dict()
        id_to_config = dict()
        metadata = dict()
        current_path_to_dataset_measurements = os.path.join(
            path_to_dataset_measurements, f"{seed}"
        )
        incumbent_json = os.path.join(
            current_path_to_dataset_measurements, "intensifier.json"
        )
        configs_json = os.path.join(
            current_path_to_dataset_measurements, "runhistory.json"
        )
        with open(incumbent_json) as incumbent_file, open(configs_json) as config_file:
            intensifier = json.load(incumbent_file)
            configs = json.load(config_file)
            trajectory = intensifier["trajectory"]
            costs = trajectory[-1]["costs"]
            incumbent_ids = trajectory[-1]["config_ids"]
            multiobjective = type(costs[0]) is list
            for index, id in enumerate(incumbent_ids):
                if multiobjective:
                    cost = [costs[index][0], costs[index][1] * ENERGYTOJOULE]
                else:
                    cost = [costs[index]]
                id_to_metric[id] = cost
                id_to_config[id] = configs["configs"][f"{id}"]

            configs_data = configs["data"]
            all_configs = configs["configs"]
            if get_predictions:
                only_incumbent = False

            else:
                only_incumbent = True
            # scale resulting metric

            for i in range(len(configs_data)):
                id = configs_data[i][0]
                if multiobjective:
                    configs_data[i][4][1] = np.exp(
                        configs_data[i][4][1]
                    )  # * ENERGYTOJOULE
                if not only_incumbent or id in id_to_metric:
                    # reset values
                    fit_energy_cores = np.inf
                    fit_energy_sockets = np.inf
                    fit_time = np.inf
                    predict_energy_cores = np.inf
                    predict_energy_sockets = np.inf
                    predict_time = np.inf
                    # print(configs_data[i][9])
                    # Runstatus empty indicates successful run
                    if not configs_data[i][9]:
                        file_count_fit = 0
                        file_count_predict = 0
                        # reset values
                        fit_energy_cores = 0
                        fit_energy_sockets = 0
                        fit_time = 0
                        predict_energy_cores = 0
                        predict_energy_sockets = 0
                        predict_time = 0
                        # find the log folders that is closest to the measurement by smac
                        upper_bound = int(configs_data[i][8] * 1000 * ONEMILLION)
                        lower_bound = int(configs_data[i][7] * 1000 * ONEMILLION)
                        # print(upper_bound, lower_bound)
                        # print(measurements[0],measurements[-1])
                        base_paths = [
                            x
                            for x in measurements
                            if int(x) >= lower_bound and int(x) <= upper_bound
                        ]

                        # print(base_paths)
                        for base_path in base_paths:
                            base_path = os.path.join(log_path, base_path)
                            # extend to all folds
                            fold_paths = [base_path + f"_{j}" for j in range(folds)]
                            for fold_path in fold_paths:
                                current_path = fold_path + "_fit"
                                if os.path.exists(current_path):
                                    # get training costs
                                    current_configuration_data = np.load(
                                        lzma.open(current_path, mode="rb"),
                                        allow_pickle=True,
                                    )
                                    # get energy time and hardwarecounters for predict
                                    fit_energy_cores += get_energy_from_cores(
                                        measurement=current_configuration_data,
                                        cores=cores,
                                    )
                                    fit_energy_sockets += get_energy_from_sockets(
                                        measurement=current_configuration_data
                                    )
                                    fit_time += current_configuration_data[1]
                                    file_count_fit += 1
                                current_path = fold_path + "_predict"
                                if os.path.exists(current_path):
                                    # get inference costs

                                    current_configuration_data = np.load(
                                        lzma.open(current_path, mode="rb"),
                                        allow_pickle=True,
                                    )
                                    # get energy time and hardwarecounters for predict
                                    predict_energy_cores += get_energy_from_cores(
                                        measurement=current_configuration_data,
                                        cores=cores,
                                    )
                                    predict_energy_sockets += get_energy_from_sockets(
                                        measurement=current_configuration_data
                                    )
                                    predict_time += current_configuration_data[1]
                                    file_count_predict += 1
                                    # if get_predictions:
                                    #     current_path = fold_path + "_predictions"
                                    #     if os.path.exists(current_path):
                                    #         current_predictions = np.load(
                                    #             lzma.open(current_path, mode="rb"),
                                    #             allow_pickle=True,
                                    #         )

                        if file_count_fit > 0:
                            fit_energy_cores = fit_energy_cores
                            fit_energy_sockets = fit_energy_sockets
                            fit_time = fit_time
                        else:
                            fit_energy_cores = np.inf
                            fit_energy_sockets = np.inf
                            fit_time = np.inf
                            predict_energy_cores = np.inf
                            predict_energy_sockets = np.inf
                            predict_time = np.inf
                        if file_count_predict > 0:
                            predict_energy_cores = predict_energy_cores
                            predict_energy_sockets = predict_energy_sockets
                            predict_time = predict_time
                        else:
                            fit_energy_cores = np.inf
                            fit_energy_sockets = np.inf
                            fit_time = np.inf
                            predict_energy_cores = np.inf
                            predict_energy_sockets = np.inf
                            predict_time = np.inf
                    else:
                        if multiobjective:
                            configs_data[i][4][0] = np.inf
                        else:
                            configs_data[i][4] = np.inf
                    if multiobjective:
                        metadata[id] = [
                            fit_energy_cores * ENERGYTOJOULE,
                            fit_energy_sockets * ENERGYTOJOULE,
                            fit_time / ONEMILLION,
                            predict_energy_cores * ENERGYTOJOULE,
                            predict_energy_sockets * ENERGYTOJOULE,
                            predict_time / ONEMILLION,
                            configs_data[i][4][0],
                        ]
                    else:
                        metadata[id] = [
                            fit_energy_cores * ENERGYTOJOULE,
                            fit_energy_sockets * ENERGYTOJOULE,
                            fit_time / ONEMILLION,
                            predict_energy_cores * ENERGYTOJOULE,
                            predict_energy_sockets * ENERGYTOJOULE,
                            predict_time / ONEMILLION,
                            configs_data[i][4],
                        ]
                    pd.DataFrame.from_dict(metadata).to_parquet(
                        os.path.join(
                            current_path_to_dataset_measurements, "metadata.parquet"
                        )
                    )
            yield (
                multiobjective,
                id_to_metric,
                id_to_config,
                configs_data,
                metadata,
                all_configs,
            )


# WIP
def get_metric_location(metric: str, inference: bool = False) -> str:
    if metric == "bacc":
        metric_location = "-bacc"
    else:
        metric_location = ""
    if inference:
        metric_location = f"{metric_location}-inference"
    return metric_location


def get_smac_associated_logs(
    dataset_id: int,
    cores_socket: int,
    seed: int,
    validate: bool = False,
    runtime: int = 24,
    metric: str = "bacc",
    meta_method: str = "metafeatures",
    obj: str = "mo",
    meta_source: str = "mo",
) -> str:
    metric_location = get_metric_location(metric)

    smac_name = f"{dataset_id}_{cores_socket}"
    log_name = f"{smac_name}_{seed}"
    log_name = os.path.join(
        os.getcwd(),
        f"logs/smac-{runtime}h{metric_location}/{meta_method}/{meta_source}/{obj}/{log_name}",
    )
    return log_name


def energy_to_joule(energy):
    ONEMILLION = 1000000
    ENERGYTOJOULE = 15.3 / ONEMILLION
    return energy * ENERGYTOJOULE
    # return (SCALING_FACTOR * (energy) * 1000000)


def get_energy_from_cores(measurement, cores):
    # adjust for also measuring sockets
    return np.sum(measurement[-(cores + 2) : -2])


def get_energy_from_sockets(measurement):
    return measurement[-1]


def get_latest_modified_folder(
    dataset_measurements: list[str], basepath: str = ""
) -> str:
    latest_measurement = os.path.join(basepath, dataset_measurements[0])
    for path in dataset_measurements:
        if os.path.getmtime(os.path.join(basepath, path)) > os.path.getmtime(
            latest_measurement
        ):
            latest_measurement = os.path.join(basepath, path)
    return latest_measurement


# for non smac measurements
def get_dataset_id_task_from_path(path_to_dataset_measurements: str) -> tuple[int, str]:
    subdirectories = re.split("/", path_to_dataset_measurements)
    for i in range(len(subdirectories)):
        if subdirectories[i] == "classification" or subdirectories[i] == "regression":
            # return dataset id and task. dataset id a subdirectory of the task
            return int(subdirectories[i + 1]), subdirectories[i]


def is_dominated(cost, incumbent_cost) -> bool:
    dominated_attribute = False
    dominates_attribute = False
    for index in range(len(cost)):
        if cost[index] > incumbent_cost[index]:
            dominated_attribute = True
        if cost[index] < incumbent_cost[index]:
            dominates_attribute = True
    return dominated_attribute and not dominates_attribute


def get_meta_method_smac_path(
    basepath,
    use_meta: bool = False,
    validate: bool = False,
    meta_method: str = "metafeatures",
    obj: str = "mo",
    runtime: int = 24,
    meta_source: str = "mo",
    metric: str = "bacc",
    inference: bool = False,
    validate_selection: bool = False,
):
    metric_location = get_metric_location(metric, inference=inference)
    if validate:
        if validate_selection:
            validate = "validate-selection"
        else:
            validate = "validate"
        if use_meta:
            output_directory = os.path.join(
                f"{basepath}",
                "smac_logs",
                f"{validate}-energy-meta{metric_location}",
                f"{meta_method}",
                f"{meta_source}",
                f"{obj}-{runtime}h",
            )
        else:
            output_directory = os.path.join(
                f"{basepath}",
                "smac_logs",
                f"{validate}-energy-{obj}-{runtime}h{metric_location}",
            )
    else:
        output_directory = os.path.join(
            f"{basepath}",
            "smac_logs",
            f"energy-{obj}-with-outer-test-{runtime}h{metric_location}",
        )

    return output_directory


def get_plot_directory(
    basepath,
    meta_method: str = "metafeatures",
    obj: str = "mo",
    runtime: int = 24,
    meta_source: str = "mo",
    plot_name: str = "pareto",
    metric: str = "bacc",
    inference: bool = False,
    selection: bool = False,
):
    metric_location = get_metric_location(metric, inference=inference)
    if selection:
        plot_name = f"{plot_name}-selection"
    plot_directory = os.path.join(
        f"{basepath}",
        f"plots{metric_location}",
        f"{plot_name}",
        f"{obj}",
        f"{meta_source}",
        f"{meta_method}",
    )
    return plot_directory
