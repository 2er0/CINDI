import copy
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.multiprocessing import Process, set_start_method

from data_handling import prepare_train_data, create_all_test_sets, load_data, get_optimization_goal_sequence_and_goal
from dataset.mtads_loading import load_all_stored_datasets
from execute_scoring import run_test_scoring
from global_utils import (
    format_exception,
    generator_seek,
    flow_parse_args,
    args_parameter_merger, global_random_sequence
)
from models.executor_factory import ExecutorFactory
from models.flow_param_optimizer import CMAParamOptimizer
from models.real_nvp import DEVICE, update_device

"""
Base script for running ANEO datasets with external imputation methods for fixing anomalies in the training data.
"""


def create_dataset_with_nans(train_sequence_source: pd.DataFrame) -> pd.DataFrame:
    # copy the train sequence to avoid modifying the original
    train_sequence = train_sequence_source.copy(deep=True)
    # reset index
    train_sequence.reset_index(drop=True, inplace=True)

    # get all rows with 1 in the last column
    train_sequence_anomaly_indexes = train_sequence.index[train_sequence.iloc[:, -1] == 1].tolist()

    # set the values in the rows with anomalies to NaN for all columns except the first and last
    for index in train_sequence_anomaly_indexes:
        train_sequence.iloc[index, 1:-1] = np.nan

    return train_sequence


def interpolate_with_func(train_sequence_source: pd.DataFrame, interpolate_func,
                          context_length: int = 3600, needs_grad: bool = False) -> pd.DataFrame:
    # copy the train sequence to avoid modifying the original
    train_sequence = train_sequence_source.copy(deep=True)
    # reset index
    train_sequence.reset_index(drop=True, inplace=True)

    # get all rows with 1 in the last column
    train_sequence_anomaly_indexes = train_sequence.index[train_sequence.iloc[:, -1] == 1].tolist()
    # group the indexes into consecutive sequences
    grouped_anomaly_indexes = []
    current_group = []
    for index in train_sequence_anomaly_indexes:
        if not current_group or index == current_group[-1] + 1:
            current_group.append(index)
        else:
            grouped_anomaly_indexes.append(current_group)
            current_group = [index]
    if current_group:
        grouped_anomaly_indexes.append(current_group)

    # interpolate each group of consecutive indexes
    for group in grouped_anomaly_indexes:
        logger.info(f"Interpolating indexes: {group}")
        start_interpolate_idx = group[0]
        prediction_length = group[-1] - group[0] + 1
        # get the data to interpolate
        if context_length == -1:
            data_start_idx = 0
        elif start_interpolate_idx - context_length < 0:
            data_start_idx = 0
        else:
            data_start_idx = start_interpolate_idx - context_length

        data_numpy = train_sequence.iloc[data_start_idx:start_interpolate_idx, 1:-1].to_numpy()
        # save to numpy
        # np.save("data_to_interpolate.npy", data_numpy[-context_length:, :])
        # convert to torch tensor
        data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
        # run interpolation method
        if needs_grad:
            reconstruction = interpolate_func(data_tensor, prediction_length)  # add batch dimension
        else:
            with torch.no_grad():
                reconstruction = interpolate_func(data_tensor, prediction_length)  # add batch dimension

        # convert back to numpy
        reconstruction_np = reconstruction.cpu().numpy()  # remove batch dimension
        # insert reconstructed values back into the train sequence
        train_sequence.iloc[start_interpolate_idx: start_interpolate_idx + prediction_length, 1:-1] = reconstruction_np

    # return the interpolated train sequence
    return train_sequence


def interpolate_with_dynamix(train_sequence_source: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the training sequence with the given interpolation method.
    :param train_sequence_source: The training sequence to interpolate.
    :param interpolation_method: The interpolation method to use.
    :return: The interpolated training sequence.
    """
    from models_external.dynamix.src.utilities.utilities import load_hf_model
    from models_external.dynamix.src import DynaMixForecaster
    # Load the pre-trained model
    model = load_hf_model("dynamix-3d-alrnn-v1.0")
    # Set model to evaluation mode
    model.eval()
    # Create a forecaster with the trained model
    forecaster = DynaMixForecaster(model)

    def interpolate_func(data_tensor: torch.Tensor, prediction_length: int):
        return forecaster.forecast(
            context=data_tensor,
            horizon=prediction_length,
            preprocessing_method="pos_embedding",
            standardize=True,
            fit_nonstationary=False,
            initial_x=None
        )

    train_sequence = interpolate_with_func(train_sequence_source, interpolate_func)

    return train_sequence


def interpolate_with_knowimp(train_sequence_source: pd.DataFrame) -> pd.DataFrame:
    # create dataset with nans
    train_sequence = create_dataset_with_nans(train_sequence_source)

    if os.getenv("DATAREDUCE", "False").lower() == "true":
        # use only last 3600 samples for imputation to save time
        logger.warning("Reducing data for imputation to last 3600 samples")
        train_sequence = train_sequence.iloc[-3600:]

    from models_external.KnewImp.model.wgf_imp import NeuralGradFlowImputer
    from models_external.KnewImp.utils.utils import enable_reproducible_results

    # default args from exper_wgf.py
    def interpolate_func(data_tensor: torch.Tensor, prediction_length: int):
        # data_tensor: [context_length, num_features] -> might need to reduce context length for speed
        data_numpy = data_tensor.numpy()
        # create nan numpy array of [prediction_length, num_features]
        to_fill_section = np.full((prediction_length, data_tensor.shape[1]), fill_value=np.nan, dtype=np.float32)
        data_with_nans = np.vstack([data_numpy, to_fill_section]).copy()

        enable_reproducible_results(4)
        model = NeuralGradFlowImputer(entropy_reg=10.0, bandwidth=0.5,
                                      score_net_epoch=200, niter=2,
                                      initializer=None, mlp_hidden=[128, 128], lr=1.0e-1,
                                      score_net_lr=1.0e-3)

        x_filled = model.fit_transform(data_with_nans, verbose=False, report_interval=10)

        return torch.tensor(x_filled[-prediction_length:, :], dtype=torch.float32)

    train_sequence = interpolate_with_func(train_sequence, interpolate_func, context_length=-1, needs_grad=True)

    return train_sequence


interpolation_func_dict = {"dynamix": interpolate_with_dynamix,
                           "knowimp": interpolate_with_knowimp}


def run(run_args=None, full_config=None, generator=(), device=None, **kwargs):
    # hacking the system to have more details over the data for fixing
    if run_args is not None:
        max_past_range = run_args["dataset"]["max_past_range"]
    else:
        max_past_range = full_config["dataset"]["max_past_range"]
    sequence_name, dataset_parameters, train_sequence, test_sequence, _ = load_data(generator, max_past_range)
    if "no-anomaly" in sequence_name:
        # skip the no-anomaly sequences
        logger.info("Skipping anomaly in sequence")
        return

    if DEVICE == torch.device("cpu"):
        logger.info("No GPU available - running on CPU")
        device = update_device("cpu")
    elif isinstance(device, str):
        device = update_device(device)
    else:
        device = DEVICE
    logger.info(f"Running with device: {device}")

    logger.warning(f"START EXPERIMENT | {sequence_name}")
    if run_args is not None:
        full_config = args_parameter_merger(run_args, sequence_name, dataset_parameters)
    logger.info(full_config)
    # create copy of base config for resetting for each interpolation method
    base_config = copy.deepcopy(full_config)

    interpolation_method = kwargs.get("interpolation_method", "knowimp")
    interpolation_func = interpolation_func_dict[interpolation_method]

    logger.warning(f"Running with interpolation method: {interpolation_method}")
    train_sequence_interpolated = interpolation_func(train_sequence)

    full_config = copy.deepcopy(base_config)
    full_config["project"]["interpolation_method"] = interpolation_method

    try:
        (samples, sample_hist, sample_dates, sample_anomalies, add_noise,
         normalize_factors) = prepare_train_data(full_config, train_sequence_interpolated)
    except (IndexError, TypeError) as e:
        logger.info("Not enough data for training")
        logger.error("BaseException: An exception occurred: {}".format(e))
        print(format_exception(e))
        return

    # check if test_sequence is a tuple containing validation and test sets
    if isinstance(test_sequence, tuple):
        # if so, we need to create test sets for both validation and test
        validation_sequences = create_all_test_sets(test_sequence[0], full_config, normalize_factors)
        test_sequences = create_all_test_sets(test_sequence[1], full_config, normalize_factors)
    else:
        validation_sequences = [(None, None, None, None, None)]  # Placeholder dummy for validation sequences
        test_sequences = create_all_test_sets(test_sequence, full_config, normalize_factors)

    full_config["run"]["device"] = device
    full_config["dataset"]["normalize_factors"] = normalize_factors
    previous_run = ""
    full_config["project"]["previous_run"] = previous_run
    unique_identifier = global_random_sequence()
    full_config["project"]["unique_identifier"] = unique_identifier

    # setup self optimization parameter search
    if not full_config["run"]["self_optimization"]:
        raise ValueError("Self optimization is not enabled")

    if isinstance(test_sequence, tuple):
        _, opt_eval, opt_eval_hist, opt_eval_anomalies, opt_eval_dates = next(validation_sequences)
        # optimization_metric = "auc_vus_balance"
        optimization_metric = "auc_vus_sanity_balance"
    else:
        (opt_eval, opt_eval_hist, opt_eval_dates, opt_eval_anomalies,
         optimization_metric) = get_optimization_goal_sequence_and_goal(full_config,
                                                                        load_all_stored_datasets,
                                                                        normalize_factors)
    _, opt_test, opt_test_hist, opt_test_anomalies, opt_test_dates = next(test_sequences)

    full_config["run"]["optimization_metric"] = optimization_metric

    iteration = interpolation_method

    logger.info("No anomalies found in training data, skipping fixing loop")
    run_fixing = True
    last_run = True
    optimization_metric = "auc_vus_balance"
    base_config["run"]["optimization_metric"] = optimization_metric

    while run_fixing:
        logger.info(f"Running iteration {iteration} on sequence {sequence_name}")

        # run optimization search every iteration
        full_config["project"]["previous_run"] = previous_run

        optimizer = CMAParamOptimizer(full_config, samples, sample_hist, sample_dates,
                                      sample_anomalies,
                                      opt_eval, opt_eval_hist, opt_eval_dates, opt_eval_anomalies,
                                      opt_test, opt_test_hist, opt_test_dates, opt_test_anomalies,
                                      metric=optimization_metric)
        optimizer.optimize()
        best_config, optimization_trace, best_model_weights = optimizer.get_results()
        full_config["model"]["code_configuration"] = best_config["model"]["code_configuration"]

        executor = ExecutorFactory.create_executor(full_config)
        executor.load_model_from_weights(best_model_weights)
        executor.start_logging(f"{iteration}")
        executor.update_config("optimization_trace", optimization_trace)
        executor.save_model()

        # save imputed training sequence for analysis
        train_sequence_interpolated.to_csv(f"{executor.offline_dir}/imputed_train_sequence_{interpolation_method}.csv",
                                           index=False)

        # evaluate the model on the training set
        if isinstance(test_sequence, tuple):
            run_test_on = test_sequence[1]
        else:
            run_test_on = test_sequence

        # run test scoring
        run_test_scoring(executor, (samples, sample_dates, sample_anomalies), run_test_on)

        if last_run:
            logger.info(f"Finished iteration {iteration} on sequence {sequence_name}")
            run_fixing = False
            break

    logger.warning(f"END EXPERIMENT | {sequence_name}")


if __name__ == "__main__":
    args = flow_parse_args()

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    all_iter = generator_seek(load_all_stored_datasets(args["dataset"]["name"]))
    current_args = {"dataset": {"sequence_name": "aneo_grid1_1.04_timeFalse"}}

    for gen in all_iter:
        if gen[0] != current_args["dataset"]["sequence_name"]:
            continue
        logger.debug(f"============ RUN IMPUTATION ============")
        # logger.debug(run_path)

        g, p, trains, tests, callback = load_data(gen)
        _gen = (g, p, trains, tests)

        run_func = partial(run, run_args=args, full_config=current_args, generator=_gen, device="cuda:0")
        p = Process(target=run_func)
        p.start()
        p.join()

        time.sleep(10)
