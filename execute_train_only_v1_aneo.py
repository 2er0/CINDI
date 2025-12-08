import copy
import time
from functools import partial

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
Base script for running ANEO datasets without fixing anomalies in training data
"""


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

    try:
        (samples, sample_hist, sample_dates, sample_anomalies, add_noise,
         normalize_factors) = prepare_train_data(full_config, train_sequence)
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

    base_config = copy.deepcopy(full_config)

    iteration = "train_only"

    logger.info("No anomalies found in training data, skipping fixing loop")
    run_fixing = True
    last_run = True
    optimization_metric = "auc_vus_balance"
    base_config["run"]["optimization_metric"] = optimization_metric

    while run_fixing:
        logger.info(f"Running iteration {iteration} on sequence {sequence_name}")

        # run optimization search every iteration
        full_config = copy.deepcopy(base_config)
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
    current_args = None

    for i, gen in enumerate(all_iter):
        if i == 0:
            continue
        # if gen[0] != current_args["dataset"]["sequence_name"]:
        #    continue
        logger.debug(f"============ RUN IMPUTATION ============")
        # logger.debug(run_path)

        g, p, trains, tests, callback = load_data(gen)
        _gen = (g, p, trains, tests)

        run_func = partial(run, run_args=args, full_config=current_args, generator=_gen, device="cuda:0")
        p = Process(target=run_func)
        p.start()
        p.join()

        time.sleep(10)
