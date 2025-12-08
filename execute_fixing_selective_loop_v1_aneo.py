import copy
import time
from functools import partial

import numpy as np
import torch
from loguru import logger
from torch.multiprocessing import Process, set_start_method

from data_handling import prepare_train_data, create_all_test_sets, load_data, get_optimization_goal_sequence_and_goal
from dataset.mtads_loading import load_all_stored_datasets
from execute_scoring import run_test_scoring, aneo_imputation, calculate_should_be_fixed_score
from global_utils import (
    format_exception,
    generator_seek,
    flow_parse_args,
    args_parameter_merger, global_random_sequence
)
from models.executor_factory import ExecutorFactory
from models.flow_imputers import point_or_ranged_imputation, calculate_imputation_error, \
    calculate_shadow_error, sort_and_group_ranges
from models.flow_param_optimizer import CMAParamOptimizer
from models.real_nvp import DEVICE, update_device

"""
Base script for running ANEO datasets with model interpolation methods and sanity check model selection and 
limiting the parameter search space
"""


def run(run_args=None, full_config=None, generator=(), device=None, **kwargs):
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

    # extract training anomalies
    flagged_training_anomalies = np.where(sample_anomalies == 1)[0]
    grouped_flagged_training_anomalies = point_or_ranged_imputation(flagged_training_anomalies)
    grouped_flagged_training_anomalies = sort_and_group_ranges(grouped_flagged_training_anomalies)
    # convert grouped flagged anomalies into a dict for tracking and fixing
    grouped_anomalies = {i: {"indexes": grouped_flagged_training_anomalies[i],
                             "imputed_in": [],
                             "prob_in": [],
                             "threshold_in": []} for i in
                         range(len(grouped_flagged_training_anomalies))}

    run_fixing = True
    last_run = False
    iteration = 0
    fix_iterations = 10  # 10 iterations is the maximum of iterations to run for fixing anomalies

    if len(grouped_anomalies.keys()) == 0:
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

        previous_run = str(executor.offline_dir).split("/")[-1]

        try:
            # setup thresholds
            train_re_ranged_nll_prob, train_all_probs_np = executor.predict(sequence_name, samples, sample_hist,
                                                                            sample_dates, sample_anomalies, False)
            _, train_latent_space = executor.predict_individual(sequence_name, samples, sample_hist, sample_dates,
                                                                sample_anomalies)
            executor.time_line_plot("BeforeImputing", samples, samples, train_all_probs_np, sample_anomalies,
                                    train_re_ranged_nll_prob, sample_dates, sample_dates,
                                    save_to_disk=True, show=False)
            # make_latent_space_grid_plot(train_latent_space)

            to_fix_entries = []
            for ix, fix_range_dict in grouped_anomalies.items():
                # calculate score if area should be fixed or not
                should_fix_flag, mean_fix_range_nll, upper_bound = calculate_should_be_fixed_score(
                    train_all_probs_np[:, 0],
                    sample_anomalies,
                    fix_range_dict["indexes"])
                if should_fix_flag:
                    to_fix_entries.append(ix)
                    # if the area should be fixed, add the range to the imputed ranges
                    grouped_anomalies[ix]["imputed_in"].append(iteration)
                    grouped_anomalies[ix]["prob_in"].append(mean_fix_range_nll)
                    grouped_anomalies[ix]["threshold_in"].append(upper_bound)

            if iteration < fix_iterations and len(to_fix_entries) > 0:
                run_fixing = True
            else:
                # no more sections to fix
                logger.warning(
                    f"No more sections to fix or max iteration reached, stopping the loop after {iteration} iterations."
                    f"Run one more iteration with detection goal.")
                optimization_metric = "auc_vus_balance"
                # switch the optimization metric to detection
                base_config["run"]["optimization_metric"] = optimization_metric
                # run one last time with detection goal
                run_fixing = True
                last_run = True
                # increase the iteration counter
                iteration += 1
                # skip this fixing iteration and jump to the last run
                continue

                # set the latent space sample mode
            latent_space_sample_mode = "sampling"

            fix_ranges = []
            for ix in to_fix_entries:
                # get the ranges to fix
                fix_ranges.append(grouped_anomalies[ix]["indexes"])
            # save the current status to the executor config
            executor.update_config("anomaly_imputations_status", {"anomaly_tracking": grouped_anomalies,
                                                                  "to_fix_entries": to_fix_entries})

            (samples, sample_hist, sample_anomalies, sample_dates,
             train_sequence) = aneo_imputation(executor, train_sequence, iteration, fix_ranges,
                                               latent_space_sample_mode,
                                               samples, sample_hist, sample_dates, sample_anomalies)

            if full_config["dataset"].get("shadow_channels", False):
                # run shadow channel analysis
                shadow_analysis = calculate_shadow_error(executor, samples, sample_hist, sample_dates, sample_anomalies)
                # save shadow channel analysis to the config
                executor.update_config("shadow_check", shadow_analysis)

            if full_config["run"].get("sanity_check", False):
                # run sanity check
                sanity_analysis = calculate_imputation_error(executor, samples, sample_hist, sample_dates,
                                                             sample_anomalies)
                # save sanity check to the config
                executor.update_config("sanity_check", sanity_analysis)

            # increase the iteration counter
            iteration += 1

        except (ValueError, AttributeError, BaseException, RuntimeError) as e:
            logger.error("An exception occurred: {}".format(e))
            print(format_exception(e))
            raise e

    logger.warning(f"END EXPERIMENT | {sequence_name}")


if __name__ == "__main__":
    args = flow_parse_args()

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    """
    # Example usage:
    --project="TFselfopt_fix_selective_loop_v1-aneo-testing"
    --max_past_range=51
    --self_optimization=True
    --model_type="tcNF-base"
    --dataset="aneo_with_noise"
    --shadow_channels=False
    --code_configuration={\"st_linear\":true,\"batch_norm\":\"none\",\"input_embedding\":\"none\"} 
    """

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
