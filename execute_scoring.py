from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from plotly_gif import GIF

from data_handling import create_all_test_sets
from data_preprocessing import de_normalize
from models.executor import Executor
from models.flow_executor import FlowExecutor
from models.flow_imputers import get_shadow_channels_and_time_features, get_sample
from plot_utils import plot_reconstruction_error, \
    plot_imputation_with_probability_background


def date_to_timestamp(date):
    return date.value // 10 ** 9


def move_time(date, value, unit):
    return date + pd.Timedelta(value, unit=unit)


def plot_time_and_latent_spaces(executor: Executor, test_name, samples, test, all_probs_np,
                                test_is_anomaly, re_ranged_nll_prob, sample_dates, test_data_w_dates,
                                sample_anomalies, individual_probs, latent):
    # stop plotting if it takes to long
    try:
        executor.time_line_plot(test_name, samples, test, all_probs_np, test_is_anomaly, re_ranged_nll_prob,
                                sample_dates, test_data_w_dates, show=False)
    except Exception as e:
        logger.error(f"Failed to create plots: {e}, continuing with next dataset")

    try:
        executor.test_line_plot(test_name, samples, test, all_probs_np, test_is_anomaly, re_ranged_nll_prob,
                                test_data_w_dates, show=False)
    except Exception as e:
        logger.error(f"Failed to create plots: {e}, continuing with next dataset")

    try:
        executor.latent_space_plot(test_name, samples, test, latent, individual_probs,
                                   sample_anomalies, test_is_anomaly)
    except Exception as e:
        logger.error(f"Failed to create plots: {e}, continuing with next dataset")

    try:
        executor.roc_curve_plot(test_name, re_ranged_nll_prob, test_is_anomaly)
    except Exception as e:
        logger.error(f"Failed to create plots: {e}, continuing with next dataset")


def run_test_scoring(executor: FlowExecutor, sample_data: tuple[np.ndarray], test_sequence: pd.DataFrame):
    full_config = executor.full_config
    normalize_factors = full_config["dataset"]["normalize_factors"]

    # run the test
    test_sequences = create_all_test_sets(test_sequence, full_config, normalize_factors)
    test_name, test, test_hist, test_is_anomaly, test_data_w_dates = next(test_sequences)

    re_ranged_nll_prob, all_probs_np = executor.predict(test_name, test, test_hist, test_data_w_dates,
                                                        test_is_anomaly, False)
    individual_probs, latent = executor.predict_individual(test_name, test, test_hist, test_data_w_dates,
                                                           test_is_anomaly)

    scores = {}
    scores[test_name] = executor.score(re_ranged_nll_prob, test_is_anomaly)
    scores["mean"] = scores[test_name]

    logger.info(scores[test_name])

    executor.update_config("score", {"test": scores})
    executor.save_config()

    # plot the results
    plot_time_and_latent_spaces(executor, test_name, sample_data[0], test, all_probs_np,
                                test_is_anomaly, re_ranged_nll_prob, sample_data[1], test_data_w_dates,
                                sample_data[2], individual_probs, latent)

    run_test_reconstruction_scoring(executor, test_sequence, length=48, days=7)


def run_test_reconstruction_scoring(executor: FlowExecutor, test_sequence: pd.DataFrame, length: int = 48,
                                    days: int = 7):
    full_config = executor.full_config

    max_past_range = full_config["dataset"]["max_past_range"]
    normalize_factors = full_config["dataset"]["normalize_factors"]

    sum_of_scores = 0.0
    plotting_collection = {}

    if full_config["dataset"]["name"] == "aneo_with_noise":
        start_timestamp = pd.to_datetime("2023-01-22 00:00:00", utc=True)
        scores = {"start": "2023-01-22 00:00:00"}
        step_size = 0
    else:
        start_timestamp = max_past_range
        scores = {"start": start_timestamp}
        if test_sequence.shape[0] < 200:
            step_size = (test_sequence.shape[0] - start_timestamp) // days
        else:
            step_size = length

    for i in range(0, days):
        if full_config["dataset"]["name"] == "aneo_with_noise":
            start = move_time(start_timestamp, i, "day")
            start_with_past = move_time(start, -max_past_range, "h")
            end = move_time(start, length, "h")
            test_sequence_filtered = test_sequence.loc[date_to_timestamp(start_with_past):date_to_timestamp(end)]
        else:
            start = i * (step_size // 2) + max_past_range
            start_with_past = max(0, start - max_past_range)
            end = start + step_size
            test_sequence_filtered = test_sequence.iloc[start_with_past:end]
        # run the test
        test_sequences = create_all_test_sets(test_sequence_filtered, full_config, normalize_factors)
        _, test_data, test_hist, test_is_anomaly, test_data_w_dates = next(test_sequences)
        test_data = test_data[-length:]  # take the last 48 steps of the test data
        test_hist = test_hist[-length:]
        test_is_anomaly = test_is_anomaly[-length:]
        test_data_w_dates = test_data_w_dates[-length:]

        (sanity_check_analysis, collection_of_samples,
         collection_of_sample_probs) = calculate_reconstruction_error(executor, test_data, test_hist,
                                                                      test_data_w_dates, test_is_anomaly)
        plotting_collection[i] = {"ground_truth": test_data,
                                  "samples": collection_of_samples,
                                  "sample_probs": collection_of_sample_probs}

        scores[i] = sanity_check_analysis
        sum_of_scores += sanity_check_analysis["mean_diff"]

    sum_of_scores /= days
    scores["mean"] = sum_of_scores

    logger.info(f"Mean reconstruction score of 7 days: {scores['mean']}")

    executor.update_config("score", {"forecast": scores})
    executor.save_config()

    # plot the results
    plot_reconstruction_error(executor.offline_dir, plotting_collection)


def calculate_reconstruction_error(executor: FlowExecutor, samples: np.ndarray, sample_hist: np.ndarray,
                                   sample_dates: np.ndarray, sample_anomalies: np.ndarray,
                                   sample_size: int = 1,
                                   diff_norm: Callable = np.linalg.norm, epsilon: float = 0.02,
                                   ignore_buffer: bool = False):
    current_past = sample_hist[0]
    collection_of_diffs = []
    collection_of_samples = []
    collection_of_sample_probs = []

    for i in range(samples.shape[0]):
        # sample from the center
        sampled = executor.sample(None, current_past, sample_dates[i])
        x_samples = np.array(sampled[0])
        # create mean sample
        mean_sample = np.mean(x_samples, axis=0)
        # create normed difference factor between actual sample and sampled
        sample_diff = diff_norm(samples[i] - mean_sample)
        collection_of_diffs.append(sample_diff)
        # update the history
        current_past = np.vstack([current_past[1:], mean_sample])
        # append the mean sample to the collection
        collection_of_samples.append(mean_sample)
        # append the sample probabilities to the collection
        collection_of_sample_probs.append(np.mean(sampled[1]))

    # calculate the mean of the differences
    mean_diff = np.mean(collection_of_diffs)
    # calculate the standard deviation of the differences
    std_diff = np.std(collection_of_diffs)
    # store the imputation score
    imputation_scores = {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "within_epsilon": mean_diff < epsilon,
        "diffs": collection_of_diffs,
    }

    return imputation_scores, np.array(collection_of_samples), np.array(collection_of_sample_probs)


def get_previous_latent_positions(executor: Executor, index: int, samples: np.ndarray, sample_hist: np.ndarray,
                                  sample_dates: np.ndarray, sample_anomalies: np.ndarray, hist_size=10):
    local_samples = samples[index - hist_size: index + 1]
    local_hist = sample_hist[index - hist_size: index + 1]
    local_dates = sample_dates[index - hist_size: index + 1]
    local_anomalies = sample_anomalies[index - hist_size: index + 1]
    _, latent = executor.predict_individual(None, local_samples, local_hist, local_dates,
                                            local_anomalies)
    return latent


def aneo_imputation(executor: FlowExecutor, train_sequence: pd.DataFrame, iteration: int, fix_ranges: list,
                    latent_space_sample_mode: str, samples, sample_hist, sample_dates, sample_anomalies,
                    use_mean_sample: bool = False):
    """
    Impute values in the training sequence based on the latent space of the model.
    :param executor: Executor instance to use for sampling and prediction.
    :param train_sequence: Source training sequence DataFrame to impute values into.
    :param iteration: Iteration number for logging and tracking.
    :param fix_ranges: List of ranges to fix in the training sequence.
    :param latent_space_sample_mode: str, mode for sampling from the latent space, either "sampling" or "var".
    :param samples: Precomputed samples from the training sequence.
    :param sample_hist: Historical data for the samples.
    :param sample_dates: Dates corresponding to the samples.
    :param sample_anomalies: Anomalies in the samples, used to filter out invalid samples.
    :param use_mean_sample: If True, use the mean sample for imputation instead of the center sampling.
    :return: samples, sample_hist, sample_anomalies, sample_dates
    """
    full_config = executor.full_config
    normalize_factors = full_config["dataset"]["normalize_factors"]
    sequence_channels = full_config["dataset"]["channels"]

    # calculate the difference between the max past view and the current past view of the model
    # the last position is the new index start past=[0:max_past_range - 1]
    position_correction = full_config["dataset"]["max_past_range"] - 1

    data_channels, shadow_channel_count, shadow_expected_value, time_features = get_shadow_channels_and_time_features(
        executor, samples, sample_anomalies)

    train_re_ranged_nll_prob, train_all_probs_np = executor.predict(0, samples, sample_hist,
                                                                    sample_dates, sample_anomalies, False)
    _, train_latent_space = executor.predict_individual(0, samples, sample_hist, sample_dates,
                                                        sample_anomalies)
    before_imputing_samples = np.copy(samples)
    train_sequence_unchanged = train_sequence.copy(deep=True)

    if not isinstance(fix_ranges[0], list):
        fix_ranges = [fix_ranges]

    for sub_iteration, fix_range in enumerate(fix_ranges):

        gif = GIF()
        # impute the values in the range

        # track the new values for imputation
        new_values = []
        # current hist tracker
        current_hist = None
        # store previous values for plotting
        new_sampled_values_for_plot = []
        new_sampled_nll_probs_for_plot = []

        for iteration_fixing_step, position in enumerate(
                range(max(0, np.min(fix_range)), np.max(fix_range) + 1)):
            original_position = position_correction + position
            last = position == np.max(fix_range)

            # Get the current samples history
            if current_hist is None:
                current_hist = sample_hist[position]
            else:
                # update the current history with not mutable values
                current_hist[data_channels:] = sample_hist[position, data_channels:]

            local_latent = get_previous_latent_positions(executor, position, samples, sample_hist,
                                                         sample_dates, sample_anomalies)

            valid_new_x_samples = []
            valid_new_nll_probs = []
            valid_new_z_probs = []
            valid_new_det = []
            valid_new_z_samples = []

            # Get the most likely sample in data space given the current history
            sampled, _ = get_sample(executor, None, samples[position], current_hist,
                                    sample_dates[position], sample_anomalies[position],
                                    shadow_channel_count, shadow_expected_value,
                                    time_features)

            # gather a minimum of 128 new samples that fit the current sample stable values
            gen_samples = 256
            while len(valid_new_x_samples) < 128:

                if latent_space_sample_mode == "sampling":
                    new_x_samples, z_sample = get_sample(executor, gen_samples, samples[position], current_hist,
                                                         sample_dates[position], sample_anomalies[position],
                                                         shadow_channel_count, shadow_expected_value,
                                                         time_features)

                else:
                    raise NotImplementedError()

                valid_new_x_samples.extend(new_x_samples[0])
                valid_new_nll_probs.extend(new_x_samples[1])
                valid_new_z_probs.extend(new_x_samples[2])
                valid_new_det.extend(new_x_samples[3])
                valid_new_z_samples.extend(z_sample)

            valid_new_x_samples = np.array(valid_new_x_samples)
            valid_new_nll_probs = np.array(valid_new_nll_probs)
            if use_mean_sample:
                # calculate mean and std for the new samples
                new_sample = valid_new_x_samples.mean(axis=0)
            else:
                new_sample = sampled[0][0]

            # update the current history with the new sample
            current_hist = np.vstack([current_hist[1:], new_sample])

            # set the new values in the training sequence
            # denormalize the value
            new_value = []
            sampled_new_values = []
            for i in range(len(new_sample)):
                new_value.append(de_normalize(new_sample[i], i, normalize_factors))
                sampled_new_values.append([de_normalize(valid_new_x_samples[j, i], i, normalize_factors) for j in
                                           range(valid_new_x_samples.shape[0])])

            sampled_new_values = np.array(sampled_new_values).T

            # store new possible values for plotting and the probabilities
            new_sampled_values_for_plot.append(sampled_new_values)
            new_sampled_nll_probs_for_plot.append(valid_new_nll_probs)

            old_value = train_sequence_unchanged.iloc[original_position, 1:-1].tolist()

            train_sequence.iloc[original_position, 1:-1] = new_value[:sequence_channels]

            logger.info(f"Imputed value at: {original_position}, {old_value} -> {new_value}")
            new_values.append((new_value, old_value, original_position))

            # create local view of the imputation change
            new_values_view = train_sequence.iloc[original_position - 48:original_position + 24, :].copy()

            y_all_options_nll_probs = None
            y_all_positions_de_normalized = None

            # get the full original values of the train sequence for plotting
            old_values_view = train_sequence_unchanged.iloc[original_position - 48:original_position + 24, :].copy()
            impute_position = train_sequence_unchanged.iloc[original_position, :].name

            # plot_imputation_change(old_values_view, new_values_view, impute_position, low_value, high_value,
            #                        y_all_options_nll_probs, y_all_positions_de_normalized,
            #                        local_latent, valid_new_z_samples,
            #                        executor.offline_dir,
            #                        f"Iteration: {iteration}, Step: {iteration_fixing_step} "
            #                        f"Position: {impute_position}",
            #                        False, gif)
            plot_imputation_with_probability_background(old_values_view, new_values_view,
                                                        impute_position, data_channels,
                                                        new_sampled_values_for_plot, new_sampled_nll_probs_for_plot,
                                                        f"Iteration: {iteration}, Step: {iteration_fixing_step}, "
                                                        f"Position: {datetime.fromtimestamp(impute_position)}",
                                                        True if last else False,
                                                        f"{executor.offline_dir}/{iteration}_{sub_iteration}_"
                                                        f"{np.min(fix_range)}_{np.max(fix_range)}_imputation",
                                                        gif)

        # recompute the train sequence with the new values
        train_sequences_recompute = create_all_test_sets(train_sequence, full_config, normalize_factors)
        _, samples, sample_hist, sample_anomalies, sample_dates = next(train_sequences_recompute)

        # generate gif of the imputation change
        gif_file = f"{executor.offline_dir}/{iteration}_{sub_iteration}_{np.min(fix_range)}_{np.max(fix_range)}_imputation.gif"
        length = (np.max(fix_range) - np.min(fix_range)) * 200
        gif.create_gif(gif_path=gif_file, length=length)

        # save the imputed values to the config
        executor.update_config("imputed_values",
                               {f"iteration_{iteration}_{sub_iteration}": new_values,
                                f"start_{iteration}_{sub_iteration}": np.min(fix_range),
                                f"end_{iteration}_{sub_iteration}": np.max(fix_range)})
        executor.save_config()

    executor.time_line_plot(f"BeforeAndAfterImputing_{iteration}",
                            before_imputing_samples, samples, train_all_probs_np,
                            sample_anomalies,
                            train_re_ranged_nll_prob, sample_dates, sample_dates,
                            save_to_disk=True, show=False)

    _, after_train_probs_np = executor.predict(0, samples, sample_hist,
                                               sample_dates, sample_anomalies, False)

    executor.before_after_time_line_plot(iteration,
                                         before_imputing_samples, train_all_probs_np[:, 0],
                                         samples, after_train_probs_np[:, 0],
                                         sample_anomalies, sample_dates,
                                         save_to_disk=True, show=False)

    return samples, sample_hist, sample_anomalies, sample_dates, train_sequence


def calculate_should_be_fixed_score(nll: np.ndarray, sample_anomalies: np.ndarray, fix_range: list,
                                    std_factor: float = 2.0):
    """
    Calculate the score for the should be fixed areas.
    :param nll: The negative log likelihood of the samples.
    :param sample_anomalies: The anomalies in the samples.
    :param fix_range: The ranges that should be fixed.
    :param std_factor: The factor to multiply the standard deviation with to get a wider range.
    :return: The score for the should be fixed areas.
    """

    # get random areas to calculate a mean and standard deviation of the nll
    non_anomaly_sections = np.where(sample_anomalies == 0)[0]
    set_non_anomaly_sections = set(non_anomaly_sections)
    # pick 10 random sections of the non-anomaly sections with a minimum length of 24 hours or a maximum of 72 hours if fix_range is larger
    selected_probs = []
    counter = 0
    while counter < 10:
        start = np.random.choice(non_anomaly_sections)
        end = start + np.random.randint(24, 72)
        indexes = list(range(start, end + 1))
        # ensure that the indexes are within the non anoamly sections
        if not set(indexes).issubset(set_non_anomaly_sections):
            continue
        if end >= len(nll):
            continue
        selected_probs.extend(nll[start:end])
        counter += 1

    # calculate the mean and standard deviation of the nll in the range
    mean_nll = np.mean(selected_probs)
    std_nll = np.std(selected_probs) * std_factor  # multiply by # to get a wider range for the standard deviation
    # calculate the upper bound
    upper_bound = mean_nll + std_nll

    # calculate the mean nll in the fix range
    mean_fix_range_nll = np.mean(nll[fix_range[0]:fix_range[-1]])

    # check if the mean nll in the fix range is larger than the upper bound
    if mean_fix_range_nll > upper_bound:
        logger.info(f"Should be fixed score: {mean_fix_range_nll} > {upper_bound}, ")
        return True, mean_fix_range_nll, upper_bound
    else:
        logger.info(f"Should not be fixed, score for range {fix_range} : {mean_fix_range_nll} <= {upper_bound}")
        return False, mean_fix_range_nll, upper_bound
