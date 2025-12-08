from collections.abc import Callable
from itertools import groupby
from operator import itemgetter

import numpy as np
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

from models.executor import Executor
from models.flow_executor import FlowExecutor


def point_or_ranged_imputation(indexes: np.ndarray):
    """
    Check how long the first continuous range is.
    :param indexes: The indexes to check.
    :return: Indexes to impute
    """
    if len(indexes) == 1:
        return indexes
    else:
        ranges = []
        for k, g in groupby(enumerate(indexes), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            ranges.append(group)
        return ranges


def sort_and_group_ranges(groups: list[list], max_length: int = 24):
    """
    Sort and group the ranges by length.
    :param groups: The groups to sort and group.
    :param max_length: The maximum length of the ranges.
    :return: Sorted and grouped ranges.
    """
    # sort by length
    sorted_groups = sorted(groups, key=len)
    return sorted_groups
    # # group by length
    # grouped_ranges = []
    # other_ranges = []
    # for g in sorted_groups:
    #     if len(g) <= max_length:
    #         grouped_ranges.append(g)
    #     else:
    #         other_ranges.append(g)
    # return [grouped_ranges, *other_ranges]


def filter_samples_by_shadow_channel_epsilon(samples: np.ndarray,
                                             shadow_channel_count: int, shadow_expected_value: float,
                                             epsilon: float = 0.05):
    """
    Filter the samples based on the shadow channel and epsilon.
    :param samples: The samples to filter.
    :param shadow_channel_count: The number of shadow channels in the end of each sample
    :param shadow_expected_value: The expected value of the shadow channels
    :param epsilon: The epsilon error threshold.
    :return:
    """
    samples_shadow_channels = samples[:, -shadow_channel_count:]
    # calculate the difference
    samples_shadow_diff = shadow_expected_value - samples_shadow_channels
    # calculate absolute difference
    samples_shadow_diff_abs = np.abs(samples_shadow_diff)
    # calculate the mean diff per sample
    mean_diff = np.mean(samples_shadow_diff_abs, axis=1)
    # filter based on epsilon
    within_epsilon = np.where(mean_diff < epsilon)[0]
    return within_epsilon


def filter_samples_by_time_features(new_x_samples: np.ndarray, samples: np.ndarray, time_features: int = 6,
                                    epsilon: float = 0.05):
    """
    Filter the samples based on the time features and epsilon.
    :param new_x_samples: The new samples to filter.
    :param samples: The original samples to compare against.
    :param time_features: The number of time features in the samples.
    :param epsilon: The epsilon error threshold.
    :return: The filtered samples.
    """
    # calculate the difference
    diff = np.abs(new_x_samples[:, -time_features:] - samples[-time_features:])
    # calculate the mean diff per sample
    mean_diff = np.mean(diff, axis=1)
    # filter based on epsilon
    within_epsilon = np.where(mean_diff < epsilon)[0]
    return within_epsilon


def calculate_imputation_error(executor: FlowExecutor, samples: np.ndarray, sample_hist: np.ndarray,
                               sample_dates: np.ndarray, sample_anomalies: np.ndarray,
                               sample_size: int = 100,
                               imputation_length: int = 50, max_areas: int = 5,
                               diff_norm: Callable = np.linalg.norm, epsilon: float = 0.2,
                               ignore_buffer: bool = False):
    """
    Calculate the imputation error by sampling and comparing the samples with
    the original data as a self-regressive process.

    :param executor: The executor to use for sampling.
    :param samples: The original samples to compare against.
    :param sample_hist: The history of the samples.
    :param sample_dates: The dates of the samples.
    :param sample_anomalies: The anomalies in the samples.
    :param sample_size: The number of samples to generate for each step.
    :param imputation_length: The length of the imputation range.
    :param max_areas: The maximum number of areas to test for imputation.
    :param diff_norm: The function to use for calculating the norm of the difference.
    :param epsilon: The error threshold for the imputation.
    :param ignore_buffer: Whether to ignore the buffer at the beginning of the ranges.
    """
    # get areas for imputation
    list_of_possible_ranges = get_areas_for_impute_error(executor, imputation_length, sample_anomalies,
                                                         ignore_buffer)
    data_channels, shadow_channel_count, shadow_expected_value, time_features = get_shadow_channels_and_time_features(
        executor, samples, sample_anomalies)

    imputation_scores = []
    for r in range(min(max_areas, len(list_of_possible_ranges))):
        # get the range to test
        test_range_indexes = list_of_possible_ranges[r]
        # set up an updating history for the imputation
        current_past = None
        # sum individual differences for each sampled point
        collection_of_diffs = []
        for i in test_range_indexes:
            # Get the current sample history
            if current_past is None:
                current_past = sample_hist[i]
            else:
                # ensure that the shadow channels and time features are set to the expected values
                current_past[data_channels:] = sample_hist[i, data_channels:]

            # create new sample from the center with current self-regressive history
            sampled, _ = get_sample(executor, None, samples[i], current_past,
                                    sample_dates[i], sample_anomalies[i],
                                    shadow_channel_count, shadow_expected_value,
                                    time_features)
            # sampled = executor.sample(None, current_past, sample_dates[i])
            x_sampled = np.array(sampled[0])
            # create normed difference factor between actual sample and sampled
            sample_diff = diff_norm(samples[i] - x_sampled)
            collection_of_diffs.append(sample_diff)
            # update the history - self-regressive
            current_past = np.vstack([current_past[1:], x_sampled])

        # calculate the mean of the differences
        mean_diff = np.mean(collection_of_diffs)
        # calculate the standard deviation of the differences
        std_diff = np.std(collection_of_diffs)
        # store the imputation score
        imputation_scores.append({
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "range": test_range_indexes,
            "diff_norms": collection_of_diffs,
        })

    # calculate total mean error
    total_mean_error = np.mean([s["mean_diff"] for s in imputation_scores])
    # check if the imputation scores are within the epsilon
    within_epsilon = total_mean_error <= epsilon
    # return decision and details
    return {
        "within_epsilon": within_epsilon,
        "total_mean_error": total_mean_error,
        "imputation_scores": imputation_scores
    }


def get_sample(executor: FlowExecutor, number_of_samples: int, sample: np.ndarray, current_hist: np.ndarray,
               sample_date: np.ndarray, sample_anomaly: np.ndarray,
               shadow_channel_count: int = 0, shadow_expected_value: float = 0.0,
               time_features: int = 0, recurrent_level: int = 0):
    """
    Get a random sample or from center.

    :param executor: FlowExecutor to use for sampling.
    :param number_of_samples: Number of samples to generate or None for center sampling.
    :param sample: Current sample to replace as reference for filtering.
    :param current_hist: History of the sample to use for sampling.
    :param sample_date: Date of the sample to use for sampling.
    :param sample_anomaly: Anomaly flag of the sample to use for sampling.
    :param shadow_channel_count: Shadow channel count to use in the sampling.
    :param shadow_expected_value: Expected value of the shadow channels to use in the sampling.
    :param time_features: Number of time features to use in the sampling.
    :param recurrent_level: Recurrent level to limit the amount of try to find a valid datapoint.
    :return:
    """
    if number_of_samples is None:
        # create new sample from the center or random
        new_x_samples = executor.sample(None, current_hist, sample_date)
        gen_samples = 1
    else:
        # create new random sample based on a random sample from the executor
        new_x_samples = executor.sample(number_of_samples, current_hist, sample_date)
        gen_samples = number_of_samples

    # get latent representation of the sample
    _, z_sample = executor.predict_individual(None, new_x_samples[0],
                                              np.repeat([current_hist], gen_samples,
                                                        axis=0),
                                              np.repeat([sample_date], gen_samples,
                                                        axis=0),
                                              np.repeat([sample_anomaly], gen_samples,
                                                        axis=0))

    if shadow_channel_count > 0 and recurrent_level < 5:
        valid_sample_indexes = filter_samples_by_shadow_channel_epsilon(new_x_samples[0],
                                                                        shadow_channel_count,
                                                                        shadow_expected_value)
        if valid_sample_indexes.shape[0] == 0:
            logger.warning("No valid samples found based on shadow channels. Generating 100 extra random samples. "
                           f"Current recurrent level: {recurrent_level}")
            recurrent_level += 1
            return get_sample(executor, 100, sample, current_hist, sample_date, sample_anomaly,
                              shadow_channel_count, shadow_expected_value, time_features,
                              recurrent_level)
        new_x_samples = [s[valid_sample_indexes] for s in new_x_samples]
        # replace new_x_samples predicted shadow channels with the expected value
        new_x_samples[0][:, -shadow_channel_count:] = shadow_expected_value
        z_sample = z_sample[valid_sample_indexes]

    if time_features > 0 and recurrent_level < 5:
        # filter out samples that have time features that are not in the range of the original sample
        valid_sample_indexes = filter_samples_by_time_features(new_x_samples[0],
                                                               sample,
                                                               time_features)
        if valid_sample_indexes.shape[0] == 0:
            logger.warning("No valid samples found based on time features. Generating 100 extra random samples. "
                           f"Current recurrent level: {recurrent_level}")
            recurrent_level += 1
            return get_sample(executor, 100, sample, current_hist, sample_date, sample_anomaly,
                              shadow_channel_count, shadow_expected_value, time_features,
                              recurrent_level)
        new_x_samples = [s[valid_sample_indexes] for s in new_x_samples]
        # replace new_x_samples predicted time features with the original sample time features
        new_x_samples[0][:, -time_features:] = sample[-time_features:]
        z_sample = z_sample[valid_sample_indexes]

    if number_of_samples is not None and number_of_samples > 0 and recurrent_level > 0:
        # create mean sample from the generated samples
        new_mean_x_samples = [np.mean(new_x_samples[i], axis=0, keepdims=True) for i in range(len(new_x_samples))]
        new_mean_z_sample = np.mean(z_sample, axis=0, keepdims=True)
        return new_mean_x_samples, new_mean_z_sample
    else:
        return new_x_samples, z_sample


def get_shadow_channels_and_time_features(executor: Executor, samples: np.ndarray, sample_anomalies: np.ndarray):
    """
    Get the shadow channels and time features from the executor configuration.
    :param executor: Executor to get the configuration from.
    :param samples: Samples to check for shadow channels.
    :param sample_anomalies: Anomalies in the samples to check for shadow channels.
    :return:
    """
    full_config = executor.full_config
    sequence_channels = full_config["dataset"]["channels"]
    latent_space = full_config["model"]["code_configuration"].get("input_shape", (1, 4))[1]
    if full_config["dataset"].get("shadow_channels", False):
        shadow_channel_count = latent_space - sequence_channels
        first_valid_sample = np.where(sample_anomalies == 0)[0][0]
        shadow_expected_value = samples[first_valid_sample, -1]
    else:
        shadow_channel_count = 0
        shadow_expected_value = 0

    time_features = full_config["dataset"].get("number_of_time_features", 0)

    data_channels = sequence_channels - shadow_channel_count - time_features

    return data_channels, shadow_channel_count, shadow_expected_value, time_features


def calculate_shadow_error(executor: Executor, samples: np.ndarray, sample_hist: np.ndarray,
                           sample_dates: np.ndarray, sample_anomalies: np.ndarray,
                           sample_size: int = 10,
                           imputation_length: int = 50, max_areas: int = 5,
                           diff_norm: Callable = np.linalg.norm, epsilon: float = 0.2,
                           ignore_buffer: bool = False):
    # check if shadow channels are used
    if executor.full_config["dataset"].get("shadow_channels", False):
        # get areas for imputation
        list_of_possible_ranges = get_areas_for_impute_error(executor, imputation_length, sample_anomalies,
                                                             ignore_buffer)

        sequence_channels = executor.full_config["dataset"]["channels"]
        shadow_channel_count = samples.shape[1] - sequence_channels

        shadow_scores = []
        for r in range(min(max_areas, len(list_of_possible_ranges))):
            # get the range to test
            test_range_indexes = list_of_possible_ranges[r]
            # set up an updating history for the imputation
            current_past = sample_hist[test_range_indexes[0]]
            # sum individual differences for each sampled point
            shadow_of_diffs = []
            for i in test_range_indexes:
                sampled = executor.sample(sample_size, current_past, sample_dates[i])
                x_samples = np.array(sampled[0])
                # create mean sample
                mean_sample = np.mean(x_samples, axis=0)
                # create normed difference factor between actual sample and sampled
                shadow_diff = diff_norm(samples[i, -shadow_channel_count:] - mean_sample[-shadow_channel_count:])
                shadow_of_diffs.append(shadow_diff)
                # update the history
                current_past = np.vstack([current_past[1:], mean_sample])

            # calculate the mean of the differences
            mean_shadow_diff = np.mean(shadow_of_diffs)
            # calculate the standard deviation of the differences
            std_diff = np.std(shadow_of_diffs)
            # store the imputation score
            shadow_scores.append({
                "mean_diff": mean_shadow_diff,
                "std_diff": std_diff,
                "range": test_range_indexes,
                "diff_norms": shadow_of_diffs,
            })

        # calculate total mean error
        total_mean_error = np.mean([s["mean_diff"] for s in shadow_scores])
        # check if the imputation scores are within the epsilon
        within_epsilon = total_mean_error <= epsilon
        # return decision and details
        return {
            "within_epsilon": within_epsilon,
            "total_mean_error": total_mean_error,
            "shadow_scores": shadow_scores
        }
    else:
        return {
            "within_epsilon": True,
            "total_mean_error": 0,
            "shadow_scores": []
        }


def get_areas_for_impute_error(executor, imputation_length, sample_anomalies,
                               ignore_buffer: bool = False):
    # find areas in sample anomalies without anomaly (flag 0)
    anomaly_indexes = np.where(sample_anomalies == 0)[0]
    # find continuous ranges
    ranges = point_or_ranged_imputation(anomaly_indexes)
    # filter based on the length of the ranges
    ranges = [r for r in ranges if len(r) > imputation_length]
    if not ignore_buffer:
        ranges = [r[executor.full_config["dataset"]["max_past_range"]:] if r[0] > 0 else r
                  for r in ranges]
    # randomize the ranges
    np.random.shuffle(ranges)
    # calculate the number of possible sub ranges of imputation length in each range
    sub_ranges = [len(r) // imputation_length for r in ranges]
    # create clean ranges for imputation
    list_of_possible_ranges = []
    for r, number_of_sub_ranges in enumerate(sub_ranges):
        for i in range(0, number_of_sub_ranges, 2):
            start = ranges[r][0] + i * imputation_length
            end = start + imputation_length
            list_of_possible_ranges.append(list(range(start, end)))
    return list_of_possible_ranges


def calc_imputation_error(data: np.ndarray, imputed_data: np.ndarray):
    """
    Calculate the imputation error.

    :param data: The original data.
    :param imputed_data: The imputed data.
    :return: The imputation error.
    """
    mse = mean_squared_error(data, imputed_data)
    mae = mean_absolute_error(data, imputed_data)
    rmse = root_mean_squared_error(data, imputed_data)
    logger.info(f"Imputation guard check: MSE: {mse}, MAE: {mae}, RMSE: {rmse}")
    return {'mse': mse, 'mae': mae, 'rmse': rmse}


def get_guard_metrics(data: np.ndarray, nu_data: np.ndarray, parameters: dict, to_fix: dict):
    logger.info("Before anomaly area")
    end_of_first_guard = to_fix["to_fix_range"][0] - parameters["past"]
    metrics_first_guard = calc_imputation_error(data[:end_of_first_guard], nu_data[:end_of_first_guard])
    logger.info("In anomaly area")
    begin_of_second_guard = to_fix["to_fix_range"][1] - parameters["past"]
    metrics_second_guard = calc_imputation_error(data[begin_of_second_guard:], nu_data[begin_of_second_guard:])

    guard_metrics = {
        "guard_count": 2,
        0: metrics_first_guard,
        1: metrics_second_guard
    }

    return guard_metrics
