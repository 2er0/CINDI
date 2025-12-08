import operator

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from loguru import logger

from global_utils import data_guard
from models.real_nvp import TRAIN_TEST_SPLIT_SEED


def normalize(samples: np.ndarray, samples_hist: np.ndarray,
              z_score: bool = False,
              col_sub: np.ndarray = None, col_divide: np.ndarray = None,
              col_hist_sub: np.ndarray = None, col_hist_divide: np.ndarray = None) -> (
        np.ndarray, np.ndarray, dict[str, np.ndarray]
):
    """
    Mean normalization per column/feature
    https://datagy.io/python-numpy-normalize/
    """
    if col_divide is None:
        # acquire normalization factors for the main input
        if z_score:
            # Z-score normalization
            col_mean = samples.mean(axis=0)
            col_std = samples.std(axis=0)
            col_global_mean = samples.mean()
            col_global_std = samples.std()
            col_divide = np.where(col_std == 0, col_global_std, col_std)
            col_sub = np.where(col_mean == 0, col_global_mean, col_mean)
        else:
            # use min-max normalization to -1 and 1
            col_max = samples.max(axis=0)
            col_min = samples.min(axis=0)
            glob_max = samples.max()
            col_max[col_max == 0] = glob_max
            # per calc the divider
            col_divide = col_max - col_min
            col_divide = np.where(col_divide == 0, col_max, col_divide)
            col_sub = col_min

    if col_hist_divide is None:
        # acquire normalization factors for the past
        if z_score:
            # Z-score normalization
            col_hist_mean = samples_hist.mean(axis=1).mean(axis=0)
            col_hist_std = samples_hist.std(axis=1).std(axis=0)
            col_hist_global_mean = samples_hist.mean()
            col_hist_global_std = samples_hist.std()
            col_hist_divide = np.where(col_hist_std == 0, col_hist_global_std, col_hist_std)
            col_hist_sub = np.where(col_hist_mean == 0, col_hist_global_mean, col_hist_mean)
        else:
            # use min-max normalization to -1 and 1
            col_hist_max = samples_hist.max(axis=1).max(axis=0)
            col_hist_min = samples_hist.min(axis=1).min(axis=0)
            glob_hist_max = samples_hist.max()
            col_hist_max[col_hist_max == 0] = glob_hist_max
            # per calc the divider
            col_hist_divide = col_hist_max - col_hist_min
            col_hist_divide = np.where(col_hist_divide == 0, col_hist_max, col_hist_divide)
            col_hist_sub = col_hist_min

    for i in range(samples.shape[1]):
        # normalize the main input
        if z_score:
            # Z-score normalization
            samples[:, i] = (samples[:, i] - col_sub[i]) / col_divide[i]
        else:
            # use min-max normalization to -1 and 1
            samples[:, i] = 2 * ((samples[:, i] - col_sub[i]) / col_divide[i]) - 1

    for i in range(samples_hist.shape[2]):
        # normalize the past
        if z_score:
            # Z-score normalization
            samples_hist[:, :, i] = (samples_hist[:, :, i] - col_hist_sub[i]) / col_hist_divide[i]
        else:
            # use min-max normalization to -1 and 1
            samples_hist[:, :, i] = 2 * ((samples_hist[:, :, i] - col_hist_sub[i]) / col_hist_divide[i]) - 1

    return samples, samples_hist, {"z_score": z_score,
                                   "col_sub": col_sub,
                                   "col_divide": col_divide,
                                   "col_hist_sub": col_hist_sub,
                                   "col_hist_divide": col_hist_divide}


def de_normalize(value, channel, normalize_factors):
    """
    De-normalize the value
    :param value: value to de-normalize
    :param channel: channel to de-normalize
    :param normalize_factors: normalization factor
    :return: de-normalized value
    """
    if normalize_factors["z_score"]:
        return value * normalize_factors["col_divide"][channel] + normalize_factors["col_sub"][channel]
    else:
        return (value + 1) * normalize_factors["col_divide"][channel] / 2 + normalize_factors["col_sub"][channel]


def extract_windows_vectorized(array: pd.DataFrame, sub_window_size: int):
    """
    Create a windowed array with indexes for the windows
    :param array: array to create windows
    :param sub_window_size: size of the window
    :return: array with indexes for the windows
    """
    sub_windows = (
        # expand_dims are used to convert a 1D array to 2D array.
            np.expand_dims(np.arange(sub_window_size), 0) +
            np.expand_dims(np.arange(array.shape[0] + 1 - sub_window_size), 0).T
    )
    return sub_windows


def _prepare_dataset(data: pd.DataFrame, window_size: int, base_features: list[str], date_column: str):
    """
    Prepare the data based on the window size and the date information
    :param data: dataframe
    :param window_size: size of the window
    :param base_features: list of features to consider
    :param date_column: column with the date information
    :return: slice information, windowed data, windowed dates
    """
    slices = extract_windows_vectorized(data, window_size)
    if slices.shape[0] > 1_000_000:
        logger.info("Reducing number of windows to save memory drastically")
        slices = slices[::3, :]
    elif slices.shape[0] > 400_000:
        logger.info("Reducing number of windows to save memory")
        slices = slices[::2, :]
    data_w = [[data.iloc[s][bf].to_numpy() for s in slices] for bf in [base_features]]  # 'High', 'Low',
    data_w = np.concatenate(data_w)
    data_w_dates = [[data.iloc[s][date_column].to_numpy() for s in slices] for _ in [base_features]]
    data_w_dates = np.concatenate(data_w_dates)
    data_w_anomaly = [[data.iloc[s]["is_anomaly"].to_numpy() for s in slices] for _ in [base_features]]
    data_w_anomaly = np.concatenate(data_w_anomaly)

    return slices, data_w, data_w_dates, data_w_anomaly


def _dataset_split(data_w, test_sections, follows, window_size):
    wsh = window_size // 2
    training_samples = np.asarray(range(data_w.shape[0]))
    selection = []
    for i in range(test_sections):
        s_start = np.random.choice(training_samples, size=1, replace=False)[0]
        s_end = s_start + follows
        while s_start not in training_samples and s_end not in training_samples:
            s_start -= 1
            s_end -= 1
            if s_start < 0:
                s_end = max(training_samples)
                s_start = s_end - follows
        s = list(range(s_start, s_end))
        selection.extend(s)
        training_samples = np.delete(training_samples, s)
        training_samples = np.delete(training_samples, range(s_start - wsh, s_start))
        training_samples = np.delete(training_samples, range(s_end, s_end + wsh))

    selection = np.asarray(selection)

    validation_samples = data_w[selection]
    data_w = np.delete(data_w, selection, axis=0)

    return data_w, validation_samples, selection


def create_windowed_dataset(data_: pd.DataFrame, channels: int, shadow_channels: bool, date_column: str, past: int = 3):
    """
    Create a windowed dataset
    :param data_: dataframe
    :param channels: list of feature channels
    :param shadow_channels: if True, add shadow channels
    :param date_column: column with the date information
    :param past: number of past information
    :return: windowed samples, windowed history samples, windowed dates
    """
    data = data_.copy()
    if (data.shape[-1]) % 2 == 1:
        # this acts as a fixing for add dimension and as a shadow channel
        data.insert(data.shape[1] - 2, f"value-{channels}", 0.5)
        channels += 1
    elif shadow_channels:
        # add two shadow channels
        data.insert(data.shape[1] - 1, f"value-{channels}", 0.5)
        data.insert(data.shape[1] - 1, f"value-{channels + 1}", 0.5)
        channels += 2

    feature_columns = [f"value-{i}" for i in range(channels)]
    if feature_columns[0] not in data.columns:
        feature_columns = data.columns.tolist()
        feature_columns.remove(date_column)
        feature_columns.remove("is_anomaly")

    # NOTE dropped the plus one on the past here to be more consistent
    slices, data_w, data_w_dates, data_w_anomaly = _prepare_dataset(data, past, feature_columns, date_column)

    samples = []
    sample_hist = []
    dates = []
    anomalies = []
    for s, d, a in zip(data_w, data_w_dates, data_w_anomaly):
        samples.append(s[-1])
        sample_hist.append(s[:-1])
        dates.append(d[-1])
        anomalies.append(a[-1])

    samples = np.stack(samples)
    sample_hist = np.stack(sample_hist)
    data_w_dates = np.asarray(dates)
    data_w_anomaly = np.asarray(anomalies)
    return samples, sample_hist, data_w_dates, data_w_anomaly


def create_dataset_with_past_from_dataset(samples: np.ndarray, sample_hist: np.ndarray,
                                          sample_dates: np.ndarray, anomalies: np.ndarray = None,
                                          past: int = 3) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray]
):
    """
    Create a dataset with past information
    :param samples: samples as rows of features
    :param sample_hist: history samples as rows of feature matrices
    :param sample_dates: dates to each sample
    :param anomalies: anomalies for each sample
    :param past: number of past information
    :return: windowed samples, windowed history samples, windowed dates
    """
    # samples, sample_hist, data_w_dates = create_windowed_dataset(data, channels, date_column, past)

    number_of_samples = samples.shape[0]
    number_of_valid_samples = int(number_of_samples * 0.2)
    number_of_samples_per_section = int(number_of_valid_samples // 5)

    all_indexes = np.array(list(range(samples.shape[0] - number_of_samples_per_section)))
    train_test = train_test_split(all_indexes, test_size=5, random_state=TRAIN_TEST_SPLIT_SEED)

    validation_sections_start = []
    for ss in sorted(train_test[1]):
        if len(validation_sections_start) == 0:
            validation_sections_start.append(ss)
        else:
            if ss - validation_sections_start[-1] >= number_of_samples_per_section:
                validation_sections_start.append(ss)
            else:
                validation_sections_start.append(validation_sections_start[-1] + number_of_samples_per_section)

    validation_selection = set()
    full_validation_selection_cleanup = set()
    for vss in validation_sections_start:
        validation_selection.update(range(vss, vss + number_of_samples_per_section))
        full_validation_selection_cleanup.update(range(vss - past, vss + number_of_samples_per_section + past))

    validation_selection = np.array(sorted(list(validation_selection)))
    full_validation_selection_cleanup = np.unique(np.array(sorted(list(full_validation_selection_cleanup))))
    full_validation_selection_cleanup = full_validation_selection_cleanup[0 <= full_validation_selection_cleanup]
    full_validation_selection_cleanup = full_validation_selection_cleanup[
        full_validation_selection_cleanup <= np.max(all_indexes)]

    all_indexes = np.array(list(range(samples.shape[0])))
    selection = np.delete(all_indexes, full_validation_selection_cleanup, axis=0)

    validation_selection = validation_selection[validation_selection < samples.shape[0]]

    train_samples = samples[selection]
    train_hist_samples = sample_hist[selection]
    train_sample_dates = sample_dates[selection]
    validation_samples = samples[validation_selection]
    validation_hist_samples = sample_hist[validation_selection]
    validation_sample_dates = sample_dates[validation_selection]
    if anomalies is not None:
        validation_anomalies = anomalies[validation_selection]
    else:
        validation_anomalies = None

    return ((train_samples, train_hist_samples, train_sample_dates),
            (validation_samples, validation_hist_samples, validation_sample_dates, validation_anomalies),
            (selection, validation_selection))


def prepend_zero(x, t, d):
    # prepend zero beginning
    x = np.concatenate([[t[0]], x], axis=0)
    t = np.concatenate([np.zeros((1, x.shape[1])), t], axis=0)
    t = np.expand_dims(t, axis=1)
    d.insert(0, d[0] - 1)
    return x, t, d


def prepend_noise(x, t, d, a):
    # prepend noise beginning
    prev = np.reshape(t[0], (1, t.shape[-1]))
    x = np.concatenate([prev, x], axis=0)
    rand_prev = np.random.normal(0, 0.1, (1, *t.shape[1:]))
    t = np.concatenate([rand_prev, t], axis=0)
    d_prev = d[0] - (d[1] - d[0])
    d = np.concatenate([[d_prev], d], axis=0)
    if a is not None:
        a = np.concatenate([[0], a], axis=0)
    else:
        a = None
    return x, t, d, a


def create_dataset_from_dataset_with_end_as_validation(samples: np.ndarray, sample_hist: np.ndarray,
                                                       sample_dates: np.ndarray, anomalies: np.ndarray = None,
                                                       past: int = 1, noise: bool = True) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray]
):
    """
    Create a dataset with the end as validation
    :param samples: samples as rows of features
    :param sample_hist: history samples as rows of feature matrices
    :param sample_dates: dates to each sample
    :param anomalies: anomalies for each sample
    :param past: number of past information
    :param noise: add noise to the samples
    :return: windowed samples, windowed history samples, windowed dates
    """
    number_of_samples = samples.shape[0]
    number_of_valid_samples = int(number_of_samples * 0.20)

    number_of_valid_samples_no_bleed = number_of_valid_samples + past

    train_samples = samples[:-number_of_valid_samples_no_bleed]
    train_hist_samples = sample_hist[:-number_of_valid_samples_no_bleed]
    train_sample_dates = sample_dates[:-number_of_valid_samples_no_bleed]

    validation_samples = samples[-number_of_valid_samples:]
    validation_hist_samples = sample_hist[-number_of_valid_samples:]
    validation_sample_dates = sample_dates[-number_of_valid_samples:]
    if anomalies is not None:
        validation_anomalies = anomalies[-number_of_valid_samples:]
    else:
        validation_anomalies = None

    if noise:
        samples, sample_hist, sample_dates, _ = prepend_noise(samples, sample_hist, sample_dates, None)
        train_samples, train_hist_samples, train_sample_dates, _ = prepend_noise(train_samples, train_hist_samples,
                                                                                 train_sample_dates, None)
        validation_samples, validation_hist_samples, validation_sample_dates, validation_anomalies = prepend_noise(
            validation_samples,
            validation_hist_samples,
            validation_sample_dates,
            validation_anomalies)

    selection = np.array(list(range(number_of_valid_samples_no_bleed)))
    validation_selection = np.array(list(range(number_of_valid_samples, samples.shape[0])))

    return ((train_samples, train_hist_samples, train_sample_dates),
            (validation_samples, validation_hist_samples, validation_sample_dates, validation_anomalies),
            (selection, validation_selection))


def create_windowed_dataset_from_dict(data: dict, past: int, features: list[str] = None) -> (
        np.ndarray, np.ndarray, list, list
):
    """
    Create windowed dataset from a dictionary
    :param data: dict storing the data
    :param past: number of past information
    :param features: list of feature names to consider
    :return: windowed samples, windowed history samples, windowed dates
    """
    # the minimal size is past + 1 for the current input value
    min_size = past + 1
    samples = []
    sample_hist = []
    anomalies = []
    data_w_dates = []
    for time, v in tqdm(sorted(data.items(), key=operator.itemgetter(0))):
        if len(v) == 0:
            continue

        values = []
        values_hist = []
        for name in features:
            row = v[name]
            if len(row) < min_size:
                values = None
                break
            else:
                # top contains current values
                values.append(row[0])
                values_hist.append(row[1:min_size])
        if not values:
            # skip if not enough data
            continue
        samples.append(np.asarray(values))
        sample_hist.append(np.asarray(values_hist))
        anomalies.append(v["is_anomaly"])
        data_w_dates.append(time)
    samples = np.asarray(samples)
    sample_hist = np.asarray(sample_hist)
    data_w_dates = np.asarray(data_w_dates)
    anomalies = np.asarray(anomalies)

    return samples, sample_hist, data_w_dates, anomalies


def create_windowed_dataset_from_dict_list(data: dict, past: int) -> (
        np.ndarray, np.ndarray, list, list
):
    """
    Create windowed dataset from a dictionary list
    :param data: dict storing the data
    :param past: number of past information
    :return: windowed samples, windowed history samples, windowed dates
    """
    # the minimal size is past + 1 for the current input value
    min_size = past + 1
    samples = []
    sample_hist = []
    anomalies = []
    data_w_dates = []

    first = data[list(data.keys())[0]]
    add_dimension = False
    if len(first["input"]) % 2 == 1:
        add_dimension = True

    for time, v in tqdm(sorted(data.items(), key=operator.itemgetter(0))):
        if len(v) == 0:
            continue
        if len(v["past"]) < min_size:
            continue
        value = v["input"]
        history = v["past"]
        history.append(value)
        # time is top to bottom which we reverse here
        history = np.asarray(list(reversed(history)))
        for i in range(history.shape[1]):
            if np.unique(history[:, i]).shape[0] == 1:
                continue
            # local window normalization per feature
            raise ValueError("Local window normalization per feature check if still needed")
            history[:, i] = (history[:, i] - history[:, i].min()) / history[:, i].ptp()

        if np.isnan(history).any():
            continue

        # add some noise
        history += np.random.normal(0, 0.1, history.shape)

        if add_dimension:
            history = np.c_[history, np.full((history.shape[0], 1), 0.5)]

        samples.append(history[0])
        sample_hist.append(history[1:min_size])
        anomalies.append(v["is_anomaly"])
        data_w_dates.append(time)

    samples = np.asarray(samples)
    sample_hist = np.asarray(sample_hist)
    data_w_dates = np.asarray(data_w_dates)
    anomalies = np.asarray(anomalies)

    return samples, sample_hist, data_w_dates, anomalies


def create_dataset_from_dict_matrix(data: dict, past: int) -> (
        np.ndarray, np.ndarray, list, list
):
    """
    Create windowed dataset from a dictionary list
    :param data: dict storing the data
    :param past: number of past information
    :return: windowed samples, windowed history samples, windowed dates
    """
    # the minimal size is past + 1 for the current input value
    samples = []
    sample_hist = []
    anomalies = []
    data_w_dates = []

    random_sample = data[list(data.keys())[0]]
    add_dimension = False
    if len(random_sample["input"]) % 2 == 1:
        add_dimension = True

    for time, v in tqdm(sorted(data.items(), key=operator.itemgetter(0))):
        if len(v) == 0:
            continue
        value = v["input"]
        if add_dimension:
            value = np.c_[value, np.full((value.shape[0], 1), 0.5 + np.random.normal(0, 0.1, 1))]

        history = v["past"]

        if np.isnan(history).any():
            logger.debug(f"NaN in history for {time}")
            raise ValueError(f"NaN in history for {time}")

        samples.append(value)
        sample_hist.append(history)
        anomalies.append(v["is_anomaly"])
        data_w_dates.append(time)

    samples = np.asarray(samples)
    sample_hist = np.asarray(sample_hist)
    data_w_dates = np.asarray(data_w_dates)
    anomalies = np.asarray(anomalies)

    return samples, sample_hist, data_w_dates, anomalies
