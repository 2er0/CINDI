import inspect
from types import GeneratorType, FunctionType

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm

from data_preprocessing import create_windowed_dataset_from_dict, create_windowed_dataset_from_dict_list, \
    create_dataset_from_dict_matrix, create_windowed_dataset, normalize


def load_data(generator=(), past=3):
    # get the dataset and prepare the training and testing data
    if isinstance(generator, GeneratorType) or isinstance(generator, tuple):
        _generator = generator
    elif isinstance(generator, FunctionType):
        # check if the generator is a function and call it with the needed arguments
        args = inspect.getfullargspec(generator).kwonlyargs
        kargs = {}
        if "past_size" in args:
            kargs["past_size"] = past
        _generator = generator(**kargs)
    else:
        _generator = generator

    if _generator[1].get("callback", False):
        logger.info("Callback function collected, disable callback for other uses")
        group, parameters, train_sequence, test_sequence, callback = _generator
        parameters["callback"] = False
    else:
        group, parameters, train_sequence, test_sequence = _generator
        callback = None
    return group, parameters.copy(), train_sequence, test_sequence, callback


def get_optimization_goal_sequence_and_goal(full_config, load_all_stored_datasets, normalize_factors):
    sequence_name = full_config["dataset"]["sequence_name"]
    # optimize the parameters for the model
    optimization_test_sequence = None
    logger.info(f"Checking for optimization test sequence on dataset '{full_config['dataset']['name']}'")
    if full_config["dataset"]["name"] in ["fsb", "srb"]:
        # Find the corresponding sequence pair for the optimization testing
        if "-no-anomaly" in sequence_name:
            find_supervised_training_sequence = sequence_name.replace("-no-anomaly", "")
            # use a balanced metric between auc and vus for the optimization guidance
            optimization_metric = "auc_vus_balance"
        else:
            find_supervised_training_sequence = f"{sequence_name}-no-anomaly"
            # use a balanced metric between the validation and evaluation losses for the optimization guidance

            if full_config["run"].get("use_only_checks", False):
                optimization_metric = "checks_only"
            elif full_config["run"].get("use_checks_in_optimization", False):
                optimization_metric = "losses_and_checks"
            else:
                optimization_metric = "losses"

        for option in tqdm(load_all_stored_datasets(full_config["dataset"]["name"]), desc="Searching for sequence"):
            if find_supervised_training_sequence == option[0] and sequence_name != option[0]:
                logger.info(f"Found validation sequence: {option[0]}")
                _, _, optimization_test_sequence, _, _ = load_data(option)
                break

        # prepare the test sequences for the optimization
        optimization_goal_sequences = create_all_test_sets(optimization_test_sequence, full_config,
                                                           normalize_factors)
        opt = next(optimization_goal_sequences)
        if isinstance(opt, tuple):
            _, opt_eval, opt_eval_hist, opt_eval_anomalies, opt_eval_dates = opt
        else:
            raise ValueError("No optimization test sequence found")
    else:
        logger.error("This case should not happen")
        logger.warning("Using best validation loss to guide the optimization")
        opt_eval, opt_eval_hist, opt_eval_dates, opt_eval_anomalies = None, None, None, None
        if full_config["run"].get("use_only_checks", False):
            optimization_metric = "checks_only"
        elif full_config["run"].get("use_checks_in_optimization", False):
            optimization_metric = "losses_and_checks"
        else:
            # use the validation loss as the optimization metric
            optimization_metric = "val_loss"

    return opt_eval, opt_eval_hist, opt_eval_dates, opt_eval_anomalies, optimization_metric


def reduce_data_for_debug(parameters, sequence, test=False):
    # if debug mode is enabled, reduce the dataset size
    if False and parameters["data_reduce"]:
        logger.debug("DATAREDUCE MODE")
        # reduce the dataset size for faster debugging
        if "construct" in parameters and parameters["construct"] in ["json", "jsonlist", "jsonmatrix"]:
            # if the dataset is json based, reduce the size via iterator
            train_sequence = {k: sequence[k] for k in list(sequence.keys())[:1000]}
            if test:
                for k in list(sequence.keys())[500:510]:
                    sequence[k]["is_anomaly"] = 1
        else:
            # if the dataset is pandas based, reduce the size via slicing
            sequence = sequence[:1000]
            if test:
                sequence.loc[500:510, 'is_anomaly'] = 1

        logger.warning("Reduced dataset size for debugging and reduced epochs")

    return sequence


def window_data(parameters, sequence):
    add_noise = True
    # create windowed dataset from the given training and test dataset
    if "construct" in parameters and parameters["construct"] == "json":
        # handle json based dataset for all models except LSTM
        samples, sample_hist, data_w_dates, anomalies = create_windowed_dataset_from_dict(
            sequence, parameters["max_past_range"], parameters["feature_names"]
        )
    elif "construct" in parameters and parameters["construct"] == "jsonlist":
        samples, sample_hist, data_w_dates, anomalies = create_windowed_dataset_from_dict_list(
            sequence, parameters["max_past_range"]
        )
    elif "construct" in parameters and parameters["construct"] == "jsonmatrix":
        samples, sample_hist, data_w_dates, anomalies = create_dataset_from_dict_matrix(
            sequence, parameters["max_past_range"]
        )
        parameters["max_past_range"] = sample_hist.shape[1]
        add_noise = False
    else:
        # handle normal tabular source input data by creating a windowed dataset
        samples, sample_hist, data_w_dates, anomalies = create_windowed_dataset(
            sequence, parameters["channels"], parameters["shadow_channels"], parameters.get("date_column", "timestamp"),
            parameters["max_past_range"]
        )

    # if the date column is in unix timestamp format, convert it to datetime (if past 2001, simple check)
    if data_w_dates[0] > 1000000000:
        # date_times = [datetime.datetime.fromtimestamp(d) for d in data_w_dates]
        df = pd.DataFrame(data_w_dates, columns=["date_utc"])
        df["date_utc"] = pd.to_datetime(df["date_utc"], unit="s")
        # df = add_time_features(df, add_time_full=True)
        # df.drop(columns=["date_utc"], inplace=True)
        # data_w_dates = df.to_numpy().astype(np.float32)
        data_w_dates = df.to_numpy()
        data_w_dates = np.squeeze(data_w_dates)

    return samples, sample_hist, data_w_dates, anomalies, add_noise


def add_time_features(df, month="sin", day="sin", day_of_week=None, hour="sin", add_time_full=False):
    logger.info("Adding time features")

    def sin_cos_transformer(period, data):
        sin_fn = FunctionTransformer(lambda v: np.sin(v / period * 2 * np.pi))
        cos_fn = FunctionTransformer(lambda v: np.cos(v / period * 2 * np.pi))
        return sin_fn.fit_transform(data), cos_fn.fit_transform(data)

    if month == "full" or add_time_full:
        df["sin_month"], df["cos_month"] = sin_cos_transformer(12, df["date_utc"].dt.month)
    elif month == "sin":
        df["sin_month"], _ = sin_cos_transformer(12, df["date_utc"].dt.month)

    if day == "full" or add_time_full:
        df["sin_day"], df["cos_day"] = sin_cos_transformer(31, df["date_utc"].dt.day)
    elif day == "sin":
        df["sin_day"], _ = sin_cos_transformer(31, df["date_utc"].dt.day)

    if day_of_week == "full" or add_time_full:
        df["sin_day"], df["cos_day"] = sin_cos_transformer(7, df["date_utc"].dt.dayofweek)
    elif day_of_week == "sin":
        df["sin_day"], _ = sin_cos_transformer(7, df["date_utc"].dt.dayofweek)

    if hour == "full" or add_time_full:
        df["sin_hour"], df["cos_hour"] = sin_cos_transformer(24, df["date_utc"].dt.hour)
    elif hour == "sin":
        df["sin_hour"], _ = sin_cos_transformer(24, df["date_utc"].dt.hour)

    return df


def prepare_train_data(full_config, train_sequence):
    logger.info("Prepare training data")
    parameters = full_config["dataset"]
    # reduce the dataset size for debugging
    train_sequence = reduce_data_for_debug(parameters, train_sequence)

    logger.info(f"Length of train_sequence: {len(train_sequence)}")

    # create windowed dataset for training and test data
    samples, sample_hist, data_w_dates, anomalies, add_noise = window_data(parameters, train_sequence)

    # normalize the dataset
    if "normalized" not in parameters or not parameters["normalized"]:
        samples, sample_hist, normalize_factors = normalize(samples, sample_hist)
    elif "normalize_factors" in parameters and len(parameters["normalize_factors"]) > 0:
        normalize_factors = parameters["normalize_factors"]
        samples, sample_hist, _ = normalize(samples, sample_hist, **normalize_factors)
    elif "normalize_factors" in parameters and len(parameters["normalize_factors"]) == 0:
        samples, sample_hist, normalize_factors = normalize(samples, sample_hist)
    else:
        normalize_factors = {}

    return samples, sample_hist, data_w_dates, anomalies, add_noise, normalize_factors


def create_all_test_sets(test_sequence, full_config, normalize_factors):
    logger.info("Prepare test data")
    # test_sequences = []
    if isinstance(test_sequence, FunctionType):
        test_iter = test_sequence()
    elif isinstance(test_sequence, list):
        test_iter = test_sequence
    elif isinstance(test_sequence, tuple):
        test_iter = [("0", test_sequence[1])]
    else:
        test_iter = [("0", test_sequence)]

    for test_name, test_data in tqdm(test_iter, desc="Preprocessing test data"):
        logger.info(f"Length of test_data: {len(test_data)}")
        test, test_hist, test_data_w_dates, anomalies = prepare_test_data(full_config["dataset"],
                                                                          full_config["model"]["model_type"],
                                                                          test_data,
                                                                          normalize_factors)
        yield test_name, test, test_hist, anomalies, test_data_w_dates


def prepare_test_data(parameters, model_type, test_sequence, normalize_factors):
    # reduce the dataset size for debugging
    test_sequence = reduce_data_for_debug(parameters, test_sequence, test=True)

    # create windowed dataset for training and test data
    test, test_hist, test_data_w_dates, anomalies, _ = window_data(parameters, test_sequence)

    if "normalized" not in parameters or not parameters["normalized"]:
        test, test_hist, _ = normalize(test, test_hist, **normalize_factors)
    elif "normalize_factors" in parameters:
        normalize_factors = parameters["normalize_factors"]
        test, test_hist, _ = normalize(test, test_hist, **normalize_factors)

    return test, test_hist, test_data_w_dates, anomalies
