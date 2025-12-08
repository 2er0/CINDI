import argparse
import inspect
import json
import os
import signal
import string
import sys
import traceback
from copy import deepcopy
from dataclasses import make_dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path, PosixPath
import random

import numpy as np
import torch
from loguru import logger

from models.flow_factory import model_types_parameters_default as model_types_parameters

global_config = {
    "code_configuration": {
        # base configuration for the code equals code version 3
        "coupling_stabilizer": "tanh+bias",  # "none", "tanh" or "tanh+bias" covers code version 1, 2 and 3
        "st_linear": False,  # False or True covers code version 4, 5, and 6
        "batch_norm": "none",  # "none", "BatchNorm" or "GlobalBatchNorm" covers code version 5 and 6
        # auto includes this parameter in the hyperparameter search
        "input_embedding": "none",
        # "auto" (0, 1, 2, 3), "none" (0), "positional_encoding" (1), or "time_embedding" (2, 3)
    }
}


def code_version_to_config(code_version):
    """
    Map the code version to the code configuration
    :param code_version: legacy code version to map to the code configuration
    :return: dictionary with the matching code configuration
    """
    base_config = global_config["code_configuration"]
    if code_version == 1:
        base_config["coupling_stabilizer"] = "tanh"
        base_config["batch_norm"] = "none"
        base_config["input_embedding"] = "none"
    elif code_version == 2:
        base_config["batch_norm"] = "none"
        base_config["input_embedding"] = "none"
    elif code_version == 4:
        base_config["st_linear"] = True
        base_config["batch_norm"] = "none"
        base_config["input_embedding"] = "none"
    elif code_version == 5:
        base_config["st_linear"] = True
        base_config["batch_norm"] = "BatchNorm"
        base_config["input_embedding"] = "none"
    elif code_version == 6:
        base_config["st_linear"] = False
        base_config["batch_norm"] = "GlobalBatchNorm"
        base_config["input_embedding"] = "none"
    return base_config


def parse_code_configuration(args):
    if args["code_configuration"] != "":
        logger.info(args["code_configuration"])
        # update the code configuration with the external code configuration
        try:
            code_config = global_config["code_configuration"]
            external_code_config = json.loads(args["code_configuration"])
            code_config.update(external_code_config)
        except json.JSONDecodeError:
            raise ValueError("Code configuration is not a valid JSON string")
    else:
        # use the global code configuration
        code_config = global_config["code_configuration"]
    return code_config


def flow_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, default="timeflow-base",
                        help="Name of the project tracking on a W&B server")
    parser.add_argument("--slurm_id", type=int, default=0,
                        help="Slurm job id")
    parser.add_argument("-e", "--experiment", type=str, default="opt_only",
                        help="Experiment to run")

    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--early_stopping", type=int, default=50,
                        help="Early stopping patience")

    parser.add_argument("--model_type", type=str, default="RealNVP",
                        help="Model type one or list of ['RealNVP', 'RealNVP-extended', …, 'tcNF-extended', …]")
    parser.add_argument("--code_configuration", type=str, default="",
                        help="Code configuration to run, new version of code_version")
    parser.add_argument("--max_past_range", type=int, default=3,
                        help="Number indicating the max size of a window")
    parser.add_argument("--shadow_channels", type=str, default="False",
                        help="Use shadow channels to support sampling in difficult cases")
    parser.add_argument("--sanity_check", type=str, default="False",
                        help="Run a sanity check for the model")
    parser.add_argument("--use_checks_in_optimization", type=str, default="False",
                        help="Use shadow and sanity checks in the optimization process")
    parser.add_argument("--use_only_checks", type=str, default="False",
                        help="Use shadow and sanity checks in the optimization process ONLY for model selection")

    parser.add_argument("--nruns", type=int, default=1,
                        help="Number of runs per configuration pair")
    parser.add_argument("--self_optimization", type=str, default="False",
                        help="Run self optimization for the model")
    parser.add_argument("--max_generations", type=int, default=10,  # 10,
                        help="Number of generations for the self optimization")
    parser.add_argument("--max_population", type=int, default=12,  # 12
                        help="Number of population for the self optimization")

    parser.add_argument("-d", "--dataset", type=str, default="fsb",
                        choices=["fsb", "srb", "real", "statnett",
                                 "aneo", "aneo_complex", "aneo_dynamic", "aneo_simple",
                                 "aneo_with_noise"],
                        help="Name name of the benchmark suite to generate: 'fsb' or 'srb'")
    parser.add_argument("--generator_seek", type=int, default=0,
                        help="Dataset generator seek to start with a different dataset")
    parser.add_argument("--generator_stop", type=int, default=1e10,
                        help="Dataset generator stop at a specific dataset")

    parser.add_argument("--load_pretrained", type=str, default="False",
                        help="Load pretrained model")
    parser.add_argument("--pretrained_model", type=str, default="",
                        help="Path to the pretrained model")

    args = parser.parse_args()
    args = args.__dict__
    logger.info(f"Parsed arguments: {args}")

    if str2bool(args["self_optimization"]):
        args["nruns"] = 1
    if str2bool(args["load_pretrained"]) and args["pretrained_model"] == "":
        raise ValueError("Pretrained model path is missing")

    if str2bool(args["load_pretrained"]):
        # restore the config from the pretrained model
        args = restore_config(args)
    else:
        args = create_general_config(args)

    return args


def create_general_config(args):
    code_configuration = parse_code_configuration(args)
    full_config = {
        "project": {"name": args["project"], "slurm_id": args["slurm_id"],
                    "experiment": args["experiment"]},
        "run": {"batch_size": args["batch_size"], "epochs": args["epochs"], "lr": args["lr"],
                "early_stopping": args["early_stopping"], "nruns": args["nruns"],
                "self_optimization": str2bool(args["self_optimization"]), "max_generations": args["max_generations"],
                "max_population": args["max_population"],
                "self_optimization_parallel": str2bool(os.getenv("SELF_OPTIMIZATION_PARALLEL", True)),
                "parallel_processes": int(os.getenv("PARALLEL_PROCESSES", 2)),
                "chunk_size": 300_000, "device": "cuda:0",
                "sanity_check": str2bool(args["sanity_check"]),
                "use_checks_in_optimization": str2bool(args["use_checks_in_optimization"]),
                "use_only_checks": str2bool(args["use_only_checks"])},
        "dataset": {"name": args["dataset"], "generator_seek": args["generator_seek"],
                    "generator_stop": args["generator_stop"], "max_past_range": args["max_past_range"],
                    "shadow_channels": str2bool(args["shadow_channels"]),
                    "input_shape": None, "hist_shape": None, "normalized": False,
                    "normalize_factors": {},
                    "data_reduce": str2bool(os.getenv("DATAREDUCE", False))},
        "model": {"model_type": args["model_type"],
                  # dynamic configuration for year model
                  "code_configuration": code_configuration},
        "pretrained": {"load_pretrained": str2bool(args["load_pretrained"]),
                       "pretrained_model": args["pretrained_model"]},
        "score": {"test": {}, "train": {}},
    }
    if full_config["dataset"]["data_reduce"]:
        full_config["run"]["epochs"] = 40
        full_config["run"]["max_generations"] = 2
        full_config["run"]["max_population"] = 2
    return full_config


def restore_config(args):
    logger.info("Restore config from pretrained model")

    model_folder = Path(args["pretrained_model"])
    if not model_folder.exists():
        raise ValueError(f"Model folder {model_folder} does not exist")

    config_file = model_folder / "config.json"
    if not config_file.exists():
        raise ValueError(f"Config file {config_file} does not exist")

    model_file = model_folder / "best_model.pth"
    if not model_file.exists():
        raise ValueError(f"Model file {model_file} does not exist")

    with open(config_file, "r") as f:
        args = json.load(f, object_hook=keys_to_numeric)

    args["pretrained"]["load_pretrained"] = True
    args["pretrained"]["pretrained_model_path"] = model_file

    return args


def args_parameter_merger(args, sequence_name, dataset_parameters):
    """
    Merge the arguments and the dataset parameters into one dictionary
    :param args:
    :param dataset_parameters:
    :return: merged dictionary
    """
    if args["pretrained"]["load_pretrained"]:
        logger.info("Nothing to merge, use the loaded config")
        return args

    args["dataset"]["sequence_name"] = sequence_name
    args["dataset"]["construct"] = dataset_parameters.get("construct", "table")
    args["dataset"]["date_column"] = dataset_parameters.get("date_column", "timestamp")
    args["dataset"]["number_of_time_features"] = dataset_parameters.get("number_of_time_features", 0)
    if "input_embedding" not in dataset_parameters:
        dataset_parameters["input_embedding"] = "positional_encoding"
    if args["model"]["code_configuration"].get("input_embedding", "") in ["auto", ""]:
        args["model"]["code_configuration"]["input_embedding"] = dataset_parameters.get("input_embedding",
                                                                                        "positional_encoding")
    args["dataset"]["feature_names"] = dataset_parameters.get("names", [])
    if dataset_parameters.get("shadow_channels", False):
        args["dataset"]["shadow_channels"] = dataset_parameters["shadow_channels"]
    args["dataset"]["channels"] = dataset_parameters["channels"]

    args["dataset"]["sequence_parameters"] = dataset_parameters
    return args


def format_exception(e):
    exception_list = traceback.format_stack()
    exception_list = exception_list[:-2]
    exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
    exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

    exception_str = "Traceback (most recent call last):\n"
    exception_str += "".join(exception_list)
    # Removing the last \n
    exception_str = exception_str[:-1]

    return exception_str


def data_guard(func):
    """
    Decorator to guard the data from unwanted changes.
    :param func:
    :return:
    """

    def inner(*args, **kwargs):
        args_ = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                args_.append(arg.copy())
            elif isinstance(arg, (list, dict)):
                args_.append(deepcopy(arg))
            else:
                args_.append(arg)

        args_ = tuple(args_)
        return func(*args_, **kwargs)

    return inner


def re_range(nll_prob):
    if np.min(nll_prob) == np.max(nll_prob):
        return nll_prob
    return (nll_prob - np.min(nll_prob)) / (np.max(nll_prob) - np.min(nll_prob))


def drop_time_features(data: np.ndarray, parameters: dict):
    if np.all(data[:, -1] == data[0, -1]):
        less = 1
    else:
        less = 0
    if "number_of_time_features" in parameters:
        if parameters["number_of_time_features"] + less > 0 and data.shape[1] > parameters[
            "number_of_time_features"] + less:
            data = data[:, :-(parameters["number_of_time_features"] + less)]
    return data


def generator_seek(gen, seek=0, stop=1e10, drop=False):
    for i, g in enumerate(gen):
        if drop:
            if "no-anomaly" not in g[0]:
                continue
        if seek <= i < stop:
            yield g


class Encoders(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, PosixPath):
            return str(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def keys_to_numeric(x):
    obj = {}
    for k, v in x.items():
        if k.lstrip('-').isdigit():
            obj[int(k)] = v
        elif k.strip().replace(".", "").isnumeric():
            obj[float(k)] = v
        else:
            obj[k] = v
    return obj


def save_config(config, filename):
    with open(filename, "w") as f:
        json.dump(config, f, indent=4, cls=Encoders)


def build_known_args_dict(func, params):
    specs = inspect.getfullargspec(func)
    build_args = {}
    for i, arg in enumerate(specs.args):
        if arg not in params:
            if len(specs.defaults) != len(specs.args):
                raise ValueError(f"Not enough default arguments")
            if specs.defaults[i] is not None:
                build_args[arg] = specs.defaults[i]
            else:
                raise ValueError(f"Argument {arg} is missing in the parameters")
        else:
            build_args[arg] = params[arg]
    return build_args


def handle_graceful_stop(t, p):
    if (datetime.now() - t).seconds > 2700:  # 45min run time max per process
        logger.warning(f"Terminating process {p.pid} gracefully")
        p.join(timeout=60)
        p.terminate()
    elif (datetime.now() - t).seconds > 3000:  # 30min run time max per process
        logger.error(f"Killing process {p.pid}")
        p.join(timeout=60)
        p.kill()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def global_random_sequence():
    s = string.ascii_lowercase + string.digits
    unique_identifier = ''.join(random.SystemRandom().sample(s, 10))
    return unique_identifier


def timeout(seconds, default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def signal_handler(signum, frame):
                raise TimeoutError("Timed out!")

            # Set up the signal handler for timeout
            signal.signal(signal.SIGALRM, signal_handler)

            if str2bool(os.getenv("DEBUG", False)):
                return func(*args, **kwargs)

            # Set the initial alarm for the integer part of seconds
            signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                return default
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator
