import copy
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial

import cma
import numpy as np
import pingouin as pg
from loguru import logger
from torch.multiprocessing import set_start_method, Process, Manager

from global_utils import format_exception, timeout
from models.executor import Executor
from models.executor_factory import ExecutorFactory
from models.flow_factory import model_types_parameters_meta
from models.flow_imputers import calculate_shadow_error, calculate_imputation_error
from models.real_nvp import update_train_test_seed_value

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class ParamOptimizer(ABC):

    def __init__(self, full_config, samples, sample_hist, sample_dates, sample_anomalies,
                 eval_data, eval_hist, eval_dates, eval_anomalies,
                 test, test_hist, test_dates, test_anomalies,
                 metric="val_loss"):
        super().__init__()
        self.full_config = full_config
        # store optimization data
        self.samples = samples
        self.sample_hist = sample_hist
        self.sample_dates = sample_dates
        self.sample_anomalies = sample_anomalies
        # store eval data
        self.eval = eval_data
        self.eval_hist = eval_hist
        self.eval_dates = eval_dates
        self.eval_anomalies = eval_anomalies
        # store test data
        self.test = test
        self.test_hist = test_hist
        self.test_dates = test_dates
        self.test_anomalies = test_anomalies
        # performance metric
        self.metric = metric
        self.metric_func = optimization_metric_funcs[metric]

    @abstractmethod
    def optimize(self):
        pass


class CMAParamOptimizer(ParamOptimizer):
    def __init__(self, full_config, samples, sample_hist, sample_dates, sample_anomalies,
                 eval_data, eval_hist, eval_dates, eval_anomalies,
                 test, test_hist, test_dates, test_anomalies,
                 metric="val_loss"):
        super().__init__(full_config, samples, sample_hist, sample_dates, sample_anomalies,
                         eval_data, eval_hist, eval_dates, eval_anomalies,
                         test, test_hist, test_dates, test_anomalies,
                         metric)
        dataset_name = full_config["dataset"]["name"]
        model_types_parameters = model_types_parameters_meta.get(dataset_name, model_types_parameters_meta["default"])
        self.params = model_types_parameters["default"]
        # for safety reasons, do not change the original config
        self.executor_base_params = copy.deepcopy(self.full_config)

        if self.full_config["model"]["code_configuration"].get("input_embedding", "none") != "none":
            embedding_type = self.full_config["model"]["code_configuration"]["input_embedding"]
            if embedding_type == "auto":
                self.params.update(model_types_parameters["input_embedding"])
            elif embedding_type == "positional_encoding":
                # allowing positional encoding and random nosing
                self.params["input_embedding_type"] = [0, 2]
        else:
            self.params["input_embedding_type"] = [0, 0]

        self.model_type = self.full_config["model"]["model_type"]

        self.params.update(model_types_parameters[self.model_type])
        self.param_order = list(self.params.keys())
        # max range for past
        self.params["past"] = [min(self.full_config["dataset"]["max_past_range"] - 20, self.params["past"][0]), self.full_config["dataset"]["max_past_range"]]

        # generation to evaluate and candidates to evaluate
        # if dataset_name == "aneo_with_noise" and self.full_config["run"]["max_generations"] > 2:
        #     self.max_iterations = 2
        #     self.candidates = 2
        # else:
        self.max_iterations = self.full_config["run"]["max_generations"]
        self.candidates = self.full_config["run"]["max_population"]

        self.parallel_processes = self.full_config["run"]["parallel_processes"]
        self.sigma = 0.5
        self.in_opts = {'bounds': [0, 1],
                        'popsize': self.candidates}
        self.init_params = [0.5] * len(self.param_order)
        self.best_config = None
        self.best_weights = None
        self.optimizer_trace = None

        # extend runtime for non FSB runs
        if "fsb" == self.full_config["dataset"]["name"]:
            self.run_one_func = _run_one_short  # 49min
        else:
            self.run_one_func = _run_one_long  # 1.5h

    def get_results(self):
        return self.best_config, self.optimizer_trace, pickle.loads(self.best_weights)

    def optimize(self):
        es = cma.CMAEvolutionStrategy(self.init_params, self.sigma, inopts=self.in_opts)

        optimization_trace = {"best_candidate": None,
                              "best_config": None,
                              "iterations": None}

        # make iterations configurable
        best_candidate = None
        best_candidate_value = np.inf
        best_candidate_weights = None
        iteration = 0
        while not es.stop() and iteration < self.max_iterations:
            logger.info(f"Starting optimization search iteration {iteration + 1}/{self.max_iterations} "
                        f"with {self.candidates} candidates and optimization target {self.metric}\n")
            candidates = es.ask()
            candidates_mapped = [self._map_params(x) for x in candidates]
            manager = Manager()
            result_dict = manager.dict()
            status_dict = manager.dict()
            candidates_pool_funcs = [partial(self.run_one_func,
                                             full_config=x,
                                             samples=self.samples,
                                             sample_hist=self.sample_hist,
                                             sample_dates=self.sample_dates,
                                             sample_anomalies=self.sample_anomalies,
                                             eval_data=self.eval,
                                             eval_hist=self.eval_hist,
                                             eval_dates=self.eval_dates,
                                             eval_anomalies=self.eval_anomalies,
                                             test=self.test,
                                             test_hist=self.test_hist,
                                             test_dates=self.test_dates,
                                             test_anomalies=self.test_anomalies,
                                             metric_func=self.metric_func,
                                             gen=iteration + 1,
                                             candidate=f"{iteration}.{i}",
                                             result_list=result_dict,
                                             status_dict=status_dict) for i, x in enumerate(candidates_mapped)]
            if not self.full_config["run"]["self_optimization_parallel"]:
                logger.warning("Running sequentially")
                # run sequentially
                for run_func in candidates_pool_funcs:
                    try:
                        run_func()
                    except ValueError as e:
                        logger.warning(f"Error occurred: {e}")
                        print(format_exception(e))
            else:
                # run in parallel on gpu
                wait_for = []
                for run_func in candidates_pool_funcs:
                    # run n processes in parallel
                    if len(wait_for) < self.parallel_processes:
                        p = Process(target=run_func)
                        p.start()
                        wait_for.append((datetime.now(), p))
                    else:
                        status = [p[1].is_alive() for p in wait_for]
                        while all(status):
                            time.sleep(5)
                            status = [p[1].is_alive() for p in wait_for]
                            # logger.info(status)
                            logger.info(status_dict)

                        wait_for = [p for p in wait_for if p[1].is_alive()]
                        p = Process(target=run_func)
                        p.start()
                        wait_for.append((datetime.now(), p))

                status = [p[1].is_alive() for p in wait_for]
                while any(status):
                    time.sleep(5)
                    status = [p[1].is_alive() for p in wait_for]
                    # logger.info(status)
                    logger.info(status_dict)

            # results are {"candidate": candidate,
            #              "opt_goal": optimization_loss,
            #              "opt_detail": optimization_details,
            #              "test_detail": test_details,
            #              "config": full_config,
            #              "model_weights": model_weights}
            result_list = [result_dict.get(f"{iteration}.{i}", {"candidate": f"{iteration}.{i}",
                                                                "opt_goal": 1000,
                                                                "opt_detail": None,
                                                                "test_detail": None,
                                                                "config": None,
                                                                "model_weights": None}) for i in range(len(candidates))]
            loss_only = [rl["opt_goal"] for rl in result_list]
            es.tell(candidates, loss_only)
            best_loss_in_iteration = np.min(loss_only)
            if best_loss_in_iteration < best_candidate_value:
                best_candidate_in_interation = np.where(loss_only == best_loss_in_iteration)[0]
                best_candidate = (iteration, best_candidate_in_interation[0])
                best_candidate_value = best_loss_in_iteration
                best_candidate_weights = result_list[best_candidate_in_interation[0]]["model_weights"]

            optimization_trace[iteration] = {}
            for i, x in enumerate(result_list):
                r = {k: result_list[i][k] for k in result_list[i].keys() if k not in ["model_weights", "config"]}
                r["config"] = copy.deepcopy({"model": result_list[i]["config"]["model"]})
                optimization_trace[iteration][i] = r
            iteration += 1

        optimization_trace["iterations"] = iteration
        if es.result.fbest == best_candidate_value:
            self.best_config = optimization_trace[best_candidate[0]][best_candidate[1]]["config"]
        else:
            raise ValueError("Best candidate not found in optimization trace")
        self.best_weights = best_candidate_weights
        logger.info(f"Best candidate: {best_candidate}, Best params: {self.best_config['model']['code_configuration']}")

        optimization_trace["best_config"] = self.best_config
        optimization_trace["best_candidate"] = best_candidate
        self.optimizer_trace = optimization_trace

    def _map_params(self, current_set):
        full_config = copy.deepcopy(self.executor_base_params)
        current_params = full_config["model"]["code_configuration"]
        for k, v in zip(self.param_order, current_set):
            to_range = self.params[k]
            if len(to_range) == 1:
                v_mapped = to_range[0]
            else:
                v_mapped = v * (to_range[1] - to_range[0]) + to_range[0]
                if isinstance(to_range[0], int):
                    v_mapped = np.round(v_mapped).astype(int)
                elif isinstance(to_range[0], float):
                    v_mapped = np.round(v_mapped * 10).astype(int) / 10
                else:
                    raise NotImplementedError("This type is not yet supported")
            current_params[k] = v_mapped

        full_config["model"]["code_configuration"] = current_params
        return full_config


def val_loss_metric(gen: int, executor: Executor, best_val_loss: float,
                    eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray,
                    eval_anomalies: np.ndarray) -> (float, str):
    return best_val_loss, {'values': {'best_val_loss': best_val_loss},
                           'print': f"best val_loss: {best_val_loss:.3f}"}


def score_metric(gen: int, executor: Executor, best_val_loss: float,
                 eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray, eval_anomalies: np.ndarray,
                 metric: str) -> (float, str):
    re_range_nll_prob, _ = executor.predict(None, eval_data, eval_hist, eval_dates, eval_anomalies,
                                            save_output_tensor=False)
    optimization_score = executor.score(re_range_nll_prob, eval_anomalies)[metric]
    return 1 - optimization_score, {'values': {f"{metric}_score": optimization_score},
                                    'print': f"1-({metric} score: {optimization_score:.3f})"}


auc_roc_metric = partial(score_metric, metric="AUC_ROC")
vus_roc_metric = partial(score_metric, metric="VUS_ROC")
r_auc_roc_metric = partial(score_metric, metric="R_AUC_ROC")
f_metric = partial(score_metric, metric="F")
rf_metric = partial(score_metric, metric="RF")


def score_balance_metric(gen: int, executor: Executor, best_val_loss: float,
                         eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray,
                         eval_anomalies: np.ndarray) -> (float, str):
    re_range_nll_prob, _ = executor.predict(None, eval_data, eval_hist, eval_dates, eval_anomalies,
                                            save_output_tensor=False)
    scores = executor.score(re_range_nll_prob, eval_anomalies)
    return ((1 - scores["AUC_ROC"]) * 0.3 + (1 - scores["VUS_ROC"]) * 0.7,
            {"values": {"AUC_ROC": scores["AUC_ROC"], "VUS_ROC": scores["VUS_ROC"]},
             "print": f"(1 - AUC_ROC: {scores['AUC_ROC']:.3f}) * 0.3 + (1 - VUS_ROC: {scores['VUS_ROC']:.3f}) * 0.7"})


score_factor = {0: (0.75, 0.25),
                1: (0.5, 0.5),
                2: (0.1, 0.8)}


def combined_metric(gen: int, executor: Executor, best_val_loss: float,
                    eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray, eval_anomalies: np.ndarray,
                    metric: str) -> (float, str):
    re_range_prob, _ = executor.predict(None, eval_data, eval_hist, eval_dates, eval_anomalies,
                                        save_output_tensor=False)
    optimization_score = executor.score(re_range_prob, eval_anomalies)[metric]
    factor = score_factor.get(gen // 4, (0.1, 0.8))
    # score = np.where(best_val_loss > 0, best_val_loss, best_val_loss * 0.01) - optimization_score * factor
    # return score, f" | leakyRelu(val_loss: {best_val_loss:.3f}) - {metric} score: {optimization_score:.3f} * {factor}"
    score = np.abs(best_val_loss) * factor[0] + 1 - optimization_score * factor[1]
    return (score, {"values": {"val_loss": best_val_loss, metric: optimization_score},
                    "print": (f"np.abs(val_loss: {best_val_loss:.3f}) * {factor[0]} + 1 - "
                              f"{metric} score: {optimization_score:.3f} * {factor[1]}")})


auc_roc_val_loss_metric = partial(combined_metric, metric="AUC_ROC")
vus_roc_val_loss_metric = partial(combined_metric, metric="VUS_ROC")
r_auc_roc_val_loss_metric = partial(combined_metric, metric="R_AUC_ROC")
f_val_loss_metric = partial(combined_metric, metric="F")
rf_val_loss_metric = partial(combined_metric, metric="RF")


def val_loss_and_scores(gen: int, executor: Executor, best_val_loss: float,
                        eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray,
                        eval_anomalies: np.ndarray) -> (float, str):
    re_range_nll_prob, _ = executor.predict(None, eval_data, eval_hist, eval_dates, eval_anomalies,
                                            save_output_tensor=False)
    scores = executor.score(re_range_nll_prob, eval_anomalies)
    if best_val_loss < 1:
        return ((1 - scores["AUC_ROC"]) * 0.3 + (1 - scores["VUS_ROC"]) * 0.7,
                {"values": {"AUC_ROC": scores["AUC_ROC"], "VUS_ROC": scores["VUS_ROC"]},
                 "print": f"(1 - AUC_ROC: {scores['AUC_ROC']:.3f}) * 0.3 + (1 - VUS_ROC: {scores['VUS_ROC']:.3f}) * 0.7"})
    else:
        return (best_val_loss * 0.1 + (1 - scores["AUC_ROC"]) * 0.3 + (1 - scores["VUS_ROC"]) * 0.6,
                {"values": {"val_loss": best_val_loss, "AUC_ROC": scores["AUC_ROC"], "VUS_ROC": scores["VUS_ROC"]},
                 "print": f"val_loss: {best_val_loss:.3f} * 0.1 + (1 - AUC_ROC: {scores['AUC_ROC']:.3f}) * 0.3 + "
                          f"(1 - VUS_ROC: {scores['VUS_ROC']:.3f}) * 0.6"})


def gaussian_convergence_metric(gen: int, executor: Executor, best_val_loss: float,
                                eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray,
                                eval_anomalies: np.ndarray) -> (float, str):
    _, latent = executor.predict_individual(None, eval_data, eval_hist, eval_dates, eval_anomalies)
    mardia_result = pg.multivariate_normality(latent, alpha=0.05)
    score = 1 - mardia_result.pval
    return score, {"values": {"p-value": mardia_result.pval},
                   "print": f"1 - p-value: {mardia_result.pval:.3f}"}


def losses_metric(gen: int, executor: Executor, best_val_loss: float,
                  eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray, eval_anomalies: np.ndarray) -> (
        float, str):
    _, probs = executor.predict(None, eval_data, eval_hist, eval_dates, eval_anomalies, save_output_tensor=False)
    eval_loss = np.mean(probs[:, 0])  # nll loss
    return (best_val_loss * 0.3 + eval_loss * 0.7,
            {"values": {"val_loss": best_val_loss, "eval_loss": eval_loss},
             "print": f"val_loss: {best_val_loss:.3f} * 0.3 + eval_loss: {eval_loss:.3f} * 0.7"})


def losses_and_checks_metric(gen: int, executor: Executor, best_val_loss: float,
                             eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray,
                             eval_anomalies: np.ndarray) -> (
        float, str):
    _, probs = executor.predict(None, eval_data, eval_hist, eval_dates, eval_anomalies, save_output_tensor=False)
    eval_loss = np.mean(probs[:, 0])  # nll loss

    if executor.full_config["dataset"].get("shadow_channels", False):
        shadow_check_analysis = calculate_shadow_error(executor, eval_data, eval_hist, eval_dates, eval_anomalies)
        shadow_score = shadow_check_analysis["total_mean_error"]
    else:
        shadow_score = 0
    if executor.full_config["run"].get("sanity_check", False):
        sanity_check_analysis = calculate_imputation_error(executor, eval_data, eval_hist, eval_dates, eval_anomalies)
        sanity_score = sanity_check_analysis["total_mean_error"]
    else:
        sanity_score = 0

    score = best_val_loss * 0.1 + eval_loss * 0.5 + shadow_score + sanity_score
    return (score,
            {"values": {"val_loss": best_val_loss, "eval_loss": eval_loss,
                        "shadow_score": shadow_score, "sanity_score": sanity_score},
             "print": f"val_loss: {best_val_loss:.3f} * 0.3 + eval_loss: {eval_loss:.3f} * 0.7 + "
                      f"shadow_score: {shadow_score:.3f} + sanity_score: {sanity_score:.3f}", })


def checks_only_metric(gen: int, executor: Executor, best_val_loss: float,
                       eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray,
                       eval_anomalies: np.ndarray) -> (
        float, str):
    if not executor.full_config["run"].get("sanity_check", False) or not executor.full_config["dataset"].get(
            "shadow_channels", False):
        raise ValueError("No checks to run, please set 'sanity_check' or 'shadow_channels' in the config")

    shadow_check_analysis = calculate_shadow_error(executor, eval_data, eval_hist, eval_dates, eval_anomalies)
    shadow_score = shadow_check_analysis["total_mean_error"]

    sanity_check_analysis = calculate_imputation_error(executor, eval_data, eval_hist, eval_dates, eval_anomalies)
    sanity_score = sanity_check_analysis["total_mean_error"]

    score = shadow_score + sanity_score
    return (score,
            {"values": {"shadow_score": shadow_score, "sanity_score": sanity_score},
             "print": f"shadow_score: {shadow_score:.3f} + sanity_score: {sanity_score:.3f}"})


def score_sanity_balance_metric(gen: int, executor: Executor, best_val_loss: float,
                                eval_data: np.ndarray, eval_hist: np.ndarray, eval_dates: np.ndarray,
                                eval_anomalies: np.ndarray) -> (float, str):
    re_range_nll_prob, _ = executor.predict(None, eval_data, eval_hist, eval_dates, eval_anomalies,
                                            save_output_tensor=False)
    scores = executor.score(re_range_nll_prob, eval_anomalies)

    # calculate a self-regressive reconstruction error over a set of windows of specific size
    sanity_check_analysis = calculate_imputation_error(executor, eval_data, eval_hist, eval_dates, eval_anomalies)
    sanity_score = sanity_check_analysis["total_mean_error"]

    return ((1 - scores["AUC_ROC"]) * 0.3 + (1 - scores["VUS_ROC"]) * 0.7 + sanity_score,  # * 0.5,  # sanity score is not multiplied by 0.5 anymore
            {"values": {"AUC_ROC": scores["AUC_ROC"], "VUS_ROC": scores["VUS_ROC"], "sanity_score": sanity_score},
             "print": f"(1 - AUC_ROC: {scores['AUC_ROC']:.3f}) * 0.3 + (1 - VUS_ROC: {scores['VUS_ROC']:.3f}) * 0.7"
                      f" + sanity_score: {sanity_score:.3f}"})


optimization_metric_funcs = {
    "val_loss": val_loss_metric,
    "auc_roc": auc_roc_metric,
    "vus_roc": vus_roc_metric,
    "r_auc_roc": r_auc_roc_metric,
    "f": f_metric,
    "rf": rf_metric,
    "auc_roc_val_loss": auc_roc_val_loss_metric,
    "vus_roc_val_loss": vus_roc_val_loss_metric,
    "r_auc_roc_val_loss": r_auc_roc_val_loss_metric,
    "f_val_loss": f_val_loss_metric,
    "rf_val_loss": rf_val_loss_metric,
    "auc_vus_balance": score_balance_metric,
    "val_loss_and_scores": val_loss_and_scores,
    "gaussian_convergence": gaussian_convergence_metric,
    "losses": losses_metric,
    "losses_and_checks": losses_and_checks_metric,
    "checks_only": checks_only_metric,
    "auc_vus_sanity_balance": score_sanity_balance_metric,
}


@timeout(2940)  # 49min timeout
def _run_one_short(*args, **kwargs):
    return _run_one_optimization(*args, **kwargs)


@timeout(5400)  # 1.5h timeout
def _run_one_long(*args, **kwargs):
    return _run_one_optimization(*args, **kwargs)


def _run_one_optimization(full_config, samples, sample_hist, sample_dates, sample_anomalies,
                          eval_data, eval_hist, eval_dates, eval_anomalies,
                          test, test_hist, test_dates, test_anomalies,
                          metric_func,
                          gen, candidate, result_list, status_dict=None):
    def update_status_dict(status):
        if status_dict is not None:
            status_dict[candidate] = status
        if not full_config["run"]["self_optimization_parallel"]:
            logger.info(f"Status {candidate}: {status}")

    # reduce the dataset size if it is too big
    samples, sample_hist, sample_dates, sample_anomalies = size_reducer(samples, sample_hist, sample_dates,
                                                                        sample_anomalies)

    if eval_data is None:
        eval_data = samples
        eval_hist = sample_hist
        eval_dates = sample_dates
        eval_anomalies = sample_anomalies

    # update hist data to fit the expected past
    sample_hist = sample_hist[:, -full_config["model"]["code_configuration"].get("past", 1):, :]
    eval_hist = eval_hist[:, -full_config["model"]["code_configuration"].get("past", 1):, :]
    test_hist = test_hist[:, -full_config["model"]["code_configuration"].get("past", 1):, :]

    # update the config with the current candidate and input shapes
    full_config["model"]["code_configuration"]["input_shape"] = samples.shape
    full_config["model"]["code_configuration"]["hist_shape"] = sample_hist.shape

    logger.info(f"Start candidate {candidate}: {full_config['model']}")
    update_train_test_seed_value(full_config["model"]["code_configuration"]["seed"])

    try:
        executor = ExecutorFactory.create_executor(full_config)
        best_val_loss, epoch, best_epoch, model_weights, train_losses, val_losses = executor.fit_light(samples,
                                                                                                       sample_hist,
                                                                                                       sample_dates,
                                                                                                       sample_anomalies,
                                                                                                       update_status_dict)
        optimization_loss, optimization_details = metric_func(gen, executor, best_val_loss, eval_data, eval_hist,
                                                              eval_dates,
                                                              eval_anomalies)

        optimization_details["epoch"] = epoch
        optimization_details["best_epoch"] = best_epoch
        optimization_details["train_losses"] = train_losses
        optimization_details["val_losses"] = val_losses

        # allways run auc & vus test on the test set but don't use it for optimization
        test_loss, test_details = score_balance_metric(gen, executor, best_val_loss, test, test_hist, test_dates,
                                                       test_anomalies)
        model_weights = pickle.dumps(model_weights)

        update_status_dict("Done")
    except Exception as e:
        logger.warning(f"Exception occurred: {e}")
        print(format_exception(e))
        update_status_dict("Error")

        optimization_loss = 1000
        model_weights = None
        optimization_details = {"print": "Exception occurred", "values": {}}
        test_loss = 1000
        test_details = {"print": "Exception occurred", "values": {}}
    logger.info(f"End candidate {candidate}")
    logger.info(f"{candidate} Opti Loss: {optimization_loss:.3f} | {optimization_details['print']}")
    logger.info(f"{candidate} Test loss: {test_loss:.3f} | {test_details['print']}")

    result_list[candidate] = {"candidate": candidate,
                              "opt_goal": optimization_loss,
                              "opt_detail": optimization_details,
                              "test_detail": test_details,
                              "config": full_config,
                              "model_weights": model_weights}


def size_reducer(data, data_hist, data_dates, data_anomalies, max_size: int = 100_000):
    if data is None:
        return data, data_hist, data_dates, data_anomalies
    if data.shape[0] > max_size:
        # stepping solution instead of reducing the amount of data
        # this should be a parameter and sequence depending
        stepping = data.shape[0] // max_size
        data = data[::stepping]
        data_hist = data_hist[::stepping]
        data_dates = data_dates[::stepping]
        data_anomalies = data_anomalies[::stepping]
    return data, data_hist, data_dates, data_anomalies
