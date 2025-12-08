from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, List

import numpy as np
from loguru import logger

from global_utils import save_config, drop_time_features, timeout, global_random_sequence
from plot_utils import plot_roc_curve, plot_all_multiple_detection, plot_2d_latent_dist_space, plot_test_detection, \
    plot_before_after_impute
from vus.metrics import get_metrics
from vus.utils.slidingWindows import find_length


class Executor(ABC):
    """
    Abstract base class for an Executor that defines the interface for fitting and predicting.
    """

    def __init__(self, full_config):
        """
        Initialize the Executor with run arguments and parameters.

        :param run_args: Arguments for the run.
        :param parameters: Parameters for the Executor.
        """
        super().__init__()
        self.full_config = full_config
        self.model_type = self.full_config["model"]["model_type"]
        self.device = self.full_config["run"]["device"]

        if full_config["dataset"]["sequence_parameters"].get("print_name", None) is not None:
            self.sequence_name = full_config["dataset"]["sequence_parameters"]["print_name"]
        else:
            self.sequence_name = self.full_config["dataset"]["sequence_name"]
        self.offline_dir = None
        self.load_pretrained = self.full_config["pretrained"].get("load_pretrained", False)
        if self.load_pretrained:
            logger.info("Run with pretrained model")
            self.pretrained_model_path = self.full_config["pretrained"]["pretrained_model_path"]
            self.offline_dir = (self.full_config["model"]
                                .get("offline_dir",
                                     str(self.full_config["pretrained"]["pretrained_model_path"])
                                     .replace("/best_model.pth", "")))
        self.chunk_size = self.full_config["run"]["chunk_size"]

    def get_config(self):
        """
        Get the current configuration.

        :return: The current configuration as a dictionary.
        """
        return self.full_config

    def update_config(self, section, config):
        """
        Update the current configuration with new values.

        :param section: Section of the configuration to update.
        :param config: Dictionary containing new configuration values.
        """
        if section not in self.full_config:
            self.full_config[section] = config
        else:
            self.full_config[section].update(config)

    def save_config(self):
        """
        Save the current configuration to a file if offline directory is set.
        """
        if self.offline_dir is not None:
            save_config(self.get_config(), f"{self.offline_dir}/config.json")

    def start_logging(self, extra=None, random_overwrite=False):
        """
        Start logging to Weights and Biases (wandb) and locally to disk.
        """
        # prepare logging to wand and locally to disk
        run_store_path = f"./wandb/{self.full_config['project']['name']}"
        Path(run_store_path).mkdir(parents=True, exist_ok=True)

        if random_overwrite or self.full_config["project"].get("unique_identifier") is None:
            unique_identifier = global_random_sequence()
            self.full_config["project"]["unique_identifier"] = unique_identifier
        else:
            unique_identifier = self.full_config["project"]["unique_identifier"]

        self.offline_dir = (f"{run_store_path}/"
                            f"{self.full_config['model']['model_type']}#"
                            f"{self.full_config['dataset']['sequence_name']}#"
                            f"{'' if extra is None else extra}#"
                            f"{unique_identifier}")

        self.full_config["model"]["offline_dir"] = self.offline_dir
        Path(self.offline_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Logging to {self.offline_dir}")

        # save run configuration to local disk
        self.save_config()

    @abstractmethod
    def get_model(self):
        """
        Get the model instance.

        :return: The model instance.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Path):
        """
        Load the model from the provided path.

        :param model_path: Path to the model file.
        """
        pass

    @abstractmethod
    def fit(self, samples: np.ndarray, sample_hist: np.ndarray, sample_dates: np.ndarray,
            anomalies: Union[list, np.ndarray], choice_samples: int = 100) -> None:
        """
        Fit the model to the provided samples.

        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :param sample_dates: The dates corresponding to the samples as a numpy array.
        :param anomalies: The anomalies in the data, can be a list or numpy array.
        :param choice_samples: Number of samples to choose for fitting, Optional.
        """
        pass

    @abstractmethod
    def predict(self, test_name: Union[str, None], test: np.ndarray, test_hist: np.ndarray,
                test_dates: np.ndarray, anomalies: Union[list, np.ndarray], save_output_tensor: bool = True) -> (
            np.ndarray, Union[np.ndarray, List[np.ndarray]]):
        """
        Predict the outcomes for the provided test sequences.

        :param test_name: Name of the test.
        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param test_dates: Dates corresponding to the test data as a numpy array.
        :param anomalies: Anomalies in the test data, can be a list or numpy array.
        :param save_output_tensor:  If True, the output tensor will be saved to the offline directory.
        :return: A tuple containing:
                 - full negative log probability
                 - log probability
                 - log determinant
        """
        pass

    @abstractmethod
    def predict_individual(self, test_name: Union[str, None], test: np.ndarray, test_hist: np.ndarray,
                           test_dates: np.ndarray, anomalies: Union[list, np.ndarray]) -> (
            Union[np.ndarray, List[np.ndarray]],
            Union[np.ndarray, List[np.ndarray]],
    ):
        """
        Predict individual outcomes for visualization purposes.

        :param test_name: Name of the test.
        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param test_dates: Dates corresponding to the test data as a numpy array.
        :param anomalies: Anomalies in the test data, can be a list or numpy array.
        :return: A tuple containing:
                 - full log probability
                 - output
        """
        pass

    def score(self, nll_log: np.ndarray, is_anomaly: np.ndarray) -> dict:
        """
        Calculate the score based on negative log likelihood and anomaly status.

        :param nll_log: Negative log likelihood values.
        :param is_anomaly: Boolean array indicating anomaly status.
        :return: Dictionary containing scores and sliding window length.
        """
        estimated_sliding_window = find_length(is_anomaly)
        scores = get_metrics(nll_log, is_anomaly, metric='all', slidingWindow=estimated_sliding_window)
        scores["sliding_window"] = estimated_sliding_window
        return scores

    @timeout(120)
    def time_line_plot(self, name, train, test, all_probs_np, is_anomaly, nll_prob, train_dates,
                       test_dates, save_to_disk=True, show=False, title=None):

        """
        Create a timeline plot for the given data.

        :param name: Name of the plot.
        :param train: Training data.
        :param test: Test data.
        :param all_probs_np: All probabilities as a numpy array.
        :param is_anomaly: Boolean array indicating anomaly status.
        :param nll_prob: Negative log likelihood probabilities.
        :param train_dates: Dates corresponding to the training data.
        :param test_dates: Dates corresponding to the test data.
        :param save_to_disk: If True, the plot will be saved to disk.
        :param show: If True, the plot will be shown.
        :param title: Title of the plot.
        """
        cs = self.chunk_size
        for chunk in range(test.shape[0] // cs + 1):
            logger.info("Create time plot: {}".format(chunk))
            _train, _test = self._drop_time_features(train, test)
            _train = _train[chunk * cs:(chunk + 1) * cs]
            _test = _test[chunk * cs:(chunk + 1) * cs]
            _all_probs_np = all_probs_np[chunk * cs:(chunk + 1) * cs] if all_probs_np is not None else None
            _is_anomaly = is_anomaly[chunk * cs:(chunk + 1) * cs]
            _nll_prob = nll_prob[chunk * cs:(chunk + 1) * cs] if nll_prob is not None else None
            _test_dates = test_dates[chunk * cs:(chunk + 1) * cs]
            _train_dates = train_dates[chunk * cs:(chunk + 1) * cs]

            if title is None:
                title = f"{self.sequence_name}, Iteration: {name}, Model type: {self.model_type}"

            # create plot with continues result
            fig = plot_all_multiple_detection(
                _train,
                _test,
                _all_probs_np,
                (_is_anomaly, _nll_prob),
                _test_dates,
                _train_dates,
                title
            )
            if save_to_disk:
                img_byte = fig.to_image(format="png", width=2000, height=1200)
                with open(f"{self.offline_dir}/{name}_{chunk}_overview.png", "wb+") as destination:
                    destination.write(img_byte)
                # fig.write_html(f"{self.offline_dir}/{name}_{chunk}_overview.html")
            if show:
                fig.show()

    @timeout(120)
    def test_line_plot(self, name, train, test, all_probs_np, is_anomaly, nll_prob,
                       test_dates, save_to_disk=True, show=False, title=None):
        """
        Create a timeline plot for the given test data

        :param name: Name of the plot.
        :param train: Training data.
        :param test: Test data.
        :param all_probs_np: All probabilities as a numpy array.
        :param is_anomaly: Boolean array indicating anomaly status.
        :param nll_prob: Negative log likelihood probabilities.
        :param test_dates: Dates corresponding to the test data.
        :param save_to_disk: If True, the plot will be saved to disk.
        :param show: If True, the plot will be shown.
        :param title: Title of the plot.
        """
        cs = self.chunk_size
        for chunk in range(test.shape[0] // cs + 1):
            logger.info("Create time plot: {}".format(chunk))
            _, _test = self._drop_time_features(train, test)
            _test = _test[chunk * cs:(chunk + 1) * cs]
            _all_probs_np = all_probs_np[chunk * cs:(chunk + 1) * cs] if all_probs_np is not None else None
            _is_anomaly = is_anomaly[chunk * cs:(chunk + 1) * cs]
            _nll_prob = nll_prob[chunk * cs:(chunk + 1) * cs] if nll_prob is not None else None
            _test_dates = test_dates[chunk * cs:(chunk + 1) * cs]

            if title is None:
                title = f"{self.sequence_name}, Iteration: {name}, Model type: {self.model_type}"

            # create plot with continues result
            fig = plot_test_detection(
                _test,
                _all_probs_np,
                (_is_anomaly, _nll_prob),
                _test_dates,
                title
            )
            if save_to_disk:
                img_byte = fig.to_image(format="png", width=2000, height=1200)
                with open(f"{self.offline_dir}/{name}_{chunk}_test_overview.png", "wb+") as destination:
                    destination.write(img_byte)
                fig.write_html(f"{self.offline_dir}/{name}_{chunk}_test_overview.html")
                fig.write_image(f"{self.offline_dir}/{name}_{chunk}_test_overview.pdf", format="pdf",
                                width=2000, height=1200)
            if show:
                fig.show()

    @timeout(120)
    def before_after_time_line_plot(self, iteration,
                                    before_imputing_samples, nll_before,
                                    after_imputation_samples, nll_after,
                                    sample_anomalies, sample_dates,
                                    save_to_disk=True, show=False):
        """
        Create a timeline plot that compares the previous version of the sequences with the new version
        including both anomaly scores

        :param iteration: Iteration loop count
        :param before_imputing_samples: Samples before imputation steps
        :param nll_before: NLL of the samples before imputation steps
        :param after_imputation_samples: Samples after imputation steps
        :param nll_after: NLL of the samples after imputation steps
        :param sample_anomalies: Samples anomaly flagging
        :param sample_dates: Dates corresponding to the samples
        :param save_to_disk: If True, the plot will be saved to disk.
        :param show: If True, the plot will be shown.
        """

        title = f"{self.sequence_name}, Iteration: {iteration}, Model type: {self.model_type}"

        # create plot with continues result
        fig = plot_before_after_impute(
            before_imputing_samples, nll_before,
            after_imputation_samples, nll_after,
            sample_anomalies, sample_dates,
            title
        )
        if save_to_disk:
            img_byte = fig.to_image(format="png", width=2000, height=1200)
            with open(f"{self.offline_dir}/{iteration}_before_after.png", "wb+") as destination:
                destination.write(img_byte)
            fig.write_html(f"{self.offline_dir}/{iteration}_before_after.html")
            fig.write_image(f"{self.offline_dir}/{iteration}_before_after.pdf", format="pdf",
                            width=2000, height=1200)
        if show:
            fig.show()

    @timeout(180)
    def latent_space_plot(self, name, train, test, transformed_latent_space, individual_probs, sample_anomalies,
                          is_anomaly, title=None):
        """
        Create a latent space plot for the given data.

        :param name: Name of the plot.
        :param train: Training data.
        :param test: Test data.
        :param transformed_latent_space: Transformed latent space data.
        :param individual_probs: Individual probabilities.
        :param sample_anomalies: Anomalies in the sample data.
        :param is_anomaly: Boolean array indicating anomaly status.
        """

        cs = self.chunk_size
        for chunk in range(test.shape[0] // cs + 1):
            logger.info("Create time plot: {}".format(chunk))
            _is_anomaly = is_anomaly[chunk * cs:(chunk + 1) * cs]
            _sample_anomalies = sample_anomalies[chunk * cs:(chunk + 1) * cs]

            _train, _test = self._drop_time_features(train, test)
            _train = _train[chunk * cs:(chunk + 1) * cs]
            _test = _test[chunk * cs:(chunk + 1) * cs]
            _transformed_latent_space = transformed_latent_space[chunk * cs:(chunk + 1) * cs]
            _individual_probs = individual_probs[chunk * cs:(chunk + 1) * cs]

            title = f"{self.sequence_name}, Iteration: {name}, Model type: {self.model_type}"

            latent_fig = plot_2d_latent_dist_space(_train, _test, _transformed_latent_space,
                                                   _individual_probs, _sample_anomalies, _is_anomaly,
                                                   title)
            img_byte = latent_fig.to_image(format="png", width=2000, height=1200)
            with open(f"{self.offline_dir}/{name}_{chunk}_latent_space.png", "wb+") as destination:
                destination.write(img_byte)
            latent_fig.write_html(f"{self.offline_dir}/{name}_{chunk}_latent_space.html")
            latent_fig.write_image(f"{self.offline_dir}/{name}_{chunk}_latent_space.pdf", format="pdf",
                                   width=2000, height=1200)

    @timeout(60)
    def roc_curve_plot(self, name, prob, is_anomaly):
        """
        Create a ROC plot.

        :param name: Name of the plot.
        :param prob: Probabilities.
        :param is_anomaly: Boolean array in         dicating anomaly status.
        """
        cs = self.chunk_size
        for chunk in range(prob.shape[0] // cs + 1):
            logger.info("Create roc curve plot: {}".format(chunk))
            _is_anomaly = is_anomaly[chunk * cs:(chunk + 1) * cs]
            _prob = prob[chunk * cs:(chunk + 1) * cs]

            title = f"{self.sequence_name}, Iteration: {name}"

            fig = plot_roc_curve(_prob, _is_anomaly,
                                 title)
            img_byte = fig.to_image(format="png", width=500, height=500)
            with open(f"{self.offline_dir}/{name}_{chunk}_roc_curve.png", "wb+") as destination:
                destination.write(img_byte)
            fig.write_html(f"{self.offline_dir}/{name}_{chunk}_roc_curve.html")
            fig.write_image(f"{self.offline_dir}/{name}_{chunk}_roc_curve.pdf", format="pdf",
                            width=500, height=500)

    def _drop_time_features(self, train, test):
        _train = drop_time_features(train, self.full_config["dataset"])
        _test = drop_time_features(test, self.full_config["dataset"])
        return _train, _test
