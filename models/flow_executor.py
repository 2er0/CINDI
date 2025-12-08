from abc import abstractmethod, ABC
from collections.abc import Callable
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from data_preprocessing import create_dataset_with_past_from_dataset, create_dataset_from_dataset_with_end_as_validation
from global_utils import re_range, format_exception
from models.executor import Executor
from models.flow_factory import flow_factory
from models.real_nvp import OpenDualDataProvider


def checkpoint(model, filename):
    """
    Save the model state to a file.

    :param model: The model to save.
    :param filename: The filename to save the model state to.
    """
    torch.save(model.state_dict(), filename)


def resume(model, filename, device):
    """
    Load the model state from a file.

    :param model: The model to load the state into.
    :param filename: The filename to load the model state from.
    :param device: The device to load the model state into.
    """
    try:
        save_state = torch.load(filename, map_location=device)
        current_state = model.state_dict()
        model.load_state_dict(save_state)
    except RuntimeError as e:
        print(save_state.keys())
        print(current_state.keys())
        raise e


class FlowExecutor(Executor, ABC):
    """
    Executor class for flow-based models, inheriting from the Executor abstract base class.
    """

    def __init__(self, full_config):
        """
        Initialize the FlowExecutor with run arguments, parameters, and device.

        :param full_config: The full configuration dictionary.
        """
        super().__init__(full_config)

        self.lr = self.full_config["run"]["lr"]
        self.batch_size = self.full_config["run"]["batch_size"]
        self.epochs = self.full_config["run"]["epochs"]
        self.early_stop_thresh = self.full_config["run"]["early_stopping"]

        # request and build a flow from the factory
        self.flow = flow_factory(self.full_config).to(self.device)

        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr)

        if self.load_pretrained:
            logger.info(f"Loading pretrained model from {self.pretrained_model_path}")
            self.load_model(self.pretrained_model_path)

    def get_model(self):
        """
        Get the flow model instance.

        :return: The flow model instance.
        """
        return self.flow

    def load_model(self, model_path: Union[Path, str, None] = None):
        """
        Load the model from the provided path.

        :param model_path: The path to the model file.
        """
        if model_path is None:
            model_path = f"{self.offline_dir}/best_model.pth"
        resume(self.flow, model_path, self.device)

    def load_model_from_weights(self, model_weights: dict):
        """
        Load the model from the provided weights.

        :param model_weights: The weights of the model.
        """
        logger.info("Loading model from weights")
        self.flow.load_state_dict(model_weights)

    def save_model(self):
        """
        Save the model to the provided path.

        """
        checkpoint(self.flow, f"{self.offline_dir}/best_model.pth")

    def fit(self, samples: np.ndarray, sample_hist: np.ndarray, sample_dates: np.ndarray,
            anomalies: Union[list, np.ndarray], choice_samples: int = 1000) -> None:
        """
        Fit the flow model to the provided samples.

        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :param sample_dates: The dates corresponding to the samples as a numpy array.
        :param anomalies: The anomalies in the data, can be a list or numpy array.
        :param choice_samples: Number of samples to use for sampling error calculation.
        """
        samples, sample_hist, sample_dates, _ = self._pre_data_processing(samples, sample_hist, sample_dates)
        (train, train_hist, train_dates), (
            val, val_hist, val_dates, val_anomalies) = self._create_train_validation_split(samples,
                                                                                           sample_hist,
                                                                                           sample_dates,
                                                                                           anomalies)

        train_loader, train_batches = self.__create_loader(train, train_hist, train_dates)
        val_loader, val_batches = self.__create_loader(val, val_hist, val_dates)

        # prepare sampling error calculation
        normal_validation_points = np.where(val_anomalies == 0)[0]
        choice_samples = min(normal_validation_points.shape[0], choice_samples)
        sampling_val_loader, sampling_val_batches = self.__select_random_samples_for_sampling_error(
            normal_validation_points, val, val_hist, val_dates, choice_samples, self.batch_size * 10
        )

        best_val_loss = None
        best_epoch = None

        for epoch in range(self.epochs):
            # training
            self.flow.train()
            t_loss, t_log_prob, t_log_det = self._one_epoch(train_loader, train_batches)
            # validation
            self.flow.eval()
            v_loss_only, v_log_prob, v_log_det = self._one_eval_epoch(val_loader, val_batches)

            # calculate sampling error from a set of samples
            sample_mean = self.__sampling_error(sampling_val_loader)

            # calculate total validation loss
            # v_loss = 0.99 * sample_mean + 0.01 * v_loss_only
            v_loss = sample_mean

            s = ("{m} | Epoch: {e:04d} | Loss Train: {tl:.4f}, Val: {vl:.4f} | "
                 "Log Prob Train: {tlp:.4f}, Val: {vlp:.4f} | "
                 "Log Det Train: {tld:.4f}, Val: {vld:.4f}").format(sep=" ", m=self.model_type, e=epoch + 1,
                                                                    tl=t_loss, vl=v_loss,
                                                                    tlp=t_log_prob, vlp=v_log_prob,
                                                                    tld=t_log_det, vld=v_log_det)
            logger.info(s)

            # early stopping with 10 epochs grace period
            if epoch > 20:
                if best_val_loss is None:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    self.save_model()
                elif best_val_loss > v_loss:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    self.save_model()
                elif epoch - best_epoch > self.early_stop_thresh:
                    logger.info("Early stopped training at epoch %d" % (best_epoch + 1))
                    break  # terminate the training loop

        logger.info(f"Training finished, best model at epoch {best_epoch + 1}, loading best model")
        self.load_model()

    def fit_light(self, samples: np.ndarray, sample_hist: np.ndarray, sample_dates: np.ndarray,
                  anomalies: Union[list, np.ndarray], update_status_func=None, choice_samples: int = 1_000_000):
        """
        Fit the flow model to the provided samples with a lighter version of the training loop.
        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :param sample_dates: The dates corresponding to the samples as a numpy array.
        :param anomalies: The anomalies in the data, can be a list or numpy array.
        :param update_status_func: Function to update the status during training.
        :param choice_samples: Number of samples to use for sampling error calculation.
        :return:
        """
        samples, sample_hist, sample_dates, anomalies = self._pre_data_processing(samples, sample_hist,
                                                                                  sample_dates, anomalies)
        (train, train_hist, train_dates), (
            val, val_hist, val_dates, val_anomalies) = self._create_train_validation_split(samples,
                                                                                           sample_hist,
                                                                                           sample_dates,
                                                                                           anomalies)
        train_loader, train_batches = self.__create_loader(train, train_hist, train_dates)
        val_loader, val_batches = self.__create_loader(val, val_hist, val_dates)

        # prepare sampling error calculation
        normal_validation_points = np.where(val_anomalies == 0)[0]
        choice_samples = min(normal_validation_points.shape[0], choice_samples)
        sampling_val_loader, sampling_val_batches = self.__select_random_samples_for_sampling_error(
            normal_validation_points, val, val_hist, val_dates, choice_samples, self.batch_size
        )

        best_val_loss = None
        best_epoch = None
        best_weights = None
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # training
            self.flow.train()
            t_loss, t_log_prob, t_log_det = self._one_epoch(train_loader, train_batches)
            train_losses.append(t_loss)
            # validation
            self.flow.eval()
            v_loss_only, v_log_prob, v_log_det = self._one_eval_epoch(val_loader, val_batches)

            # calculate sampling error from a set of samples
            sample_mean = self.__sampling_error(sampling_val_loader)

            # calculate total validation loss
            # v_loss = 0.99 * sample_mean + 0.01 * v_loss_only
            v_loss = sample_mean
            val_losses.append(v_loss)

            if update_status_func is not None:
                update_status_func(f"{epoch + 1}/{self.epochs} | Val:{v_loss:.3f}: t{t_loss:.2f} v{v_loss_only:.2f}")

            # early stopping with 3 epochs grace period
            if True or epoch > 3:
                if best_val_loss is None:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    best_weights = self.flow.state_dict()
                elif best_val_loss > v_loss:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    best_weights = self.flow.state_dict()
                elif epoch - best_epoch > self.early_stop_thresh:
                    # logger.info("Early stopped training at epoch %d" % (best_epoch + 1))
                    break  # terminate the training loop

            if best_val_loss is not None and best_val_loss < -1000:
                logger.warning(f"Early stopped training at epoch {best_epoch + 1}")
                logger.warning(f"Losses | Train: {t_loss}, Val {v_loss}")
                logger.warning("Overfitted!!!")
                break

        return best_val_loss, epoch, best_epoch, best_weights, train_losses, val_losses

    def __select_random_samples_for_sampling_error(self, normal_validation_points, val, val_hist, val_dates,
                                                   choice_samples: int, batch_size: int = None):
        if normal_validation_points.shape[0] == choice_samples:
            sample_selection = normal_validation_points
        else:
            sample_selection = np.random.choice(normal_validation_points, choice_samples)
        sampling_val = val[sample_selection]
        sampling_val_hist = val_hist[sample_selection]
        sampling_val_dates = val_dates[sample_selection]
        if batch_size is None:
            batch_size = choice_samples
        sampling_val_loader, sampling_val_batches = self.__create_loader(sampling_val,
                                                                         sampling_val_hist,
                                                                         sampling_val_dates,
                                                                         batch_size=batch_size)
        return sampling_val_loader, sampling_val_batches

    def __predict_with_func(self, func=None, test=None, test_hist=None, test_dates=None) -> (
            List[Union[np.ndarray, List[np.ndarray]]]):
        """
        Predict using a specified function on the test data.

        :param func: The function to use for prediction.
        :param test: The test samples as a numpy array.
        :param test_hist: The historical data of the test samples as a numpy array.
        :param test_dates: The dates corresponding to the test samples as a numpy array.
        :return: List of prediction results.
        """
        self.flow.eval()
        self._pre_test_model_preparing()

        test_loader, test_batches = self.__create_loader(test, test_hist, test_dates)
        with torch.no_grad():
            batched_probs = []
            for i, (x, past, dates) in enumerate(test_loader):
                x, past = self.flow.run_time_embedding(x, past, dates)
                prob_outputs = func(x, past)
                batched_probs.append(prob_outputs)

        return batched_probs

    def predict(self, test_name: Union[str, None], test: np.ndarray, test_hist: np.ndarray, test_dates: np.ndarray,
                anomalies: Union[list, np.ndarray],
                save_output_tensor: bool = True) -> (
            np.ndarray, Union[np.ndarray, List[np.ndarray]]):
        """
        Predict the outcomes for the provided test sequences.

        :param test_name: Name of the test.
        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param test_dates: Dates corresponding to the test samples as a numpy array.
        :param anomalies: Anomalies in the test data, can be a list or numpy array.
        :param save_output_tensor: Save the output tensor.
        :return: A tuple containing:
                 - re-ranged negative log likelihood probability
                 - all probabilities as a numpy array
                    - negative log likelihood
                    - log probability
                    - log determinant
        """
        if test_name is not None:
            logger.info(f"Testing on {test_name} | Predict")
        test, test_hist, test_dates, _ = self._pre_data_processing(test, test_hist, test_dates)
        # get probs
        batched_probs = self.__predict_with_func(self._calc_log_prob, test, test_hist, test_dates)

        all_probs = []
        for prob in zip(*batched_probs):
            prob = torch.hstack(prob)
            prob = prob.cpu().numpy()
            all_probs.append(prob)

        # negative log likelihood
        all_probs[0] = -all_probs[0]
        # scale the negative log likelihood to a range between 0 and 1
        re_ranged_nll_prob = re_range(all_probs[0])

        # save probability scores (model output)
        all_probs_np = np.vstack(all_probs).T
        if save_output_tensor:
            with open(f"{self.offline_dir}/{test_name}_raw_all_prob.npy", "wb") as f:
                np.save(f, all_probs_np)

        return re_ranged_nll_prob, all_probs_np

    def predict_individual(self, test_name: str, test: np.ndarray, test_hist: np.ndarray, test_dates: np.ndarray,
                           anomalies: Union[list, np.ndarray]) -> (Union[np.ndarray, List[np.ndarray]],
                                                                   Union[np.ndarray, List[np.ndarray]]):
        """
        Predict individual outcomes for visualization purposes.

        :param test_name: Name of the test.
        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param test_dates: Dates corresponding to the test samples as a numpy array.
        :param anomalies: Anomalies in the test data, can be a list or numpy array.
        :return: A tuple containing:
                 - individual probabilities
                 - latent space representation
        """
        if test_name is not None:
            logger.info(f"Testing on {test_name} | Predict Individual")
        test, test_hist, test_dates, _ = self._pre_data_processing(test, test_hist, test_dates)
        batched_probs = self.__predict_with_func(self._calc_individual_log_prob, test, test_hist, test_dates)

        individual_probs_and_latent = []
        for prob in zip(*batched_probs):
            prob = torch.vstack(prob)
            prob = prob.cpu().numpy()
            individual_probs_and_latent.append(prob)

        # drop the determinant for visualization
        individual_probs = -(individual_probs_and_latent[0])  # + individual_probs_and_latent[1])

        return individual_probs, individual_probs_and_latent[2]

    def predict_with_grad(self, test: np.ndarray, test_hist: np.ndarray, test_dates: np.ndarray) -> (
            np.ndarray, Union[np.ndarray, List[np.ndarray]]):
        """
        Predict the outcomes for the provided test sequences.

        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param test_dates: Dates corresponding to the test samples as a numpy array.
        :return: A tuple containing:
                 - negative log likelihood probability
                 - input gradient
                 - past gradient
                 - date gradient
        """
        test, test_hist, test_dates, _ = self._pre_data_processing(test, test_hist, test_dates)
        # get probs
        self.flow.eval()
        self._pre_test_model_preparing()

        test_loader, test_batches = self.__create_loader(test, test_hist, test_dates, batch_size=1)
        batched_input_grad = []
        batched_past_grad = []
        batched_date_grad = []
        batched_probs = []
        for i, (x, past, date) in enumerate(test_loader):
            x.requires_grad_()
            past.requires_grad_()
            date.requires_grad_()
            x, past = self.flow.run_time_embedding(x, past, date)
            # prob_outputs = self._calc_individual_log_prob(x, past)
            prob_outputs = self._calc_log_prob(x, past)
            try:
                if prob_outputs[0].shape[0] > 1:
                    prob_outputs_mean = prob_outputs[0].mean()
                    prob_outputs_mean.backward()
                else:
                    prob_outputs[0].backward()
            except RuntimeError as e:
                logger.error(f"RuntimeError: An exception occurred: {e}")
                print(format_exception(e))
                raise e
            batched_input_grad.append(x.grad)
            if past.grad is not None:
                batched_past_grad.append(past.grad)
            if date.grad is not None:
                batched_date_grad.append(date.grad)
            batched_probs.append(prob_outputs[0])

        input_grad = torch.vstack(batched_input_grad)
        if len(batched_past_grad) > 0:
            past_grad = torch.vstack(batched_past_grad)
        if len(batched_date_grad) > 0:
            date_grad = torch.vstack(batched_date_grad)

        prob = torch.hstack(batched_probs)
        input_grad = input_grad.cpu().detach().numpy()
        if len(batched_past_grad) > 0:
            past_grad = past_grad.cpu().detach().numpy()
        else:
            past_grad = None
        if len(batched_date_grad) > 0:
            date_grad = date_grad.cpu().detach().numpy()
        else:
            date_grad = None
        prob = prob.cpu().detach().numpy()

        # negative log likelihood
        prob = -prob

        return prob, input_grad, past_grad, date_grad

    def sample(self, n: int = 128, past: np.ndarray = None, dates: np.ndarray = None) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample from the flow model.

        :param n: Number of samples to generate.
        :param past: Historical data.
        :param dates: Dates corresponding to the samples.
        :return: The generated samples.
        """
        if len(past.shape) == 2:
            past = past[-self.full_config["model"]["code_configuration"].get("past", 1):, :]
        elif len(past.shape) == 3:
            past = past[:, -self.full_config["model"]["code_configuration"].get("past", 1):, :]
        self.flow.eval()
        with torch.no_grad():
            x, full_prob, x_probs, log_det = self.flow.sample(n, past)

        x = x.cpu().numpy()
        full_prob = full_prob.cpu().numpy()
        x_probs = x_probs.cpu().numpy()
        log_det = log_det.cpu().numpy()
        return x, full_prob, x_probs, log_det

    @abstractmethod
    def _calc_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Abstract method to calculate the log probability.

        :param x: Input data.
        :param past: Historical data.
        :return: A tuple containing:
                 - log probability
                 - log determinant
                 - additional tensor
        """
        pass

    @abstractmethod
    def _calc_individual_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Abstract method to calculate the individual log probability.

        :param x: Input data.
        :param past: Historical data.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        pass

    def _pre_data_processing(self, samples: np.ndarray, sample_hist: np.ndarray,
                             sample_dates: np.ndarray, sample_anomalies: np.ndarray = None) -> (
            np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Pre-process before the training and validation split.

        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :param sample_dates: The dates corresponding to the samples as a numpy array.
        :param sample_anomalies: The anomalies in the data, can be a list or numpy array.
        :return: A tuple containing:
                 - processed samples
                 - processed historical samples
        """
        if self.full_config["model"]["code_configuration"].get("past", 1) != sample_hist.shape[1]:
            sample_hist = sample_hist[:, -self.full_config["model"]["code_configuration"].get("past", 1):, :]
        return samples, sample_hist, sample_dates, sample_anomalies

    @abstractmethod
    def _pre_test_model_preparing(self):
        """
        Abstract method to prepare the model for testing.
        """
        pass

    @abstractmethod
    def _create_train_validation_split(self, samples: np.ndarray, sample_hist: np.ndarray,
                                       sample_dates: np.ndarray, anomalies: np.ndarray = None) -> (
            (np.ndarray, np.ndarray, np.ndarray), (np.ndarray, np.ndarray, np.ndarray, np.ndarray)):
        """
        Abstract method to create the training and validation split.

        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :param sample_dates: The dates corresponding to the samples as a numpy array.
        :param anomalies: The anomalies in the data, can be a list or numpy array.
        :return: A tuple containing:
                 - training samples and historical samples
                 - validation samples and historical samples
        """
        pass

    @abstractmethod
    def _one_epoch(self, loader: DataLoader, batches: int):
        """
        Abstract method to perform one training epoch.

        :param loader: DataLoader for the training data.
        :param batches: Number of batches.
        """
        pass

    @abstractmethod
    def _one_eval_epoch(self, loader: DataLoader, batches: int):
        """
        Abstract method to perform one evaluation epoch.

        :param loader: DataLoader for the evaluation data.
        :param batches: Number of batches.
        """
        pass

    def __sampling_error(self, loader: DataLoader,
                         diff_norm: Callable = torch.linalg.norm) -> float:
        """
        Calculate the sampling error for the given samples.

        :param loader: DataLoader for the samples.
        :param diff_norm: The function to use for calculating the norm of the difference.
        :return: The sampling error as a float.
        """
        all_diff = None
        for samples, hist_samples, _ in loader:
            sample_size = samples.shape[0]
            # get a reconstruction error with random sampling
            sampled_random, nll_prob, probs, log_det = self.flow.sample(sample_size, hist_samples)
            # create normed difference factor between actual sample and sampled
            sample_diff = diff_norm(samples - sampled_random, dim=1)

            # get a reconstruction error with center points
            center_points = torch.zeros_like(samples, device=self.device)
            sampled_center, nll_prob_, probs, log_det = self.flow.sample(center_points, hist_samples)
            # calculate the difference between the sampled center points and the actual samples
            center_diff = diff_norm(samples - sampled_center, dim=1)

            # concat the differences
            if all_diff is None:
                all_diff = torch.cat((sample_diff, center_diff), dim=0)
            else:
                all_diff = torch.cat((all_diff, sample_diff, center_diff), dim=0)

        # calculate the mean of the differences
        mean_diff = torch.mean(all_diff).detach().cpu().item()

        return mean_diff

    def __create_loader(self, samples, hist_samples, sample_dates, batch_size=None) -> (
            DataLoader, int):
        """
        Create a DataLoader for the given samples and historical samples.

        :param samples: The input samples.
        :param hist_samples: The historical samples.
        :param sample_dates: The dates corresponding to the samples.
        :param batch_size: The batch size.
        :return: A tuple containing:
                 - DataLoader instance
                 - number of batches
        """
        provider = OpenDualDataProvider(
            samples.astype(np.float32),
            hist_samples.astype(np.float32),
            sample_dates.astype(np.float32),
            self.device
        )
        if batch_size is None:
            batch_size = self.batch_size
        loader = DataLoader(provider, batch_size=batch_size, shuffle=False)
        batches = len(loader)
        if batches == 0:
            raise ValueError("No training data available, past requirement is too high")

        return loader, batches


class FlowBatchedExecutor(FlowExecutor):
    """
    Executor class for flow-based models with batched processing, inheriting from FlowExecutor.
    """

    def __init__(self, full_config):
        super().__init__(full_config)

    def _pre_test_model_preparing(self):
        pass

    def _one_epoch(self, loader: DataLoader, batches: int) -> (float, float, float):
        """
        Perform one training epoch.

        :param loader: DataLoader for the training data.
        :param batches: Number of batches.
        :return: A tuple containing:
                 - average loss
                 - average log probability
                 - average log determinant
        """
        loss_sum = 0.0
        dist_log_prob_sum = 0.0
        log_det_sum = 0.0
        for i, (x, past, date) in enumerate(loader):
            self.optimizer.zero_grad()
            x, past = self.flow.run_time_embedding(x, past, date)
            loss, dist_log_prob, log_det = self.flow.loss(x, past)

            loss.backward()
            self.optimizer.step()

            loss_sum += loss.detach().cpu().item()
            dist_log_prob_sum += dist_log_prob.detach().cpu().item()
            log_det_sum += log_det.detach().cpu().item()

        return loss_sum / batches, dist_log_prob_sum / batches, log_det_sum / batches

    def _one_eval_epoch(self, loader: DataLoader, batches: int) -> (float, float, float):
        """
        Perform one evaluation epoch.

        :param loader: DataLoader for the evaluation data.
        :param batches: Number of batches.
        :return: A tuple containing:
                 - average loss
                 - average log probability
                 - average log determinant
        """
        loss_sum = 0.0
        dist_log_prob_sum = 0.0
        log_det_sum = 0.0
        with torch.no_grad():
            self.optimizer.zero_grad()
            for i, (x, past, date) in enumerate(loader):
                x, past = self.flow.run_time_embedding(x, past, date)
                loss, dist_log_prob, log_det = self.flow.loss(x, past)

                loss_sum += loss.detach().cpu().item()
                dist_log_prob_sum += dist_log_prob.detach().cpu().item()
                log_det_sum += log_det.detach().cpu().item()

        return loss_sum / batches, dist_log_prob_sum / batches, log_det_sum / batches

    def _calc_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.flow.log_prob(x, past)

    def _calc_individual_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        probs = self.flow.individual_log_prob(x, past)
        # un-squeeze the second tensor to match the shape of the first tensor
        probs[1].unsqueeze_(1)

        return probs[0], probs[1], probs[2]

    def _create_train_validation_split(self, samples, sample_hist, sample_dates, anomalies=None) -> (
            (np.ndarray, np.ndarray, np.ndarray), (np.ndarray, np.ndarray, np.ndarray, np.ndarray)):
        # create training and validation set for all other models
        # by splitting randomly picking sections as validation and add buffers around the validation sets
        try:
            (
                (train_samples, train_hist_samples, train_samples_dates),
                (validation_samples, validation_hist_samples, validation_samples_dates, validation_anomalies),
                (_, _)
            ) = create_dataset_with_past_from_dataset(samples, sample_hist, sample_dates, anomalies,
                                                      self.full_config["model"]["code_configuration"].get("past", 1))
            return ((train_samples, train_hist_samples, train_samples_dates),
                    (validation_samples, validation_hist_samples, validation_samples_dates, validation_anomalies))
        except IndexError as e:
            logger.error(f"IndexError: An exception occurred: {e}")
            print(format_exception(e))
            raise e


class FlowNonBatchedExecutor(FlowExecutor):
    """
    Executor class for flow-based models with non-batched (sequential) processing, inheriting from FlowExecutor.
    """

    def __init__(self, full_config):
        super().__init__(full_config)
        self.batch_size = 1  # run in non-batched mode - sequential training

    def _one_epoch(self, loader: DataLoader, batches: int) -> (float, float, float):
        """
        Perform one training epoch in non-batched mode.

        :param loader: DataLoader for the training data.
        :param batches: Number of batches.
        :return: A tuple containing:
                 - average loss
                 - average log probability
                 - average log determinant
        """
        loss_sum = 0.0
        dist_log_prob_sum = 0.0
        log_det_sum = 0.0
        # tcNF-stateful - requires no batching and therefore sequential training
        self.flow.reset_rnn_hidden()
        self.optimizer.zero_grad()
        loss_list = []

        for i, (x, past, date, _) in enumerate(loader):
            x, past = self.flow.run_time_embedding(x, past, date)
            loss, dist_log_prob, log_det = self.flow.loss(x, past)
            loss_list.append(loss)

            loss_sum += loss.detach().cpu().item()
            dist_log_prob_sum += dist_log_prob.detach().cpu().item()
            log_det_sum += log_det.detach().cpu().item()

            # back propagate every 256 steps
            if i > 0 and i % 256 == 0:
                loss_for_backward = torch.sum(torch.stack(loss_list)) / 256
                loss_for_backward.backward()
                self.flow.detach()
                self.optimizer.step()
                loss_list = []

        if len(loss_list) > 0:
            loss_for_backward = torch.sum(torch.stack(loss_list)) / len(loss_list)
            loss_for_backward.backward()
            self.flow.detach()
            self.optimizer.step()

        divider = batches  # batches * self.batch_size
        return loss_sum / divider, dist_log_prob_sum / divider, log_det_sum / divider

    def _one_eval_epoch(self, loader: DataLoader, batches: int) -> (float, float, float):
        """
        Perform one evaluation epoch in non-batched mode.

        :param loader: DataLoader for the evaluation data.
        :param batches: Number of batches.
        :return: A tuple containing:
                    - average loss
                    - average log probability
                    - average log determinant
        """
        loss_sum = 0.0
        dist_log_prob_sum = 0.0
        log_det_sum = 0.0
        with torch.no_grad():
            self.optimizer.zero_grad()
            self.flow.reset_rnn_hidden()
            for i, (x, past, date) in enumerate(loader):
                x, past = self.flow.run_time_embedding(x, past, date)
                loss, dist_log_prob, log_det = self.flow.loss(x, past)

                loss_sum += loss.detach().cpu().item()
                dist_log_prob_sum += dist_log_prob.detach().cpu().item()
                log_det_sum += log_det.detach().cpu().item()

        return loss_sum / batches, dist_log_prob_sum / batches, log_det_sum / batches

    def _pre_test_model_preparing(self):
        self.flow.reset_rnn_hidden()

    def _calc_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        all_probs = [self.flow.log_prob(s_t[None], s_h_t[None])
                     for (s_t, s_h_t) in zip(x, past)]
        all_probs = [torch.as_tensor(p) for p in zip(*all_probs)]
        return all_probs[0], all_probs[1], all_probs[2]

    def _calc_individual_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        probs = [self.flow.individual_log_prob(s_t[None], s_h_t[None])
                 for (s_t, s_h_t) in zip(x, past)]
        probs = [torch.vstack(p) for p in zip(*probs)]

        return probs[0], probs[1], probs[2]

    def _create_train_validation_split(self, samples: np.ndarray, sample_hist: np.ndarray,
                                       sample_dates: np.ndarray, anomalies: np.ndarray = None) -> (
            (np.ndarray, np.ndarray, np.ndarray), (np.ndarray, np.ndarray, np.ndarray, np.ndarray)):
        # handle json based dataset for LSTM
        # use the end as validation set
        (
            (train_samples, train_hist_samples, train_samples_dates),
            (validation_samples, validation_hist_samples, validation_samples_dates, validation_anomalies),
            (_, _)
        ) = create_dataset_from_dataset_with_end_as_validation(samples, sample_hist, sample_dates, anomalies,
                                                               self.full_config["model"]["code_configuration"].get(
                                                                   "past", 1),
                                                               noise=True)

        return ((train_samples, train_hist_samples, train_samples_dates),
                (validation_samples, validation_hist_samples, validation_samples_dates, validation_anomalies))
