import os
import random
from abc import abstractmethod
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from loguru import logger
from torch.distributions import Normal, Independent
from torch.utils.data import Dataset
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

SEED = 42
TRAIN_TEST_SPLIT_SEED = SEED


def update_train_test_seed_value(seed: int):
    global TRAIN_TEST_SPLIT_SEED
    TRAIN_TEST_SPLIT_SEED = seed
    logger.warning(f"Train/Test seed value updated to {seed}")


# Set the seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

dtype = torch.float
DEVICE = None
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # PyTorch does not support MPS (Mx GPU) to 100% therefore run on CPU
    DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    if os.getenv("GPU") is not None:
        DEVICE = torch.device(f"cuda:{os.getenv('GPU', 1)}")
    else:
        DEVICE = torch.device("cuda:0")
    # Seed CUDA as well
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if DEVICE is None:
    DEVICE = torch.device("cpu")


def update_device(device):
    """
    Update the global DEVICE variable to the specified device.

    :param device: The device to set (e.g., 'cpu', 'cuda:0').
    :return: The updated device.
    """
    global DEVICE
    DEVICE = torch.device(device)
    logger.warning(f"Running with device (update): {DEVICE}")
    return DEVICE


class OpenDualDataProvider(Dataset):
    """
    Custom dataset provider for two outputs representing $x_t$ and $x_{t-k:t-1}$.
    """

    def __init__(self, data1, data2, data3, device):
        """
        Initialize the dataset with two data arrays.

        :param data1: First data array.
        :param data2: Second data array.
        """
        self.device = device
        self.data1 = torch.as_tensor(data1)
        self.data2 = torch.as_tensor(data2)
        self.data3 = torch.as_tensor(data3)

        self.move_each_batch = False

        if self.device.type == "cuda":
            # Check if the data fits into the GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            data_memory_need = (self.data1.nbytes + self.data2.nbytes + self.data3.nbytes) * 2.5
            if data_memory_need > total_memory:
                logger.warning(f"Data memory need {data_memory_need} exceeds total memory {total_memory}.")
                self.move_each_batch = True
            else:
                # logger.info(f"Data memory need {data_memory_need} fits into total memory {total_memory}.")
                self.data1 = self.data1.to(self.device)
                self.data2 = self.data2.to(self.device)
                self.data3 = self.data3.to(self.device)

    def __getitem__(self, item):
        """
        Get the data at the specified index.

        :param item: Index of the data to retrieve.
        :return: A tuple containing data1 and data2 at the specified index.
        """
        if self.move_each_batch:
            return (self.data1[item].to(self.device),
                    self.data2[item].to(self.device),
                    self.data3[item].to(self.device))
        else:
            return (self.data1[item],
                    self.data2[item],
                    self.data3[item])

    def __len__(self):
        """
        Get the length of the dataset.

        :return: The length of the dataset.
        """
        return self.data1.shape[0]


class Flow(nn.Module):
    """
    Default flow structure for normalizing flows with no past and passthrough past information.
    Base flow class.
    """

    def __init__(self, latent, bijections, full_config):
        """
        Initialize the Flow model.

        :param latent: Latent dimension.
        :param bijections: List of bijection layers.
        :param full_config: Full configuration dictionary.
        """
        super().__init__()
        self.full_config = full_config
        self.full_config["project"]["start_time"] = str(datetime.now())
        self.device = full_config["run"]["device"]
        self.latent = latent
        self.normal = Independent(Normal(
            loc=torch.zeros(self.latent, device=self.device),
            scale=torch.ones(self.latent, device=self.device),
        ), 1)
        self.bijections = nn.ModuleList(bijections)

        # setup input embedding
        self.dataset = full_config["dataset"]
        self.code_configuration = self.full_config["model"]["code_configuration"]

        self.input_embedding = self.code_configuration["input_embedding"]
        self.input_embedding_model = None

        if self.input_embedding != 'none':
            # prepare input embedding if operation is not excluded
            self.input_embedding_type = self.code_configuration["input_embedding_type"]
            self.tenc_layers = self.code_configuration.get("tenc_layers", 1)
            self.tenc_width = self.code_configuration.get("tenc_width", 1)
        else:
            # do not use input embedding at all
            self.input_embedding_type = 0
        self._setup_input_embedding()

    @property
    def base_dist(self):
        """
        Get the base distribution.

        :return: The base distribution.
        """
        return self.normal

    def _run_bijections(self, x, past):
        """
        Run the bijections on the input data.

        :param x: Input data.
        :param past: Past information.
        :return: Transformed data and log determinant.
        """
        log_det = torch.zeros(x.shape[0], device=self.device)
        for bijection in self.bijections:
            x, ldj = bijection(x, past)
            log_det += ldj
        return x, log_det

    def _run_sampling_bijections(self, num_samples, past):
        """
        Run the forward flow transformations.

        :param num_samples: Number of samples to generate.
        :param past: Past information is a 3D array.
        :return: Generated samples.
        """
        if num_samples is None:
            # if num_samples is None, we assume we want to sample one sample from the center
            z = torch.zeros((1, self.latent), dtype=dtype, device=self.device)
        elif torch.is_tensor(num_samples):
            # if num_samples is a tensor, we assume it is the points to sample from
            z = torch.as_tensor(num_samples, dtype=dtype, device=self.device)
        else:
            # if num_samples is an integer, we assume we want to sample num_samples samples
            z = self.base_dist.sample((num_samples,))

        # handle past
        if past is not None:
            # if past is a list or numpy array, convert it to a tensor
            past = torch.as_tensor(past).to(self.device)

        x = z
        probs = self.base_dist.log_prob(x)
        log_det = torch.zeros(x.shape[0], device=self.device)
        for bijection in reversed(self.bijections):
            x, ldj = bijection.inverse(x, past)
            log_det += ldj
        return x, probs, log_det

    def log_prob(self, x, past):
        """
        Calculate the log probability of the data.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - full log probability
                 - distribution log probability
                 - log determinant
        """
        u, log_det = self._run_bijections(x, past)
        dist_log_prob = self.base_dist.log_prob(u)
        full_log_prob = dist_log_prob + log_det
        return full_log_prob, dist_log_prob, log_det

    def individual_log_prob(self, x, past):
        """
        Calculate the individual log probability of the data.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        u, log_det = self._run_bijections(x, past)
        dist_log_prob = self.base_dist.base_dist.log_prob(u)
        return dist_log_prob, log_det, u

    def _extend_past_for_sampling(self, past: torch.Tensor, num_samples: int):
        if len(past.shape) < 3:
            # past is not a 3D tensor, we need to reshape it
            past = past[None]
        if past.shape[0] < num_samples:
            # past is not long enough, we need to repeat it
            past = torch.repeat_interleave(past, num_samples, dim=0)
        past = past.to(dtype)
        past = past.to(self.device)
        return past

    def sample(self, num_samples, past):
        """
        Sample data from the latent space.

        :param num_samples: Number of samples to generate.
        :param past: Past information.
        :return: Generated samples.
        """
        with torch.no_grad():
            x, probs, log_det = self._run_sampling_bijections(num_samples, past)
            nll_prob = -(probs + log_det)
            return x, nll_prob, probs, log_det

    def forward(self, x, past):
        """
        Forward processing for ONNX export.

        :param x: Input data.
        :param past: Past information.
        :return: Log probability of the data.
        """
        return self.log_prob(x, past)

    def loss(self, x, past):
        """
        Calculate the loss for the model.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - loss value (negative log probability)
                 - distribution log probability
                 - log determinant
        """
        full_log_prob, dist_log_prob, log_det = self.log_prob(x, past)
        return torch.mean(-full_log_prob), torch.mean(dist_log_prob), torch.mean(log_det)

    def _create_tenc_model(self, input_dim, output_dim):
        layers = [torch.nn.Linear(input_dim, self.tenc_width), torch.nn.ReLU()]
        for _ in range(self.tenc_layers):
            layers.append(torch.nn.Linear(self.tenc_width, self.tenc_width))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.tenc_width, output_dim))
        layers.append(torch.nn.Tanh())

        return torch.nn.Sequential(*layers)

    def _setup_input_embedding(self):
        if self.input_embedding_type == 1:
            # positional encoding, does not require parameters
            latent_dim = self.dataset["input_shape"][1]
            self.input_embedding_model = Summer(PositionalEncoding1D(latent_dim))

        elif self.input_embedding_type == 2:
            self.input_embedding_model = InputRandomizer()

        elif self.input_embedding_type == 3:
            # time embedding with a simple feedforward network and only feature size output
            latent_dim = self.dataset["input_shape"][1]
            # expecting sin + cos for month, day and hour
            self.input_embedding_model = self._create_tenc_model(6, latent_dim)

        elif self.input_embedding_type == 4:
            # time embedding with a simple feedforward network with a full sequence output
            latent_dim = self.dataset["input_shape"][1]
            total_length = self.code_configuration.get("past", 1) + 1
            output_size = latent_dim * total_length

            # expecting sin + cos for month, day and hour
            self.input_embedding_model = self._create_tenc_model(6, output_size)

        if self.input_embedding_type > 0:
            self.input_embedding_model = self.input_embedding_model.to(self.device)

    def run_time_embedding(self, x, past, date):

        if self.input_embedding_type == 1 or self.input_embedding_type == 2:
            # positional encoding
            x = x[:, None, :]
            full = torch.concat([past, x], dim=1)
            full_time_embedded = self.input_embedding_model(full)
            x = full_time_embedded[:, -1, :]
            past = full_time_embedded[:, :-1, :]

        elif self.input_embedding_type == 3:
            # time embedding with a simple feedforward network and only feature size output
            # timestamp to sin-day, cos-day, sin-month, cos-month, sin-year, cos-year
            time_embedding = self.input_embedding_model(date)
            x = torch.add(x, time_embedding)
            time_embedding_past = time_embedding[:, None, :]
            # repeat along the sequence dimension
            time_embedding_past = time_embedding_past.repeat(1, past.shape[1], 1)
            past = torch.add(past, time_embedding_past)

        elif self.input_embedding_type == 4:
            # time embedding with a simple feedforward network with a full sequence output
            # timestamp to sin-day, cos-day, sin-month, cos-month, sin-year, cos-year
            time_embedding = self.input_embedding_model(date)
            time_embedding_ = torch.reshape(time_embedding, (past.shape[0], past.shape[1] + 1, -1))
            x = torch.add(x, time_embedding_[:, -1, :])
            time_embedding_past = time_embedding_[:, :-1, :]
            past = torch.add(past, time_embedding_past)

        return x, past


class EncodedPastFlow(Flow):
    """
    Flow structure for normalizing flows with encoded past information.
    """

    def __init__(self, latent, past, past_encoder, bijections, full_config):
        """
        Initialize the EncodedPastFlow model.

        :param latent: Latent dimension.
        :param past: Past information dimension.
        :param past_encoder: List of past encoder layers.
        :param bijections: List of bijection layers.
        :param code_version: Version of the code.
        :param device: Device to run the model on.
        """
        super().__init__(latent, bijections, full_config)
        self.past = past
        self.past_encoder = nn.ModuleList(past_encoder)

    def _run_pre_encoding(self, x, past):
        """
        Run the past encoder on the past information.

        :param x: Input data.
        :param past: Past information.
        :return: Encoded past information.
        """
        num_samples = x.shape[0]
        encoded_past = past  # torch.reshape(past, (num_samples, self.latent, self.past))

        # Run the past encoder
        for encoder in self.past_encoder:
            encoded_past = encoder(encoded_past)

        encoded_past = torch.reshape(encoded_past, (num_samples, -1))
        return encoded_past

    def log_prob(self, x, past):
        """
        Calculate the log probability of the data with encoded past information.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - full log probability
                 - distribution log probability
                 - log determinant
        """
        encoded_past = self._run_pre_encoding(x, past)

        # Run normalizing flow transformations
        full_log_prob, dist_log_prob, log_det = super().log_prob(x, encoded_past)
        return full_log_prob, dist_log_prob, log_det

    def individual_log_prob(self, x, past):
        """
        Calculate the individual log probability of the data with encoded past information.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        encoded_past = self._run_pre_encoding(x, past)

        # Run normalizing flow transformations
        dist_log_prob, log_det, x = super().individual_log_prob(x, encoded_past)
        return dist_log_prob, log_det, x

    def sample(self, num_samples, past):
        """
        Sample data from the latent space with encoded past information.

        :param num_samples: Number of samples to generate.
        :param past: Past information.
        :return: Generated samples.
        """
        with torch.no_grad():
            past = torch.as_tensor(past, dtype=dtype, device=self.device)
            if num_samples is None:
                past = self._extend_past_for_sampling(past, 1)
            elif torch.is_tensor(num_samples):
                past = self._extend_past_for_sampling(past, num_samples.shape[0])
            else:
                past = self._extend_past_for_sampling(past, num_samples)

            encoded_past = past  # torch.reshape(past, (num_samples, self.latent, self.past))

            # Run the past encoder
            for encoder in self.past_encoder:
                encoded_past = encoder(encoded_past)

            if num_samples is None:
                encoded_past = torch.reshape(encoded_past, (1, -1))
            elif torch.is_tensor(num_samples):
                encoded_past = torch.reshape(encoded_past, (num_samples.shape[0], -1))
            else:
                encoded_past = torch.reshape(encoded_past, (num_samples, -1))

            return super().sample(num_samples, encoded_past)


# Implemented Bijections
class Reverse(nn.Module):
    """
    Reverse bijection layer for normalizing flows.
    """

    def setup(self, latent):
        """
        Setup method for the Reverse layer.

        :param latent: Latent dimension.
        """
        pass

    @staticmethod
    def forward(x, past):
        """
        Forward pass for the Reverse layer.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - Reversed input data.
                 - Log determinant (zeros).
        """
        return x.flip(dims=(1,)), x.new_zeros(x.shape[0])

    @staticmethod
    def inverse(z, past):
        """
        Inverse pass for the Reverse layer.

        :param z: Latent data.
        :param past: Past information.
        :return: A tuple containing:
                 - Reversed latent data.
                 - Log determinant (zeros).
        """
        return z.flip(dims=(1,)), z.new_zeros(z.shape[0])


class AbstractCoupling(nn.Module):
    """
    Abstract Coupling bijection layer for normalizing flows.
    """

    def __init__(self, net, code_configuration):
        """
        Initialize the AbstractCoupling layer.

        :param net: Neural network for the coupling layer.
        :param code_configuration: Configuration dictionary.
        """
        super().__init__()
        self.code_configuration = code_configuration
        self.net = net

        if self.code_configuration["coupling_stabilizer"] == "tanh":
            self.stabilizer = torch.tanh
        elif self.code_configuration["coupling_stabilizer"] == "tanh+bias":
            self.c_alpha_d_bias = nn.Parameter(torch.rand(net[-1].out_features // 2), requires_grad=True)
            self.stabilizer = self._run_full_stabilizer
        else:
            self.stabilizer = lambda alpha: alpha

    def _run_full_stabilizer(self, alpha):
        """
        Run the full stabilizer on the alpha values.

        :param alpha: Alpha values.
        :return: Stabilized alpha values.
        """
        return torch.tanh(alpha) * self.c_alpha_d_bias

    @abstractmethod
    def forward(self, x, past):
        raise NotImplementedError("Forward pass not implemented")

    @abstractmethod
    def inverse(self, z, past):
        raise NotImplementedError("Inverse pass not implemented")


class Coupling(AbstractCoupling):
    """
    Coupling bijection layer for normalizing flows.
    """

    def __init__(self, net, code_configuration):
        """
        Initialize the Coupling layer.

        :param net: Neural network for the coupling layer.
        :param code_configuration: Version of the code.
        """
        super().__init__(net, code_configuration)

    def forward(self, x, past):
        """
        Forward pass for the Coupling layer.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - Transformed data.
                 - Log determinant.
        """
        (z_d, x_D) = torch.chunk(x, 2, dim=1)
        mu_and_alpha = self.net(z_d)
        mu_d, alpha_d = torch.chunk(mu_and_alpha, 2, dim=1)

        # Stabilize the alpha_d values, prevent them from collapsing
        alpha_d = self.stabilizer(alpha_d)

        z_D = x_D * torch.exp(alpha_d) + mu_d
        z = torch.cat([z_d, z_D], dim=1)

        ldj = torch.sum(alpha_d, dim=1)
        return z, ldj

    def inverse(self, z, past):
        """
        Inverse pass for the Coupling layer.

        :param z: Latent data.
        :param past: Past information.
        :return: A tuple containing:
                 - Transformed data.
                 - Log determinant.
        """
        (x_d, z_D) = torch.chunk(z, 2, dim=-1)
        mu_and_alpha = self.net(x_d)
        mu_d, alpha_d = torch.chunk(mu_and_alpha, 2, dim=1)

        # Stabilize the alpha_d values, prevent them from collapsing
        alpha_d = self.stabilizer(alpha_d)

        x_D = (z_D - mu_d) * torch.exp(-alpha_d)
        x = torch.cat([x_d, x_D], dim=1)

        ldj = -torch.sum(-alpha_d, dim=1)
        return x, ldj


class ExtendedCoupling(AbstractCoupling):
    """
    Extended Coupling bijection layer for normalizing flows with past information.
    """

    def __init__(self, net, code_configuration):
        """
        Initialize the ExtendedCoupling layer.

        :param net: Neural network for the coupling layer.
        :param code_configuration: Version of the code.
        """
        super().__init__(net, code_configuration)

    def forward(self, x, past):
        """
        Forward pass for the ExtendedCoupling layer.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - Transformed data.
                 - Log determinant.
        """
        (z_d, x_D) = torch.chunk(x, 2, dim=1)
        full_in = torch.cat([z_d, past], dim=1)
        mu_and_alpha = self.net(full_in)
        mu_d, alpha_d = torch.chunk(mu_and_alpha, 2, dim=1)

        alpha_d = self.stabilizer(alpha_d)

        z_D = x_D * torch.exp(alpha_d) + mu_d
        z = torch.cat([z_d, z_D], dim=1)

        ldj = torch.sum(alpha_d, dim=1)
        return z, ldj

    def inverse(self, z, past):
        """
        Inverse pass for the ExtendedCoupling layer.

        :param z: Latent data.
        :param past: Past information.
        :return: A tuple containing:
                 - Transformed data.
                 - Log determinant.
        """
        (x_d, z_D) = torch.chunk(z, 2, dim=1)
        full_in = torch.cat([x_d, past], dim=1)
        mu_and_alpha = self.net(full_in)
        mu_d, alpha_d = torch.chunk(mu_and_alpha, 2, dim=1)

        alpha_d = self.stabilizer(alpha_d)

        x_D = (z_D - mu_d) * torch.exp(-alpha_d)
        x = torch.cat([x_d, x_D], dim=1)

        ldj = -torch.sum(-alpha_d, dim=1)
        return x, ldj


class STLinear(nn.Module):
    """
    Masked Linear to achieve a full ST spilt in one model
    Inspired by:
    https://github.com/zalandoresearch/pytorch-ts/blob/7860c9693d55b5c086867477cc33c89485ed0167/pts/modules/flows.py#L203
    """

    def __init__(self, in_features, out_features):
        super(STLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features

        mask = torch.full_like(self.linear.weight, 0, dtype=dtype)
        out_rows = out_features // 2
        in_cols = in_features // 2
        # set top left and the bottom right to 1 to allow data
        mask[:out_rows, :in_cols] = 1
        mask[-out_rows:, -in_cols:] = 1

        # Register the mask as a buffer so it’s not treated as a parameter
        self.register_buffer('mask', mask)

    def forward(self, x):
        # Apply the mask by element-wise multiplication
        masked_weight = self.linear.weight * self.mask
        return nn.functional.linear(x, masked_weight, self.linear.bias)


class BatchNorm(nn.Module):
    """
    BatchNorm
    Inspired by:
    https://github.com/zalandoresearch/pytorch-ts/blob/7860c9693d55b5c086867477cc33c89485ed0167/pts/modules/flows.py#L71
    """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.view(-1, x.shape[-1]).mean(0)
            self.batch_var = x.view(-1, x.shape[-1]).var(0, unbiased=False)

            # Update running mean and variance
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean * (1 - self.momentum)
            )
            self.running_var.mul_(self.momentum).add_(
                self.batch_var * (1 - self.momentum)
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize the input
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = torch.exp(self.log_gamma) * x_hat + self.beta

        # Calculate the log-determinant of the Jacobian as a scalar
        log_abs_det_jacobian = torch.sum(self.log_gamma - 0.5 * torch.log(var + self.eps))

        return y, log_abs_det_jacobian.expand(x.shape[0])

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Invert the transformation
        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        # Calculate the inverse log-determinant of the Jacobian as a scalar
        log_abs_det_jacobian = -torch.sum(self.log_gamma - 0.5 * torch.log(var + self.eps))

        return x, log_abs_det_jacobian.expand(y.shape[0])


class GeneralBatchNorm(nn.Module):
    """
    GeneralBatchNorm
    Inspired by:
    https://github.com/zalandoresearch/pytorch-ts/blob/7860c9693d55b5c086867477cc33c89485ed0167/pts/modules/flows.py#L71
    """

    def __init__(self, momentum=0.9, eps=1e-5):
        raise NotImplementedError("GeneralBatchNorm implementation not working as wanted")
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = torch.mean(x)
            self.batch_var = torch.var(x, unbiased=False)

            # Update running mean and variance
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean * (1 - self.momentum)
            )
            self.running_var.mul_(self.momentum).add_(
                self.batch_var * (1 - self.momentum)
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize the input
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = torch.exp(self.log_gamma) * x_hat + self.beta

        # Calculate the log-determinant of the Jacobian as a scalar
        log_abs_det_jacobian = torch.sum(self.log_gamma - 0.5 * torch.log(var + self.eps))

        return y, log_abs_det_jacobian.expand(x.shape[0])

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Invert the transformation
        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        # Calculate the inverse log-determinant of the Jacobian as a scalar
        log_abs_det_jacobian = -torch.sum(self.log_gamma - 0.5 * torch.log(var + self.eps))

        return x, log_abs_det_jacobian.expand(y.shape[0])


class ReversibleFlatten(nn.Module):
    """
    Reversible Flatten
    """

    def __init__(self, sequence_length, feature_size):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_size = feature_size

    @staticmethod
    def forward(x, past):
        batch_size, *shape = x.shape
        return x.view(batch_size, -1), x.new_zeros(x.shape[0])

    def inverse(self, x, past):
        return x.view(x.shape[0], self.sequence_length, self.feature_size), x.new_zeros(x.shape[0])


class InputRandomizer(nn.Module):
    """
    Input Randomizer
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            r_values = torch.randn_like(x[0]) * 0.05
            x = x + r_values
        return x

    @staticmethod
    def inverse(x):
        return x


if __name__ == "__main__":
    # Test some layers
    x = torch.randn(2, 10, 2)
    layer = ReversibleFlatten(10, 2)
    y, _ = layer(x, None)
    assert y.shape == (2, 20)
    z, _ = layer.inverse(y, None)
    assert z.shape == (2, 10, 2)
    print("Done")
