from functools import reduce

import numpy as np
import torch

from models.real_nvp import Reverse, ExtendedCoupling, EncodedPastFlow, \
    STLinear, dtype, BatchNorm, GeneralBatchNorm

model_types_parameters_default = {
    "default": {"coupling_layers": [3, 20],
                "input_embedding_type": [0, 2],
                "seed": [30, 100]},

    "input_embedding": {"input_embedding_type": [0, 4],
                        "tenc_layers": [1, 8],
                        "tenc_width": [1, 100]},

    "tcNF-base": {"past": [1, 100],
                  "hidden_multiplier": [1, 50],
                  "st_net_layers": [1, 8],
                  "st_dropout": [0.1, 0.9],
                  "st_funnel_factor": [1, 10]},
    "tcNF-cnn": {"past": [10, 100],
                 "hidden_multiplier": [1, 50],
                 "st_net_layers": [1, 8],
                 "st_dropout": [0.1, 0.9],
                 "st_funnel_factor": [1, 10],
                 "encoder_layers": [1, 6],
                 "encoder_size": [3, 7],
                 "encoder_dropout": [0.1, 0.9],
                 "encoder_channel_depth": [1, 10]},
    "tcNF-mlp": {"past": [1, 100],
                 "hidden_multiplier": [1, 50],
                 "st_net_layers": [1, 8],
                 "st_dropout": [0.1, 0.9],
                 "st_funnel_factor": [1, 10],
                 "encoder_layers": [1, 6],
                 "encoder_dropout": [0.1, 0.9],
                 "encoder_compression_factor": [1, 20]},
}

model_types_parameters_aneo = {
    "default": {"coupling_layers": [8, 15],
                "input_embedding_type": [0, 2],
                "seed": [30, 100],
                "hidden_multiplier": [1, 50],
                "st_net_layers": [1, 8],
                "st_dropout": [0.1, 0.9],
                "st_funnel_factor": [1, 10]},

    "input_embedding": {"input_embedding_type": [0, 4],
                        "tenc_layers": [1, 8],
                        "tenc_width": [1, 100]},

    "tcNF-base": {"past": [100, 216]},
    "tcNF-cnn": {"past": [100, 216],
                 "encoder_layers": [1, 6],
                 "encoder_size": [3, 7],
                 "encoder_dropout": [0.1, 0.9],
                 "encoder_channel_depth": [1, 10]},
    "tcNF-mlp": {"past": [100, 216],
                 "encoder_layers": [1, 6],
                 "encoder_dropout": [0.1, 0.9],
                 "encoder_compression_factor": [1, 20]}
}

model_types_parameters_meta = {"default": model_types_parameters_default,
                               "aneo_with_noise": model_types_parameters_aneo, }


def make_conditioner(feature_dim=2, add_dim=0, st_net_layers=3, hidden_multiplier=10, st_dropout=0.4,
                     st_funnel_factor=1,
                     code_configuration=None):
    # ignoring this in the parameter optimization
    io_size = feature_dim // 2

    st_layer_sizes = np.linspace(10 * hidden_multiplier,
                                 int(10 * hidden_multiplier // st_funnel_factor),
                                 st_net_layers - 1).astype(int)

    layers = [torch.nn.Linear(io_size + add_dim, 10 * hidden_multiplier),
              torch.nn.Tanh(),
              torch.nn.Dropout(st_dropout), ]

    current_size = st_layer_sizes[0]
    for next_size in st_layer_sizes[1:]:
        if next_size % 2 == 1:
            # check that next size is even to have a nice split in the STLiner mask
            next_size += 1
        if code_configuration.get("st_linear", False):
            layers.append(STLinear(current_size, next_size))
        else:
            layers.append(torch.nn.Linear(current_size, next_size))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Dropout(st_dropout))
        current_size = next_size

    if code_configuration.get("st_linear", False):
        layers.append(STLinear(current_size, feature_dim))
    else:
        layers.append(torch.nn.Linear(current_size, feature_dim))

    return torch.nn.Sequential(*layers)


# helper for creating the identity network to pass past information directly to the next steps
def identity():
    return torch.nn.Sequential(
        torch.nn.Identity()
    )


class Permute(torch.nn.Module):

    def __init__(self, pattern, *args, **kwargs):
        super(Permute, self).__init__(*args, **kwargs)
        self.pattern = pattern

    def forward(self, x):
        x = x.permute(self.pattern)
        return x


# helper for creating the CNN-encoder network for the past information encoding
def cnn(channels=2, code_configuration=None):
    encoder_layers = code_configuration["encoder_layers"]
    encoder_dropout = code_configuration["encoder_dropout"]
    encoder_channel_depth = code_configuration["encoder_channel_depth"]
    encoder_size = code_configuration["encoder_size"]

    layers = [Permute((0, 2, 1))]
    if encoder_size % 2 == 0:
        encoder_size += 1

    padding = encoder_size // 2

    out_channels = np.linspace(channels, encoder_channel_depth, encoder_layers).astype(int)

    prev_out_channels = channels
    for i in range(encoder_layers):
        layers.append(torch.nn.Conv1d(
            in_channels=prev_out_channels,
            out_channels=int(out_channels[i]),
            kernel_size=encoder_size,
            stride=1,
            padding=padding,

        ))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout1d(encoder_dropout))
        prev_out_channels = int(out_channels[i])

    layers = layers[:-2]

    return torch.nn.Sequential(*layers)


def mlp(add_dims=(20, 8), code_configuration=None):
    encoder_layers = code_configuration["encoder_layers"]
    encoder_dropout = code_configuration["encoder_dropout"]
    encoder_compression_factor = code_configuration["encoder_compression_factor"]

    full_in = reduce(lambda x_, y_: x_ * y_, add_dims)
    add = full_in // encoder_compression_factor
    comp = np.linspace(full_in, add, encoder_layers + 1).astype(int)

    layers = [torch.nn.Flatten()]
    for i in range(encoder_layers):
        layers.append(torch.nn.Linear(comp[i], comp[i + 1]))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(encoder_dropout))
    layers = layers[:-2]

    return torch.nn.Sequential(*layers), add


def build_extended_coupling_chain(latent_dim, add_dim, code_configuration):
    coupling_layers = code_configuration["coupling_layers"]
    st_net_layers = code_configuration["st_net_layers"]
    hidden_multiplier = code_configuration["hidden_multiplier"]
    st_dropout = code_configuration["st_dropout"]
    st_funnel_factor = code_configuration["st_funnel_factor"]

    couplings = [
        [ExtendedCoupling(make_conditioner(latent_dim, add_dim, st_net_layers, hidden_multiplier,
                                           st_dropout, st_funnel_factor, code_configuration), code_configuration),
         Reverse(),
         (GeneralBatchNorm() if code_configuration["batch_norm"] == "GlobalBatchNorm" else
          BatchNorm(latent_dim) if code_configuration["batch_norm"] == "BatchNorm" else None),
         ]
        for _ in range(coupling_layers)
    ]
    couplings = [layer for layers in couplings for layer in layers if layer is not None]
    couplings.append(ExtendedCoupling(make_conditioner(latent_dim, add_dim, st_net_layers, hidden_multiplier,
                                                       st_dropout, st_funnel_factor, code_configuration),
                                      code_configuration))
    return couplings


def flow_factory(full_config):
    model_type = full_config["model"]["model_type"]
    device = full_config["run"]["device"]
    code_configuration = full_config["model"]["code_configuration"]
    feature_size = code_configuration["input_shape"][1]
    past_dims = code_configuration["hist_shape"][1:]

    if model_type == "tcNF-base":
        # tcNF-base - passthrough past flow

        past = code_configuration["past"]
        add_dim = reduce(lambda x_, y_: x_ * y_, past_dims)

        couplings = build_extended_coupling_chain(feature_size, add_dim, code_configuration)

        flow = EncodedPastFlow(feature_size, past, identity(), couplings, full_config).to(
            device)
        return flow

    elif model_type == "tcNF-cnn":
        # tcNF-cnn - CNN encoder flow
        past = code_configuration["past"]
        channels = past_dims[1]

        cnn_encoder = cnn(channels, code_configuration)
        # test the encoder to check if the network works and to get the final output dimensions
        # for the flow conditioning
        test_tensor = torch.full((1, *past_dims), 0, dtype=dtype)
        output_test = cnn_encoder.forward(test_tensor)
        encoder_output_shape = output_test.shape[1:]

        add_dim = reduce(lambda x_, y_: x_ * y_, encoder_output_shape)

        couplings = build_extended_coupling_chain(feature_size, add_dim, code_configuration)

        flow = EncodedPastFlow(feature_size, past,
                               cnn_encoder,
                               couplings, full_config).to(device)
        return flow

    elif model_type == "tcNF-mlp":
        # tcNF-mlp - MLP encoder flow

        past = code_configuration["past"]
        _past_dims_ = (past, past_dims[-1])

        mlp_encoder, add_dim = mlp(_past_dims_, code_configuration)

        couplings = build_extended_coupling_chain(feature_size, add_dim, code_configuration)

        flow = EncodedPastFlow(feature_size, past, mlp_encoder,
                               couplings, full_config).to(device)
        return flow

    else:
        raise NotImplemented()
