import torch
import torch.nn as nn
import numpy as np
import random
import os
from .score_trainer import Trainer

from typing import List
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class logWriter(object):
    def __init__(self, logdir='./pytorch_tb'):
        """
        tensorboard logging systems
        :param logdir: logging dir
        """
        super().__init__()
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.writer = SummaryWriter(logdir)

    def record(self, loss_item: dict, step: int):
        """
        record the loss item along the train steps
        :param loss_item: loss function dict
        :param step: training steps
        """
        for key, value in loss_item.items():
            self.writer.add_scalar(tag=key, scalar_value=value, global_step=step)


class Swish(nn.Module):
    def __init__(self, dim=-1):
        """Swish activ bootleg from
        https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299

        Args:
            dim (int, optional): input/output dimension. Defaults to -1.
        """
        super().__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)


"""
        https://github.com/ermongroup/ncsn/blob/master/runners/toy_runner.py#L198

        Args:
            input_dim (int, optional): input dimensions. Defaults to 2.
            output_dim (int, optional): output dimensions. Defaults to 1.
            units (list, optional): hidden units. Defaults to [300, 300].
            swish (bool, optional): use swish as activation function. Set False to use
                soft plus instead. Defaults to True.
            dropout (bool, optional): use dropout layers. Defaults to False.
"""

class ToyMLP(nn.Module):
    def __init__(self, input_dim:int=2, output_dim:int=1, units:List=[300, 300], swish:bool=True, dropout:bool=False):
        """
        based on codebase: https://github.com/ermongroup/ncsn/blob/master/runners/toy_runner.py#L198
        :param input_dim (int, optional): input dimensions. Defaults to 2.
        :param output_dim (int, optional): output dimensions. Defaults to 1.
        :param units (list, optional): hidden units. Defaults to [300, 300].
        :param swish (bool, optional): use swish as activation function. Set False to use soft plus instead. Defaults to True.
        :param dropout (bool, optional): use dropout layers. Defaults to False.
        :return: dict{}  {"r2": test_r2, "rmse": test_rmse, "mape": test_mape, "mae": test_mae}
        """
        super(ToyMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for out_dim in units:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                Swish(out_dim) if swish else nn.Softplus(),
                nn.Dropout(.5) if dropout else nn.Identity()
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Energy(nn.Module):
    def __init__(self, net: nn.Module):
        """A simple energy model
        :param net (nn.Module): An energy function, the output shape of the energy function should be (b, 1). The score is computed by grad(-E(x))
        """
        super().__init__()
        self.net = net
        # self.log_prob = lambda x, sigma: -1.0 * torch.sum(self.net(x))
        self.functorch_score = torch.func.grad(self.log_prob, argnums=0)
        self.hessian_func = lambda x: torch.autograd.functional.hessian(self.log_prob, x)
        self.vmapped_hessian = torch.func.vmap(self.hessian_func, in_dims=(0,), out_dims=(0,))

    def log_prob(self, x: torch.tensor, sigma=None):
        """
        log prob func
         :param x (torch.tensor): the tensor for energy computation
        """
        return -1.0 * torch.sum(self.net(x))

    def score(self, x: torch.tensor, sigma=None):
        """
        score function based on torch.autograd.grad operator
        :param x (torch.tensor): the tensor for energy computation
        """
        x = x.requires_grad_()
        logp = -self.net(x).sum()
        return torch.autograd.grad(logp, x, create_graph=True)[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self

    def forward(self, x):
        return self.net(x)


def train_step(data: torch.tensor, trainer: Trainer):

    loss = trainer.get_loss(data)
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    return loss

def train_step_ab(data: torch.tensor, trainer: Trainer, grad_mask: torch.tensor):

    loss = trainer.get_loss(data, v=None, mask=grad_mask)
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    return loss






class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item, ...]
    def __len__(self):
        return self.data.shape[0]
