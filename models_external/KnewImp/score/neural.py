import torch
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MLP(nn.Module):
    def __init__(self, input_dim=8, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, 1, bias=True)
        self.act = lambda x: torch.sigmoid(x)

    def forward(self, x_input):
        x = self.fc1(x_input)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

def dsm_loss(model, x, v, sigma=0.1):
    """DSM loss from
    A Connection Between Score Matching
        and Denoising Autoencoders

    The loss is computed as
    x_ = x + v   # noisy samples
    s = -dE(x_)/dx_
    loss = 1/2*||s + (x-x_)/sigma^2||^2

    Args:
        x (torch.Tensor): input samples
        v (torch.Tensor): sampled noises
        sigma (float, optional): noise scale. Defaults to 0.1.

    Returns:
        DSM loss
    """
    x = x.requires_grad_()
    v = v * sigma
    x_ = x + v
    s = model.score(x_)
    loss = torch.norm(s + v / (sigma ** 2), dim=-1) ** 2
    loss = loss.mean() / 2.
    return loss


class EnergyModel(nn.Module):
    """
    write with functorch
    """
    def __init__(self, net):
        super(EnergyModel, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def sum_forward(self, x):
        return -1.0 * torch.sum(self.net(x))

    def score(self, x, sigma=None):
        # $\nabla_x{-\log{q}}$
        score_value = torch.func.grad(self.sum_forward,)(x)
        return score_value

class Energy(nn.Module):
    def __init__(self, net):
        super(Energy, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logq = -1.0 * torch.sum(self.net(x))
        return torch.autograd.grad(logq, x, create_graph=True)[0]




def dsm_functorch(model, x, v, sigma=0.1):
    x = x.requires_grad_()
    v = v * sigma
    x_ = x + v
    s = model.score(x_)
    # SM_loss = (1. / (2. * self.sigma)) * ((s + epsilon) ** 2.).sum(-1)
    loss = (1. / (2. * sigma)) * ((s + v) ** 2.).sum(-1)
    # loss = torch.norm((s) + v / (sigma ** 2), dim=-1) ** 2
    loss = torch.sum(loss) # / 2.0
    return loss


def regress_loss(x, y):
    return torch.mean((x - y) * (x - y))

if __name__ == "__main__":
    from loguru import logger
    import random, numpy as np

    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(seed=1024)

    hidden_dim = 64
    lr = 1.0e-4
    comp = 2
    d = 3
    var = 0.3
    score_epoch = 100
    initial_mix = torch.distributions.Categorical(torch.tensor([1 / comp for i in range(comp)], device=device))

    initial_comp = torch.distributions.MultivariateNormal(torch.tensor([[d * np.sqrt(3) / 2.], [-d * np.sqrt(3) / 2.]], device=device).float(),
                                                          var * torch.stack([torch.eye(1, device=device) for i in range(comp)]))
    initial_model = torch.distributions.MixtureSameFamily(initial_mix, initial_comp)
    init_sample = initial_model.sample([1000,])

    logger.info(f"the init sample shape is: {init_sample.shape}")

    import seaborn as sns

    # sns.kdeplot(init_sample.cpu().numpy().reshape((-1)))
    # plt.show()
    mlp = MLP(input_dim=1, hidden_num=hidden_dim).to(device=device)
    energy_model = EnergyModel(mlp)
    score_optimizer = torch.optim.SGD(energy_model.parameters(), lr=lr, momentum=0.9)


    for step in range(score_epoch):
        noise = torch.randn_like(init_sample)
        loss = dsm_functorch(model=energy_model, x=init_sample, v=noise, sigma=0.1)
        score_optimizer.zero_grad()
        loss.backward()
        score_optimizer.step()
        logger.info(f"epoch: {step+1:04d}, the loss of score function is: {loss:.4f}")







    #
    # input_dim, hidden_dim = 1, 2
    # seed_num = 42
    # batch_size = 32
    # noise_scale = 0.10
    # torch.manual_seed(seed=seed_num)
    # random.seed(seed_num)
    # np.random.seed(seed_num)
    #
    # mlp = MLP(input_dim=input_dim, hidden_num=hidden_dim).to(device=device)
    # energy_model = Energy(net=mlp).to(device=device)
    # optimizer = torch.optim.Adam(energy_model.parameters(), lr=1.0e-3)
    #
    #
    #
    #
    # test_tensor = torch.rand([batch_size, input_dim], device=device)
    # noise_tensor = torch.randn([batch_size, input_dim], device=device) * noise_scale
    #
    # # log_q = energy_model(test_tensor)
    # # logger.info(f"the log q shape: {log_q.shape}")
    # # score_value = energy_model.score(x=test_tensor)
    # # logger.info(f"the score shape: {score_value.shape}")
    #
    #
    # print(f"before backprop:")
    # for name, params in energy_model.named_parameters():
    #     # print(f"name: ", name, "require grads: ", params.requires_grad, )
    #     print(f"the grad of {name}:\n ", params.grad)
    #     # print(f"the value: ", params)
    # optimizer.zero_grad()
    # pred_score = energy_model.score(x=test_tensor)
    # # loss_value = regress_loss(noise_tensor, pred_score)
    # loss_value = dsm_functorch(energy_model, test_tensor, noise_tensor, sigma=1.0)
    # # logger.info(f"the loss value: {loss_value}")
    # loss_value.backward()
    # print(f"after backprop:")
    # for name, params in energy_model.named_parameters():
    #     # print(f"name: ", name, "require grads: ", params.requires_grad, )
    #     print(f"the grad of {name}:\n ", params.grad)
    #     # print(f"the value: ", params)

















