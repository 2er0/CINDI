import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import random
import matplotlib.pyplot as plt
import scipy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout_probability=0.1):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(d_in, d_hidden[0]))
        layers.append(nn.Sigmoid())
        layers.append(nn.Dropout(dropout_probability))
        for i in range(1, len(d_hidden)):
            layers.append(nn.Linear(d_hidden[i-1], d_hidden[i]))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout_probability))
        layers.append(nn.Linear(d_hidden[-1], d_out))
        self.mlp = nn.Sequential(*layers)
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


def display_histogram(X0):
    P = len(X0)
    h0,b = torch.histogram(X0.flatten(), bins=nbins, range=[-xmax, xmax])
    plt.bar( torch.linspace(-xmax,xmax,nbins), h0/P * nbins/(2*xmax), width=2*xmax/(nbins-1), align='center', color='red')
    # plt.show()
def display_trajectories(Y, m = 1000, alpha=.3, linewidth=.2):  # m=number of trajectories to display
    disp_ind = torch.round( torch.linspace(0,P-1,m) ).type(torch.int32)
    I = torch.argsort(Y[:,-1])
    Y = Y[I,:]
    for i in range(m):
        k = disp_ind[i]
        s = k/(P-1)
        plt.plot( torch.linspace(0, T, N), Y[k,:], color=[s.item(),0,1-s.item()], alpha=alpha, linewidth=linewidth )


def gauss(x,m,s): return 1/( torch.sqrt( 2*torch.tensor(torch.pi) ) * s ) * torch.exp( -(x-m)**2 / (2*s**2) )
def gaus_mixture(x,mu,sigma,a):
    y = x*0
    for i in range(len(mu)):
        y = y + a[i] * gauss(x,mu[i],sigma[i])
    return y
def sample_mixture(P, mu,sigma,a):
    a1 = np.array( a/torch.sum(a) )
    I = np.random.choice( np.arange(0, len(a)), size=(P,1), p=a1 )
    return ( mu[I] + torch.randn((P,1))*sigma[I] ).flatten()
def rho(x,t):
    sigma_t = torch.sqrt( torch.exp(-2*t)*sigma**2 + 1 -  torch.exp(-2*t) )
    return gaus_mixture(x, mu*torch.exp(-t), sigma_t,a)

if __name__ == "__main__":


    # define a Gaussian Mixture Model
    comp_num = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(seed=1024)
    weight_list = [2.0, 3.0, 2.0]
    mu_list = [-0.7, 0.0, 0.5]
    mu_list = list(map(lambda x: x * 3, mu_list))
    sigma_list = [0.02, 0.05, 0.03]
    sigma_list = list(map(lambda x: x * 9, sigma_list))
    initial_mix = torch.distributions.Categorical(torch.tensor(weight_list, device=device) / sum(weight_list))
    initial_comp = torch.distributions.MultivariateNormal(torch.tensor(mu_list, device=device).reshape((-1, 1)),
                                                          torch.stack([(sigma_list[i] ** 2)
                                                                       * torch.eye(1, device=device)
                                                                       for i in range(comp_num)]))
    initial_model = torch.distributions.MixtureSameFamily(initial_mix, initial_comp)
    init_sample = initial_model.sample([1000, ])
    rho0 = lambda x: torch.exp(initial_model.log_prob(x.reshape((-1, 1)), ))
    sum_rho0 = lambda x: torch.sum(rho0(x))
    eta0 = lambda x: torch.func.grad(sum_rho0)(x).reshape((-1)) / rho0(x)




    # 绘制gmm的hist gram
    fs = (6, 3)  # size of figures for display
    xmax = 3
    nbins = 200

    x = torch.linspace(-xmax, xmax, 1000, device=device)
    plt.figure(figsize=fs)
    plt.fill_between(x.cpu().numpy(), rho0(x).cpu().numpy())
    plt.show()

    init_rho, init_eta = rho0(init_sample), eta0(init_sample)
    logger.info(f"the init rho: {init_rho.shape}, init eta: {init_eta.shape}")

    # 绘制score图像
    plt.figure(figsize=fs)
    plt.plot(x.cpu().numpy(), rho0(x).detach().cpu().numpy(), label='$\\rho_0$')
    plt.plot(x.cpu().numpy(), eta0(x).detach().cpu().numpy() / 40, label='$\\frac{1}{40}\\eta_0$')
    plt.legend()
    plt.show()

    # obtain training sample
    N = 2000  # number of training samples
    X0 = initial_model.sample([N, ]).reshape((-1))

    Tmin = 1e-1
    Tmax = .5
    t_list = torch.linspace(Tmin, Tmax, N, device=device)

    W = torch.randn((N, ), device=device)
    Xt = torch.exp(-t_list) * X0 + torch.sqrt(1.0 - torch.exp(-2.0 * t_list)) * W
    t_map = lambda t: ( (t-Tmin)/(Tmax-Tmin) - 1/2 ) * 2*xmax # map t to the same range as x to ease training
    U = torch.vstack((Xt, t_map(t_list))).T
    V = (torch.exp(-t_list) * X0 - Xt ) / ( 1-torch.exp(-2*t_list) )

    logger.info(f"the Xt shape: {Xt.shape}, u shape: {U.shape}, v shape: {V.shape}")

    d_in = 2
    d_hidden = [300, 100, 20]
    d_out = 1
    model = MLP(d_in, d_hidden, d_out, dropout_probability=0).to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * 10)
    num_epochs = 600
    Loss_list = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = torch.sum((model(U).flatten() - V.flatten()) ** 2) / N
        loss.backward()
        optimizer.step()
        Loss_list.append(loss.item())
    plt.plot(Loss_list)
    plt.show()

    alpha = 0.01
    P, N = 500, 2500   # number of particles
    T = 2  # final time
    tau = T / N  # step size

    mu = torch.tensor([-.7, 0, .5]) * 3  # mean
    sigma = torch.tensor([.02, .05, .03]) * 9  # std
    a = torch.tensor([2, 3, 2])  # weight, should sum to 1

    Y0 = torch.randn([P]) # torch.tensor(scipy.stats.norm.ppf(np.arange(0, P) / P + 1 / (2 * P)))
    logger.info(f"Y0 shape: {Y0.shape}")
    # plt.figure(figsize=fs)
    display_histogram(Y0.cpu())
    # plt.plot(x, rho(x, torch.tensor(10)))
    #
    Y = torch.zeros((P,N), device=device)
    Y[:, 0] = Y0

    logger.info(f"the Y shape: {Y.shape}")

    t = torch.tensor(T)
    for i in range(N-1):
        input_tensor = torch.vstack([Y[:, i], t * torch.ones_like(Y[:, i])] ).to(device).T
        model_output = model( input_tensor).squeeze(-1)
        # logger.info(f"the input tensor shape: {input_tensor.shape}, model output: {model_output.shape}")
        Y[:, i+1] = Y[:, i] + tau * (Y[:, i] + (1 + alpha) * model_output) + torch.sqrt(2 * torch.tensor(tau * alpha, device=device)) * torch.randn((P,)).to(device)
        t = t-tau

    # display evolving density
    t_list = T - torch.tensor([0.01, .01, .05, .2, .99 * T])
    for i in range(len(t_list)):
        s = i / len(t_list)
        t = t_list[i]
        k = torch.round(t / tau).type(torch.int32)
        plt.subplot(len(t_list), 1, i + 1)
        plt.plot(x.cpu().numpy(), rho(x, T - t).cpu().numpy(), color='blue')
        display_histogram(Y[:, k].detach().cpu())
    plt.show()
# comp_num = 3
# weight_list =
# initial_mix = torch.distributions.Categorical(torch.tensor([1 / comp for i in range(comp)], device=device))
#
# initial_comp = torch.distributions.MultivariateNormal(
#     torch.tensor([[d * np.sqrt(3) / 2.], [-d * np.sqrt(3) / 2.]], device=device).float(),
#     var * torch.stack([torch.eye(1, device=device) for i in range(comp)]))
# initial_model = torch.distributions.MixtureSameFamily(initial_mix, initial_comp)
# init_sample = initial_model.sample([1000, ])