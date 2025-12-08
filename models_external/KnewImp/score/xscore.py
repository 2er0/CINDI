import math
import torch

# $\frac{1}{\sqrt{2\times \pi}}$
gauss_coeff = 1 / math.sqrt(2 * math.pi)
log_gauss_coeff = math.log(gauss_coeff)

def gauss_pdf(x, mu, sigma):
    """
    single gaussian pdf
    :param x: requst data
    :param mu: observation data
    :param sigma: the bandwidth
    :return: estimated pdf
    """
    return gauss_coeff / sigma * torch.exp(- (x - mu) ** 2/ (2 * sigma**2))
vmapped_gauss_pdf = torch.func.vmap(func=gauss_pdf, in_dims=(None, 0, None), out_dims=(0))
delta_value = lambda x, y: x - y
vmapped_delta_value = torch.func.vmap(func=delta_value, in_dims=(None, 0))

def gauss_log_pdf(x, mu, sigma):
    """
    single gaussian log pdf
    :param x: requst data, the shape: [1, data]
    :param mu: observation data, the shape: [1, data]
    :param sigma: the bandwidth
    :return: estimated pdf
    """
    return log_gauss_coeff - math.log(sigma) - ((x.squeeze(0) - mu) ** 2/ (2 * sigma**2))
vmapped_gauss_log_pdf = torch.func.vmap(func=gauss_log_pdf, in_dims=(None, 0, None), out_dims=(0))

def gauss_score(x, mu, sigma):
    """
    single gaussian log pdf
    :param x: requst data
    :param mu: observation data
    :param sigma: the bandwidth
    :return: estimated pdf
    """
    return - ((x.squeeze(0) - mu) / (sigma**2))
vmapped_gauss_score = torch.func.vmap(func=gauss_score, in_dims=(None, 0, None), out_dims=(0))



def parzen_log_pdf(x, mu, sigma):
    """
    parzen window kde (single sample)
    :param x: request data
    :param mu: historical data [n, data dimension]
    :param sigma: the bandwidth
    :return: estimated pdf
    """
    # $\frac{1}{n}(gaussian pdf)$
    # data_number = mu.shape[0]
    return torch.log(torch.mean(torch.exp(vmapped_gauss_log_pdf(x, mu, sigma))))
vammped_parzen_log_pdf = torch.func.vmap(func=parzen_log_pdf, in_dims=(0, None, None), out_dims=0)
# the score function (just for validation)
score_parzen = torch.func.grad(parzen_log_pdf, argnums=(0))
vmapped_score_parzen = torch.func.vmap(func=score_parzen, in_dims=(0, None, None))

# self defined parzen score
def parzen_score(x, mu, sigma):
    temp_value1 = -1.0 * vmapped_gauss_pdf(x, mu, sigma) / sigma
    temp_value2 = vmapped_delta_value(x, mu)
    grad_pdf = torch.mean(temp_value1 * temp_value2, dim=0)
    pdf = torch.exp(parzen_log_pdf(x, mu, sigma))
    return grad_pdf / (pdf + 1.0e-9) / mu.shape[-1]
# (x: request_data, [n, data_dim], mu: observational data, [n_{past observe}, data_dim], sigma: bandwidth [)
vammped_parzen_score = torch.func.vmap(func=parzen_score, in_dims=(0, None, None), out_dims=0)


if __name__ == "__main__":
    from loguru import logger
    logger.info(f"Let us start playing!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dim = 3
    # test of the score estimator
    data = torch.tensor([2, 2.5, 3, 1, 6, 7], device=device).reshape((-1, data_dim))
    x = torch.tensor([-3.0, 0.0, 1.0, 0.0, 1.0, 1.1], device=device).reshape((-1, data_dim))
    sigma = torch.tensor([1.0], device=device)
    logger.info(f"the value is: {torch.exp(vammped_parzen_log_pdf(x, data, sigma))}")
    functorch_score = vmapped_score_parzen(x, data, sigma)
    mine_score = vammped_parzen_score(x, data, sigma)
    logger.info(f"the ground truth: {functorch_score}, mine result: {mine_score}")





