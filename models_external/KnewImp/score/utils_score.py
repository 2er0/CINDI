import math
import torch


gauss_coeff = 1 / math.sqrt(2 * math.pi)
def gaussian_pdf(x, mu, sigma):
    # logger.info(f"the x shape: {x.shape}, the mu shape: {mu.shape}")
    return gauss_coeff / sigma * torch.exp(-torch.sum((x - mu) ** 2) / (2 * sigma**2))

# the vmapped gaussian pdf of x in input dim
vmapped_gauss_pdf = torch.func.vmap(func=gaussian_pdf, in_dims=(None, 0, None), out_dims=(0,))


def parzen_window_pdf(x, data, sigma):
    """
    parzen window non-parameteric pdf
    :param x: historical observations
    :param data: the request data we want pdf. Note that, this is a vmapped version (shape should be (n, dimension))
    :param sigma: the bandwidth
    :return: the mean value of the log_pdf(x)
    """
    return torch.sum(vmapped_gauss_pdf(x, data, sigma), dim=0)

# the score function
score_parzen_pdf = torch.func.grad_and_value(func=parzen_window_pdf, argnums=0)


if __name__ == "__main__":
    from loguru import logger
    logger.info(f"Let us start playing!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test of the score estimator
    data = torch.tensor([2, 2.5, 3, 1, 6], device=device)
    x = torch.tensor([3.0], device=device)
    sigma = 1.0
    logger.info(f"the value is: {parzen_window_pdf(x, data, sigma).item():.4f}")

    logger.warning(f"large scale experiment start!")
    # test of random tensor
    random_tensor = torch.randn([64, 3], device=device)
    test_tensor = torch.randn([2, 3], device=device)
    score_function, pdf_function = score_parzen_pdf(test_tensor, random_tensor, sigma)
    logger.info(f"the value is: {score_function.shape}, pdf function shape: {pdf_function.shape}")

