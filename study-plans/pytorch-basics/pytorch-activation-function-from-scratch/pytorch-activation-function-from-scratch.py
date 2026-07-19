import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x)
    if method == 'relu':
        return torch.max(torch.zeros(len(x)), x).tolist()
    if method == 'sigmoid':
        return 1 / (1 + torch.exp(-x))
    if method == 'tanh':
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    if method == 'leaky_relu':
        return torch.where(x > 0, x, 0.01 * x)
    pass