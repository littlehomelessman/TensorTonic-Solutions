import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float)
    if method=="relu":
        return torch.clamp(x, 0)
    elif method=="sigmoid":
        return 1/(1+torch.exp(-x))
    elif method=="tanh":
        return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
    elif method=="leaky_relu":
        return torch.clamp(x, 0.01 * x)

    pass