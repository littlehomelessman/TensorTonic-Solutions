import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    x = torch.tensor(values, dtype=float, requires_grad=True)
    y = torch.sum(torch.pow(x, 3) + torch.multiply(x, 2))
    y.backward()
    return x.grad.tolist()
    
    pass
