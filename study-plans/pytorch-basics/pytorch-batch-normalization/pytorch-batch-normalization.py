import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    X = torch.tensor(X, dtype=float)
    mean = torch.mean(X, dim=0)
    var = torch.mean((X - mean)**2, dim=0)
    X1 = (X - mean) / torch.sqrt(var + eps)
    return gamma * X1 + beta
    pass
