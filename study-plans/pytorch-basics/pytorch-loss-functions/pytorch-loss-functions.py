import torch
import torch.nn as nn

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.as_tensor(pred).float()
    target = torch.as_tensor(target).float()
    n = pred.shape[0]
    if method == "mse":
        return 1/n*torch.sum((target - pred)**2)
    elif method == "cross_entropy":
        log_sum_exp = torch.logsumexp(pred, dim=-1)
        target_idx = target.long()
        correct_logits = pred[torch.arange(n), target_idx]
        return torch.mean(log_sum_exp - correct_logits)
    elif method == "huber":
        return torch.nn.functional.huber_loss(pred, target, delta=delta)
        
    
