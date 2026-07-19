import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    if op == 'add':
        return torch.add(torch.tensor(x), torch.tensor(y)).tolist()
    if op == 'matmul':
        return torch.matmul(torch.tensor(x), torch.tensor(y)).tolist()
    if op == 'power':
        return torch.pow(torch.tensor(x), torch.tensor(y)).tolist()
    if op == 'multiply':
        return torch.multiply(torch.tensor(x), torch.tensor(y)).tolist()
    if op == 'max':
        return torch.max(torch.tensor(x), torch.tensor(y)).tolist()
    pass