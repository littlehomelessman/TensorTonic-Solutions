import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_features)
        

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))
