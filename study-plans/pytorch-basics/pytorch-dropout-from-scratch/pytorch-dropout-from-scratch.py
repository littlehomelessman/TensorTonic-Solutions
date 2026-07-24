import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        print(x.shape[0])
        if self.training and self.p == 1.0:
            return torch.zeros(x.shape[0])
        if self.training and self.p > 0:
            mask = (torch.rand(x.shape) >= self.p).float()
            # return mask * x / (1 - self.p)
            # print(mask, mask.shape)
            # print(x, x.shape)
            # print(x * mask)
            return  x * mask / (1 - self.p)

        return x
