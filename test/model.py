import os
import math
import torch
import torch.nn as nn
import torch.optim as optim


class TwoLayerMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x
