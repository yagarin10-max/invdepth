"""
based on https://www.youtube.com/watch?v=eT7ZhZjLBqM
"""

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim:int=2, hidden_dim:int=256, out_dim=1) -> None:
        super(MLP, self).__init__()

        self.layers = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    net = MLP()