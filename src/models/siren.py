"""
based on https://www.youtube.com/watch?v=eT7ZhZjLBqM
and official SIREN implementation (.ipynb)
"""

import torch
import torch.nn as nn
import math
import numpy as np


class LinearLayer(nn.Module):
    """
    Just a linear layer but with the specific weight initialization
    """

    def __init__(self, in_dim: int, out_dim: int, omega_0: float, is_first=False) -> None:
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.omega_0 = omega_0
        self.is_first = is_first

        self.apply(self._init_weights)

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.is_first:
                nn.init.uniform_(module.weight.data, -1 / self.in_dim, 1 / self.in_dim)
            else:
                nn.init.uniform_(
                    module.weight.data,
                    -math.sqrt(6 / self.in_dim) / self.omega_0,
                    +math.sqrt(6 / self.in_dim) / self.omega_0,
                )


class SineLayer(LinearLayer):
    def __init__(self, in_dim: int, out_dim: int, omega_0: float, is_first=False) -> None:
        super(SineLayer, self).__init__(in_dim, out_dim, omega_0, is_first)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 256,
        out_dim: int = 1,
        hidden_layers: int = 3,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
        outermost_linear: bool = False,
    ) -> None:
        super(SIREN, self).__init__()
        self.net = []
        self.net.append(SineLayer(in_dim=in_dim, out_dim=hidden_dim, omega_0=first_omega_0, is_first=True))
        for _ in range(hidden_layers):
            self.net.append(SineLayer(in_dim=hidden_dim, out_dim=hidden_dim, omega_0=hidden_omega_0, is_first=False))
        if outermost_linear:
            final_linear = nn.Linear(hidden_dim, out_dim)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, 
                                             np.sqrt(6 / hidden_dim) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(in_dim=hidden_dim, out_dim=out_dim, omega_0=hidden_omega_0, is_first=False))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class MyTanh(nn.Module):
    def __init__(self):
        super(MyTanh, self).__init__()

    def forward(self, x):
        # Apply the tanh activation function
        x = torch.tanh(2.0*x)
        return x


class TMO(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=256, hidden_layers=1):
        super().__init__()
        out_dim = 1
        
        self.net = []
        self.net.append(nn.Linear(in_dim, hidden_dim))
        self.net.append(nn.LeakyReLU(inplace=True))#(nn.ReLU(inplace=True))

        self.net.append(nn.Linear(hidden_dim, out_dim)) 
        self.net.append(MyTanh())    
        self.net = nn.Sequential(*self.net)
        

        
    def init_weights(self):
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.Tensor([0.5]))
    
    def forward(self, coords):
        output = self.net(coords)
        return output

if __name__ == "__main__":
    net = SIREN()
    print(net)
