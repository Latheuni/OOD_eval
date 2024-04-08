## packages
import torch
from torch import nn


## networks
class LinearNetwork(nn.Sequential):  # Deep Linear network
    def __init__(self, input_dim, output_dim, nodes_per_layer, num_hidden_layers):
        super(LinearNetwork, self).__init__()
        if num_hidden_layers == 0: 
            dims = [input_dim] + [output_dim]
        elif isinstance(nodes_per_layer, list):
            num_hidden_layers = len(nodes_per_layer)
            dims = [input_dim] + nodes_per_layer + [output_dim]
        else:
            dims = [input_dim] + num_hidden_layers * [nodes_per_layer] + [output_dim]

        self.predictor = nn.ModuleList()
        for i in range(0, num_hidden_layers + 1):
            self.predictor.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for i, l in enumerate(self.predictor):
            x = l(x)
        return x


class NonLinearNetwork(nn.Sequential):  # Deep Linear network
    def __init__(
        self,
        input_dim,
        output_dim,
        nodes_per_layer,
        num_hidden_layers,
        activation,
    ):
        super(NonLinearNetwork, self).__init__()

        if num_hidden_layers == 0:
            dims = [input_dim] + [output_dim]
        elif isinstance(nodes_per_layer, list):
            num_hidden_layers = len(nodes_per_layer)
            dims = [input_dim] + nodes_per_layer + [output_dim]
        else:
            dims = [input_dim] + num_hidden_layers * [nodes_per_layer] + [output_dim]

        self.predictor = nn.ModuleList()
        for i in range(num_hidden_layers + 1):
            self.predictor.append(nn.Linear(dims[i], dims[i + 1]))

            if i != num_hidden_layers:
                if activation == "elu":
                    self.predictor.append(nn.Elu())
                elif activation == "relu":
                    self.predictor.append(nn.LeakyReLU())
            else:

                if activation == "elu":
                    self.predictor.append(nn.Elu())
                elif activation == "relu":
                    self.predictor.append(nn.LeakyReLU())

    def forward(self, x):
        for i, l in enumerate(self.predictor):
            x = l(x)
        return x
