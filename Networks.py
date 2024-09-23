## packages
import torch
import numpy as np
from torch import nn
from torch import autograd
from torch.distributions.dirichlet import Dirichlet
from Normalizing_flows import *

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


class DropoutNetwork(nn.Sequential):
    def __init__(
        self,
        input_dim,
        output_dim,
        nodes_per_layer,
        num_hidden_layers,
        activation,
        dropout_p,
    ):
        super(DropoutNetwork, self).__init__()
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
            if i != num_hidden_layers:
                if activation == "elu":
                    self.predictor.append(nn.ELU())
                elif activation == "relu":
                    self.predictor.append(nn.LeakyReLU())
                elif activation == "gelu":
                    self.predictor.append(nn.GELU())
            else:
                if activation == "elu":
                    self.predictor.append(nn.ELU())
                elif activation == "relu":
                    self.predictor.append(nn.LeakyReLU())
                elif activation == "gelu":
                    self.predictor.append(nn.GELU())
            self.predictor.append(nn.Dropout(dropout_p))

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
                    self.predictor.append(nn.ELU())
                elif activation == "relu":
                    self.predictor.append(nn.LeakyReLU())
                elif activation == "gelu":
                    self.predictor.append(nn.GELU())
            else:
                if activation == "elu":
                    self.predictor.append(nn.ELU())
                elif activation == "relu":
                    self.predictor.append(nn.LeakyReLU())
                elif activation == "gelu":
                    self.predictor.append(nn.GELU())

    def forward(self, x, return_feature = False):
        for i, l in enumerate(self.predictor[:-2]):
            x = l(x) # outputs last hidden states
        x_scores = self.predictor[-2](x) # in the k-dimensional space
        x_scores = self.predictor[-1](x_scores)
        if return_feature:
            return x_scores, x
        else:
            return x_scores

## The following code is based on code from:  https://github.com/sharpenb/Posterior-Network/tree/main

class Posterior_network(nn.Sequential):  # Deep Linear network
    def __init__(
        self,
        N, #list of ints, see get_N function
        input_dim,
        output_dim,
        nodes_per_layer,
        num_hidden_layers,
        activation,
        network_type,
        n_density=6,
        density_type = "radial_flow", #use radial flow
        loss = "UCE",
        budget_function='id',
    ):
        super(Posterior_network, self).__init__()
        self.output_dim = output_dim
        self.density_type = density_type
        self.latent_dim = output_dim #set to number of classes
        self.loss_name = loss
        self.network_type = network_type
        ## Define network (as self.predictor)
        if network_type == "linear":
            self.predictor =  LinearNetwork(input_dim, output_dim, nodes_per_layer,  num_hidden_layers)

        else:
            self.predictor =  NonLinearNetwork(input_dim, output_dim, nodes_per_layer,  num_hidden_layers, "relu")
           
        ## Define density_estimator/flow
        if self.density_type == 'planar_flow':
            print('planar flow')
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(self.latent_dim, n_density, flow_type=density_type) for c in range(output_dim)])
        elif self.density_type == 'radial_flow':
            print('radialflow')
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(self.latent_dim, n_density, flow_type= self.density_type) for _ in range(output_dim)])
        elif self.density_type == 'iaf_flow':
            print('iaf flow')
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(self.latent_dim, n_density, flow_type=density_type) for c in range(output_dim)])
        else:
            raise NotImplementedError
        self.softmax = nn.Softmax(dim=-1)

        ## Define normalization and budget parameter
        __budget_functions__ = {'one': lambda N: torch.ones_like(N),
                        'log': lambda N: torch.log(N + 1.),
                        'id': lambda N: N,
                        'id_normalized': lambda N: N / N.sum(),
                        'exp': lambda N: torch.exp(N),
                        'parametrized': lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device))}
        if budget_function in __budget_functions__:
            self.N, self.budget_function = __budget_functions__[budget_function](N), budget_function
        else:
            raise NotImplementedError
        self.batch_norm = nn.BatchNorm1d(num_features=self.output_dim) #TODO check this


    def forward(self, x, return_latent = False): #Note, during training you need to optimize on the embedded space
        N = self.N
        batch_size = x.size(0)
        if self.network_type == "linear":
            zk = self.predictor(x)
        else:
            zk, _ = self.predictor(x, return_feature = True) #will return the latent space as well in zk
        zk = self.batch_norm(zk)
        log_q_zk = torch.zeros((batch_size, self.output_dim)).to(zk.device.type)
        alpha = torch.zeros((batch_size, self.output_dim)).to(zk.device.type)
        if isinstance(self.density_estimation, nn.ModuleList):
            for c in range(self.output_dim):
                log_p = self.density_estimation[c].log_prob(zk)
                log_q_zk[:, c] = log_p
                alpha[:, c] = 1. + (N[c] * torch.exp(log_q_zk[:, c]))
        else:
            log_q_zk = self.density_estimation.log_prob(zk)
            alpha = 1. + (N[:, None] * torch.exp(log_q_zk)).permute(1, 0)

        pass
        soft_output_pred = torch.nn.functional.normalize(alpha, p=1)


        if return_latent: # Used afterwards to check latent space
            if self.loss_name == "CE":
                return(soft_output_pred, zk)
            elif self.loss_name == "UCE":
                return(alpha, soft_output_pred, zk)
            else:
                raise NotImplementedError 
        else:
            if self.loss_name == "CE":
                return(soft_output_pred)
            elif self.loss_name == "UCE":
                return(alpha, soft_output_pred)
            else:
                raise NotImplementedError 


