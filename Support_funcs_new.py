import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
from collections import OrderedDict
from typing import Callable
import numpy as np
import pandas as pd

class LinearNN(nn.Module):
    def __init__(
        self, 
        num_inputs: int = 4, 
        num_layers: int = 3,
        num_neurons: int = 10,
        act: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        
        layers = []
        
        # input layer
        layers.append(nn.Linear(self.num_inputs, num_neurons))
        
        # hidden layers with linear layer and activation
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), act])
            
        # output layer
        layers.append(nn.Linear(num_neurons, 1))
        
        # build the network
        self.network = nn.Sequential(*layers)
        #print(self.network)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()
    
## above produces a callable function for the model forward pass as f in an array of an array

def make_forward_fn(
    model: nn.Module,
    derivative_order: int = 1,
) -> list[Callable]:
    
    # function output as a function of the inputs x and the parameters of the network
    def f(x: torch.Tensor, params: dict[str, torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
        if isinstance(params, tuple):
            params_dict = tuple_to_dict_parameters(model, params)
        else:
            params_dict = params
        return functional_call(model, params_dict, (x, ))
    

    # creates a list of function and its derivatives up to order derivative_order
    fns = []
    fns.append(f)
    
    
    return fns

def tuple_to_dict_parameters(
    model: nn.Module, params: tuple[nn.Parameter, ...]
) -> OrderedDict[str, nn.Parameter]:
    keys = list(dict(model.named_parameters()).keys())
    values = list(params)
    return OrderedDict(({k:v for k, v in zip(keys, values)}))

## model list created unfortunately have to maunally edit for now currently set for x, y, t, z 

def model_creation(
    inputs: int = 4,
    num_hidden: int = 3, 
    dim_hidden: int = 10
):
        model_Bx = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        model_By = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        model_Vx = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        model_Vy = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        model_rho = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        model_p = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        model_Bz = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        model_Vz = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        models = [model_Bx, model_By, model_Vx, model_Vy, model_rho, model_p, model_Bz, model_Vz]
        return models

## should work fingers crossed
