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
        num_inputs: int = 3, 
        num_layers: int = 3,
        num_neurons: int = 10,
        act: nn.Module = nn.Tanh(),
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
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        #print(f"x:{x1.shape} y:{x2.shape} t:{x3.shape}")
        return self.network(torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), dim=1)).squeeze()
    
## potentially if you can get the appropriate dims: [batch_size, num_features] then don't need the reshape, squeeze might make complicated if the output is expected as 1D tensor

def make_forward_fn(
    model: nn.Module,
    derivative_order: int = 1,
) -> list[Callable]:
    
    # function output as a function of the inputs x and the parameters of the network
    def f(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, params: dict[str, torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
        if isinstance(params, tuple):
            params_dict = tuple_to_dict_parameters(model, params)
        else:
            params_dict = params
        return functional_call(model, params_dict, (x1, x2, x3, ))
    

    # creates a list of function and its derivatives up to order derivative_order
    fns = []
    fns.append(f)
    
    
    dfunc_dx = f
    dfunc_dy = f
    dfunc_dt = f
    for _ in range(derivative_order):
        
        # compute derivative function
        dfunc_dx = grad(dfunc_dx, argnums=(0))
        dfunc_dy = grad(dfunc_dy, argnums=(1))
        dfunc_dt = grad(dfunc_dt, argnums=(2))
        
        # vmap to support batching
        dfunc_dx_vmap = vmap(dfunc_dx, in_dims=(None,0,0,0))
        dfunc_dy_vmap = vmap(dfunc_dy, in_dims=(None,0,0,0))
        dfunc_dt_vmap = vmap(dfunc_dt, in_dims=(None,0,0,0))
        
        fns.append(dfunc_dx_vmap)
        fns.append(dfunc_dy_vmap)
        fns.append(dfunc_dt_vmap)
    
    return fns

## needs completing, mostly how to deal with mutliple f's

def tuple_to_dict_parameters(
    model: nn.Module, params: tuple[nn.Parameter, ...]
) -> OrderedDict[str, nn.Parameter]:
    keys = list(dict(model.named_parameters()).keys())
    values = list(params)
    return OrderedDict(({k:v for k, v in zip(keys, values)}))

## fine as is i hope

def model_creation(
    inputs: int = 3,
    num_hidden: int = 3, 
    dim_hidden: int = 10
):
        model_Bx = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=inputs)
        model_By = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=3)
        model_Vx = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=3)
        model_Vy = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=3)
        model_rho = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=3)
        model_p = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=3)
        models = [model_Bx, model_By, model_Vx, model_Vy, model_rho, model_p]
        return models

## should work fingers crossed
