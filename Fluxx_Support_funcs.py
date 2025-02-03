import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
from collections import OrderedDict
from typing import Callable
import numpy as np
import pandas as pd

class FluxNN(nn.Module):
    def __init__(
        self, 
        num_inputs: int = 12, # all 8 outputs from plasma model + L-shell + time
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
        layers.append(nn.Linear(num_neurons, 17))
        
        # build the network
        self.network = nn.Sequential(*layers)
        #print(self.network)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()


def concatenate_previous_time_steps(input_tensor, max_time_step):
    """
    Concatenate each time step with the previous one up to a maximum time step.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, num_features).
        max_time_step (int): Maximum number of time steps to concatenate with the previous one.

    Returns:
        torch.Tensor: Concatenated tensor of shape (batch_size - max_time_step, num_features * (max_time_step + 1)).
    """
    batch_size, num_features = input_tensor.shape

    # Initialize a list to store concatenated tensors
    concatenated_tensors = []

    # Concatenate each time step with the previous one up to max_time_step
    for t in range(max_time_step, batch_size):
        previous_time_steps = input_tensor[t - max_time_step:t + 1]  # Get previous time steps
        concatenated_tensors.append(previous_time_steps.flatten())  # Flatten and append

    # Stack the concatenated tensors along a new batch dimension
    return torch.stack(concatenated_tensors)
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
