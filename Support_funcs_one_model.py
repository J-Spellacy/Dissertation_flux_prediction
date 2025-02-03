import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
from collections import OrderedDict
from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from math import floor, log10

class LinearNN(nn.Module):
    def __init__(
        self, 
        num_inputs: int = 4, 
        num_layers: int = 3,
        num_neurons: int = 10,
        act: nn.Module = nn.ELU(),
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
        layers.append(nn.Linear(num_neurons, 8))
        
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
    
def truncated_normal(batch_size, mean=0, std=0.25, lower=-1, upper=1):
    # Generate samples from a truncated normal distribution
    samples = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
    tensor_samples = torch.from_numpy(samples.rvs(batch_size)).float()
    return tensor_samples.requires_grad_(True)

def sig_figs(num: float, precision: int):
        return round(num, -int(floor(log10(abs(num)))) + (precision - 1))

def scale_sample(sample: torch.Tensor, real_min, real_max, scaler_min = -1.0, scaler_max = 1.0) -> torch.Tensor:
            sample_max, sample_min = real_max, real_min
            scaled_sample = (sample - sample_min)/(sample_max - sample_min)*(scaler_max - scaler_min) + scaler_min
            return scaled_sample

def scale_sample_t(sample: torch.Tensor, real_min, real_max, scaler_min = 0.0, scaler_max = 1.0) -> torch.Tensor:
            sample_max, sample_min = real_max, real_min
            scaled_sample = (sample - sample_min)/(sample_max - sample_min)*(scaler_max - scaler_min) + scaler_min
            return scaled_sample
        
def unscale(scaled_out: torch.Tensor, real_min, real_max) -> torch.Tensor:
        s_o_min, s_o_max = scaled_out.min(), scaled_out.max()
        unscaled_out = (scaled_out - s_o_min)/(s_o_max - s_o_min)*(real_max - real_min) + real_min
        return unscaled_out
    
def unscale_not_t(scaled_out: torch.Tensor, real_min, real_max) -> torch.Tensor:
    s_o_min, s_o_max = -1.0, 1.0
    unscaled_out = (scaled_out - s_o_min)/(s_o_max - s_o_min)*(real_max - real_min) + real_min
    return unscaled_out

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss):
        if (train_loss) < self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class convNN(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,  # Number of input channels (e.g., RGB images have 3)
        num_classes: int = 8   # Number of output classes
    ) -> None:
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # A fully connected layer for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),  # Assuming input images are 32x32
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)      # Apply convolutional layers
        x = torch.flatten(x, 1)      # Flatten the output for the fully connected layer
        x = self.fc_layers(x)        # Apply fully connected layers
        return x
    
def generate_points_on_sphere(batch_size, radius):
    # Uniformly sample azimuthal angle theta in [0, 2*pi]
    theta = 2 * torch.pi * torch.rand(batch_size)
    # Uniformly sample cosine of polar angle phi
    cos_phi = 2 * torch.rand(batch_size) - 1
    sin_phi = torch.sqrt(1 - cos_phi**2)

    # Calculate the Cartesian coordinates
    x = radius * sin_phi * torch.cos(theta)
    y = radius * sin_phi * torch.sin(theta)
    z = radius * cos_phi

    # Stack the coordinates together along the last dimension
    points = torch.stack([x, y, z], dim=-1)
    return points