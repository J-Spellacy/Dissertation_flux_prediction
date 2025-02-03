## second document
## imports dependencies and dataframe
import torch
import torch.nn as nn
import torchopt
from typing import Callable
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Support_funcs_new import make_forward_fn, LinearNN, model_creation
import matplotlib.patches as patches
import random

# Convert df to NumPy array to PyTorch tensor with dimensions [num_rows, num_features]
# 2d df
# df = pd.read_csv('C:/Users/User/Desktop/PINN_torch/PINN_data_OMNI_cutdown.csv', index_col=0)
df_val = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_cutdown.csv', index_col=0)
data_numpy_arr = df_val.values
total_tensor = torch.from_numpy(data_numpy_arr).float()

## loss definition TO DO: add other equations to create full physics loss also work on maybe improving the boundary loss maybe scaling? its a bit shit

def make_loss_fn(funcs_list: list) -> Callable:
    def loss_fn(params_list: list, x: torch.Tensor):
        
        # interior loss
        fs = []
        for targ in funcs_list:
            fs.append(targ[0])

        params_list = [tuple(params_list[i:i+10]) for i in range(0, len(params_list), 10)]
        
        Bx_params = params_list[0]
        By_params = params_list[1]
        Vx_params = params_list[2]
        Vy_params = params_list[3]
        rho_params = params_list[4]
        p_params = params_list[5]
        Bz_params = params_list[6]
        Vz_params = params_list[7]
        
        Bx = fs[0]
        By = fs[1]
        Vx = fs[2]
        Vy = fs[3]
        rho = fs[4]
        p = fs[5]
        Bz = fs[6]
        Vz = fs[7]        
        
        rho_value = rho(x, rho_params)
        Vx_value = Vx(x, Vx_params)
        Vy_value = Vy(x, Vy_params)
        Bx_value = Bx(x, Bx_params)
        By_value = By(x, By_params)
        p_value = p(x, p_params)
        Bz_value = Bz(x, Bz_params)
        Vz_value = Vz(x, Vz_params)
        
        mu_0 = 4 * np.pi * 1e-7
        
        dVxdX = torch.autograd.grad(
            inputs = x,
            outputs = Vx_value,
            grad_outputs=torch.ones_like(Vx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        drhodX = torch.autograd.grad(
            inputs = x,
            outputs = rho_value,
            grad_outputs=torch.ones_like(rho_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVydX = torch.autograd.grad(
            inputs = x,
            outputs = Vy_value,
            grad_outputs=torch.ones_like(Vy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBxdX = torch.autograd.grad(
            inputs = x,
            outputs = Bx_value,
            grad_outputs=torch.ones_like(Bx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBydX = torch.autograd.grad(
            inputs = x,
            outputs = By_value,
            grad_outputs=torch.ones_like(By_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dpdX = torch.autograd.grad(
            inputs = x,
            outputs = p_value,
            grad_outputs=torch.ones_like(p_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVzdX = torch.autograd.grad(
            inputs = x,
            outputs = Vz_value,
            grad_outputs=torch.ones_like(Vz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBzdX = torch.autograd.grad(
            inputs = x,
            outputs = Bz_value,
            grad_outputs=torch.ones_like(Bz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVydy_value = dVydX[:, 1]
        dVydt_value = dVydX[:, 2]
        dVxdx_value = dVxdX[:, 0]
        dVxdt_value = dVxdX[:, 2]
        drhodx_value = drhodX[:, 0]
        drhody_value = drhodX[:, 1]
        drhodt_value = drhodX[:, 2]
        drhodz_value = drhodX[:, 3]
        dBxdx_value = dBxdX[:, 0]
        dBxdy_value = dBxdX[:, 1]
        dBxdt_value = dBxdX[:, 2]
        dBxdz_value = dBxdX[:, 3]
        dBydx_value = dBydX[:, 0]
        dBydy_value = dBydX[:, 1]
        dBydt_value = dBydX[:, 2]
        dBydz_value = dBydX[:, 3]
        dpdx_value = dpdX[:, 0]
        dpdy_value = dpdX[:, 1]
        dVzdz_value = dVzdX[:, 3]
        dBzdx_value = dBzdX[:, 0]
        dBzdy_value = dBzdX[:, 1]
        dBzdt_value = dBzdX[:, 2]
        dBzdz_value = dBzdX[:, 3]
        
        # mass continuity equation
        #print(f"rho {Bx_value}")#, drhodx {drhodx_value}")
        
        Vxdrhodx_value = drhodx_value * Vx_value
        rhodVxdx_value =  dVxdx_value * rho_value
        drhoVxdx_value = Vxdrhodx_value + rhodVxdx_value
        
        Vydrhody_value = drhody_value * Vy_value
        rhodVydy_value =  dVydy_value * rho_value
        drhoVydy_value = Vydrhody_value + rhodVydy_value
        
        Vzdrhodz_value = drhodz_value * Vz_value
        rhodVzdz_value = dVzdz_value * rho_value
        drhoVzdz_value = Vzdrhodz_value + rhodVzdz_value
        
        div_rhoVx_value = drhoVxdx_value + drhoVydy_value + drhoVzdz_value
        
        mc_interior = drhodt_value - div_rhoVx_value
        
        # Gauss residual equation
        
        Gauss_interior = dBxdx_value + dBydy_value + dBzdz_value
        
        # induction residual equation (assumed ideal can change with adding an eta output model as well as a gamma output model)
        # based on low frequency ampere's law
        jx_value = (1/mu_0) * (dBzdy_value - dBydz_value)
        jy_value = (1/mu_0) * (dBxdz_value - dBzdx_value)
        jz_value = (1 / mu_0) * (dBydx_value - dBxdy_value)
        
        VcrossB_x = (Vy_value * Bz_value) - (Vz_value * By_value)
        VcrossB_y = (Vz_value * Bx_value) - (Vx_value * Bz_value)
        VcrossB_z = (Vx_value * By_value) - (Vy_value * Bx_value)
        
        dVcrossB_xdX = torch.autograd.grad(
            inputs = x,
            outputs = VcrossB_x,
            grad_outputs=torch.ones_like(VcrossB_x),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVcrossB_ydX = torch.autograd.grad(
            inputs = x,
            outputs = VcrossB_y,
            grad_outputs=torch.ones_like(VcrossB_y),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVcrossB_zdX = torch.autograd.grad(
            inputs = x,
            outputs = VcrossB_z,
            grad_outputs=torch.ones_like(VcrossB_z),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVcrB_xdy_value = dVcrossB_xdX[:, 1]
        dVcrB_xdz_value = dVcrossB_xdX[:, 3]
        dVcrB_ydx_value = dVcrossB_ydX[:, 0]
        dVcrB_ydz_value = dVcrossB_ydX[:, 3]
        dVcrB_zdx_value = dVcrossB_zdX[:, 0]
        dVcrB_zdy_value = dVcrossB_zdX[:, 1]
        
        curl_VcrossB_x = dVcrB_zdy_value - dVcrB_ydz_value
        curl_VcrossB_y = dVcrB_xdz_value - dVcrB_zdx_value
        curl_VcrossB_z = dVcrB_ydx_value - dVcrB_xdy_value
        
        induction_interior_x = dBxdt_value - curl_VcrossB_x# curl goes here
        induction_interior_y = dBydt_value - curl_VcrossB_y
        induction_interior_z = dBzdt_value - curl_VcrossB_z
        
        induction_interior = torch.sqrt(induction_interior_x ** 2 + induction_interior_y ** 2 + induction_interior_z ** 2)
        
        # sums physics interior equation residuals
        interior = mc_interior + Gauss_interior + induction_interior
        
        # boundary loss based on boundary data I.E. the omniweb data set 5 mins
        indices = torch.randperm(total_tensor.size(0))[:batch_size]
        boundary_sample = total_tensor[indices]
        t_boundary = boundary_sample[:, 0:1].squeeze()
        x_boundary = boundary_sample[:,7:8].squeeze()
        y_boundary = boundary_sample[:,8:9].squeeze()
        z_boundary = boundary_sample[:,9:10].squeeze()
        X_BOUNDARY = torch.cat((x_boundary.unsqueeze(1), y_boundary.unsqueeze(1), t_boundary.unsqueeze(1), z_boundary.unsqueeze(1)), dim=1)
        
        Bx_boundary_value = Bx(X_BOUNDARY, Bx_params) - boundary_sample[:,1:2].squeeze()
        By_boundary_value = By(X_BOUNDARY, By_params) - boundary_sample[:,2:3].squeeze()
        Vx_boundary_value = Vx(X_BOUNDARY, Vx_params) - boundary_sample[:,3:4].squeeze()
        Vy_boundary_value = Vy(X_BOUNDARY, Vy_params) - boundary_sample[:,4:5].squeeze()
        rho_boundary_value = rho(X_BOUNDARY, rho_params) - boundary_sample[:,5:6].squeeze()
        p_boundary_value = p(X_BOUNDARY, p_params) - boundary_sample[:,6:7].squeeze()
        Bz_boundary_value = Bz(X_BOUNDARY, Bz_params) - boundary_sample[:,10:11].squeeze()
        Vz_boundary_value = Vz(X_BOUNDARY, Vz_params) - boundary_sample[:, 11:12].squeeze()
        
        boundary_sum = Bx_boundary_value + By_boundary_value + Vx_boundary_value + Vy_boundary_value + rho_boundary_value + p_boundary_value + Bz_boundary_value + Vz_boundary_value
        
        boundary = boundary_sum / 6
        
        loss = nn.MSELoss()
        weight_phys = 0.2
        physics_loss = loss(interior, torch.zeros_like(interior))
        bc_loss = loss(boundary, torch.zeros_like(boundary))
        loss_value = weight_phys * physics_loss + bc_loss
        
        return loss_value, physics_loss, bc_loss
    return loss_fn

## initialise loss and models
if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(random.randint(1, 50))

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=3)
    parser.add_argument("-d", "--dim-hidden", type=int, default=30)
    parser.add_argument("-b", "--batch-size", type=int, default=1000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-e", "--num-epochs", type=int, default=500)

    args = parser.parse_args()

    # configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = args.num_epochs
    tolerance = 1e-8
    learning_rate_0 = args.learning_rate
    domain_x = (df_val['X_(S/C)__GSE_Re'].min(), df_val['X_(S/C)__GSE_Re'].max())
    domain_y = (df_val['Y_(S/C)__GSE_Re'].min(), df_val['Y_(S/C)__GSE_Re'].max())
    domain_t = (df_val['time'].min(), df_val['time'].max())
    domain_z = (df_val['Z_(S/C)__GSE_Re'].min(), df_val['Z_(S/C)__GSE_Re'].max())

    # creates a list of models which all output their own parameter to make up the list of targets: bx, by, vx, vy, rho, p
    models = model_creation(inputs=4, num_hidden=num_hidden, dim_hidden=dim_hidden)
    
    functions_list = []
    params_list = []
    for mos in models:
        # initialise functions
        funcs = make_forward_fn(mos, derivative_order=1)
        # f = funcs[0], dfdx = funcs[1], dfdy = funcs[2], dfdt = funcs[3]
        functions_list.append(funcs)
        # initial parameters randomly initialized
        params_list.extend(list(mos.parameters()))
        #print(len(list(mos.parameters())))
    # print(len(params_list))
    # tuples_list = [tuple(params_list[i:i+10]) for i in range(0, len(params_list), 10)]
    # for i in tuples_list:
    #     print(len(i))
    loss_fn = make_loss_fn(functions_list)

    # choose optimizer with functional API using functorch
    optimizer =torch.optim.Adam(params_list, lr=learning_rate_0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


    
## running of the model
    # train the model to do: work on making so a loop utilises the params_list and loss function properly
    loss_evolution = []
    for i in range(num_iter):
        #manual lr dynamics
        # denominator = 1 + (i/20)
        # learning_rate = learning_rate_0/denominator
        # print(learning_rate)
        # optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))
        optimizer.zero_grad()
        # sample points in the domain randomly for each epoch
        #x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])
        x_dim = torch.FloatTensor(batch_size).uniform_(domain_x[0], domain_x[1]).requires_grad_(True)#.unsqueeze(1)
        y_dim = torch.FloatTensor(batch_size).uniform_(domain_y[0], domain_y[1]).requires_grad_(True)#.unsqueeze(1)
        t_dim = torch.FloatTensor(batch_size).uniform_(domain_t[0], domain_t[1]).requires_grad_(True)#.unsqueeze(1)
        z_dim = torch.FloatTensor(batch_size).uniform_(domain_z[0], domain_z[1]).requires_grad_(True)
        x = torch.cat((x_dim.unsqueeze(1), y_dim.unsqueeze(1), t_dim.unsqueeze(1), z_dim.unsqueeze(1)), dim=1)

        # compute the loss with the current parameters
        loss, p_loss, d_loss = loss_fn(params_list, x)
        # update the parameters with functional optimizer
        
        loss.backward()
        
        optimizer.step()
        scheduler.step(loss)

        #params_list = new_params_list

        print(f"Iteration {i} with loss {float(loss)} and physics loss {p_loss}, and data loss {d_loss}")
        loss_evolution.append(float(loss))
    
## plots
    n_points = 300  # Number of points in each dimension for x and y
    
    bound = 50
    x_start, x_end = -bound, bound  # Range for x
    y_start, y_end = -bound, bound  # Range for y
    #t_start, t_end = 0, 20000000  # Range for t
    t_simulation = random.uniform(0.0, 20000000.0)
    z_simulation = random.uniform(-5.0, 5.0)
    # Generate evenly spaced points for x and y dimensions
    x_points = torch.linspace(x_start, x_end, n_points)
    y_points = torch.linspace(y_start, y_end, n_points)
    # Create a 2D grid of points for x and y
    x_grid, y_grid = torch.meshgrid(x_points, y_points, indexing="xy")
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
    # Generate random values for the t dimension, scaled to the range [t_start, t_end]
    t_values = torch.full((xy_grid.shape[0], 1), t_simulation)
    z_values = torch.full((xy_grid.shape[0], 1), z_simulation)
    #t_values = torch.rand(xy_grid.shape[0], 1) * (t_end - t_start) + t_start
    # Concatenate the t dimension with the x and y grid points
    grid_points = torch.cat((xy_grid, t_values, z_values), dim=1)
    # Set requires_grad=True for the entire grid
    grid_points.requires_grad_(True)
    tuples_list = [tuple(params_list[i:i+10]) for i in range(0, len(params_list), 10)]
    
    B_x_func = functions_list[0][0]
    B_y_func = functions_list[1][0]
    B_x_tensor = B_x_func(grid_points, tuples_list[0])
    B_y_tensor = B_y_func(grid_points, tuples_list[1])
    B_mag_tensor = torch.sqrt(B_x_tensor ** 2 + B_y_tensor ** 2)
    x_tensor = grid_points[:, 0]
    y_tensor = grid_points[:, 1]
    t_tensor = grid_points[:, 2]

    # Convert tensors to NumPy arrays
    x_array = x_tensor.detach().numpy()
    y_array = y_tensor.detach().numpy()
    t_array = t_tensor.detach().numpy()
    B_array = B_mag_tensor.detach().numpy()
    
    fig, ax = plt.subplots()
    ax.semilogy(loss_evolution)
    ax.set(title="Loss evolution", xlabel="# epochs", ylabel="Loss")
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_array, y_array, c=B_array, cmap='viridis', s=1)  # Use a colormap of your choice, 'viridis' is just an example
    plt.colorbar(scatter, label='B map') 
    full_circle = patches.Circle((0, 0), radius=1, color='black')
    ax.add_patch(full_circle)
    semicircle_white = patches.Wedge((0, 0), 1, -90, 90, color='white')
    ax.add_patch(semicircle_white)
    ax.set(title="2D Map of B magnitude with mass continuity, Gauss and MSE of bcs", xlabel="X axis", ylabel="Y axis")
    ax.set_aspect('equal', 'box')
        
    plt.show()