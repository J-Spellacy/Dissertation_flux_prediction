## imports dependencies and dataframe
#%matplotlib ipympl
import torch
import pickle
import torch.nn as nn
from typing import Callable
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Support_funcs_one_model import make_forward_fn, LinearNN, model_creation, sig_figs, scale_sample, scale_sample_t, unscale, unscale_not_t, EarlyStopping, generate_points_on_sphere
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
from math import floor, log10
from matplotlib import animation

## sets constants (si i think)
R_E = 6371000.0 # si
D_p = 0.001 # dk
gamma  = 5.0 / 3.0 # non-dim
G = 6.67430e-11 # si
M = 5.972e24 # si (kg)
mu_0 = 4 * np.pi * 1e-7 # si
eta =  0.002 # si (maybe not)
tolerance = 1e-5 
M_m = 7.8e22 # si (approx value)
m = torch.tensor([0, 0, M_m], dtype=torch.float32)  # Dipole moment (pointing in z-direction) # approximation
mu_vis = 1.4e-3 # viscosity (approximation)

## dataframe solarwind x>24 Re
# df = pd.read_csv('C:/Users/User/Desktop/PINN_torch/PINN_data_OMNI_cutdown.csv', index_col=0)
df_val = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_megacut_si_units.csv')
data_numpy_arr = df_val.values
total_tensor = torch.from_numpy(data_numpy_arr).float()
df_val2 = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_cutdown_v_kmps.csv', index_col=0)

df_sw_val = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_omni_sw_lessthan_24.csv')
dataval_numpy_arr = df_sw_val.values
total_tensor_sw_val = torch.from_numpy(dataval_numpy_arr).float()

## dataframe for flux model
df_val_flux = pd.read_csv('C:/Users/User/Desktop/PINN_torch/forfluxmodelsi.csv') # goes dataframe with columns time flux (2 MeV)
df_val_flux_numpy_arr = df_val_flux.values
df_val_flux_tensor = torch.from_numpy(df_val_flux_numpy_arr).float()
## creates ionosphere L < 3.5 Re boundary condition
grid_size = 20
io_bound = 6 * R_E#3.5 * R_E
earth_radius_bound = R_E
range_x = (-io_bound, io_bound)
range_y = (-io_bound, io_bound)
range_z = (-io_bound, io_bound)

# Create a grid
x_iono = torch.linspace(range_x[0], range_x[1], grid_size)
y_iono = torch.linspace(range_y[0], range_y[1], grid_size)
z_iono = torch.linspace(range_z[0], range_z[1], grid_size)
X_io, Y_io, Z_io = torch.meshgrid(x_iono, y_iono, z_iono, indexing='ij')
# Calculate the magnetic field
coords_stacked = torch.stack([X_io, Y_io, Z_io], dim=-1)
# Reshape to [num_measurements, 3]
num_measurements = grid_size ** 3
coords_tensor = coords_stacked.view(num_measurements, 3)
# def magnetic_field(X, Y, Z, m):
#     r_vec = torch.stack((X, Y, Z), dim=-1)
#     r = torch.norm(r_vec, dim=-1)
#     r_hat = r_vec / r.unsqueeze(-1)
#     m_dot_r_hat = torch.sum(m * r_hat, dim=-1)
#     term1 = 3 * m_dot_r_hat.unsqueeze(-1) * r_hat
#     term2 = m.view(1, 1, 1, 3)
#     B = (mu_0 / (4 * np.pi * r.unsqueeze(-1)**3)) * (term1 - term2)
#     combined_mask = (r <= io_bound) & (r >= earth_radius_bound)
#     return B, combined_mask

def magnetic_field(r_vec, m, io_bound):
    # Compute the magnitude of position vectors
    r = torch.norm(r_vec, dim=-1)

    # Create a mask for points within the bounds
    mask = (r <= io_bound) & (r >= earth_radius_bound)

    # Filter points using the mask
    r_vec_filtered = r_vec[mask]

    # Compute magnetic field B for filtered points
    B_filtered = torch.zeros_like(r_vec_filtered)  # Initialize B_filtered

    # Compute magnetic field B only for filtered points
    for i in range(r_vec_filtered.shape[0]):
        # Compute unit vector for filtered point
        r_hat = r_vec_filtered[i] / r_vec_filtered[i].norm()

        # Compute dot product of magnetic moment vector and unit vector
        m_dot_r_hat = torch.dot(m, r_hat)

        # Compute terms of magnetic field equation
        term1 = 3 * m_dot_r_hat * r_hat
        term2 = m

        # Compute magnetic field B for the filtered point
        B_filtered[i] = (mu_0 / (4 * np.pi * r_vec_filtered[i].norm()**3)) * (term1 - term2)

    return B_filtered, r_vec_filtered, mask

B_io, pos_vector, mask_unused = magnetic_field(coords_tensor, m, io_bound=io_bound)

print(B_io.shape)
Bx_masked = B_io[:, 0]  # All elements in the last dimension, index 0
By_masked = B_io[:, 1]  # All elements in the last dimension, index 1
Bz_masked = B_io[:, 2]  # All elements in the last dimension, index 2

Bx_masked = Bx_masked.squeeze()  # shape: [num_filtered_points, 1]
By_masked = By_masked.squeeze()  # shape: [num_filtered_points, 1]
Bz_masked = Bz_masked.squeeze()

print("pos_vector shape:", pos_vector.shape)
print("Bx_masked shape:", Bx_masked.shape)
print("By_masked shape:", By_masked.shape)
print("Bz_masked shape:", Bz_masked.shape)

B_coords_values_msk = torch.cat([pos_vector, Bx_masked.unsqueeze(1), By_masked.unsqueeze(1), Bz_masked.unsqueeze(1)], dim=1)
# tensor with index 0: x, 1: y, 2: z, 3: bx, 4: by, 5: bz

## unused boundary conditions dataframes
df_BSN_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/BSN_data_si_units.csv', index_col=0)
BSN_data_numpy_arr = df_BSN_BC.values
BSN_BC_tensor = torch.from_numpy(BSN_data_numpy_arr).float()

df_CRRES_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/CRRES_GSE_si_units.csv', index_col=0)
CRS_data_numpy_arr = df_CRRES_BC.values
CRRES_BC_tensor = torch.from_numpy(CRS_data_numpy_arr).float()

df_GOES_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/combined_ephem_mag_si_units.csv', index_col=0)
GOES_data_numpy_arr = df_GOES_BC.values
G_BC_tensor = torch.from_numpy(GOES_data_numpy_arr).float()

##  set limits based on absolute maximum of omni data
lim_df = df_val.abs()
x_lim = 1.684747e+09 # m
y_lim = 2.022856e+09 # m (biggest scale length)
z_lim = 2.018333e+08 # m
x_bound = (-24 * R_E, 24 * R_E) # m
y_bound = (-24 * R_E, 24 * R_E) # m
z_bound = (-24 * R_E, 24 * R_E) # m
t_lim = 1.046453e+09 # s
bx_lim = 4.4183e-05 # T
by_lim = 4.4183e-05 # T
bz_lim = 5.9763e-05 # T
vx_lim = 1.127767e+06 # m/s
vy_lim = 4.411000e+05 # m/s
vz_lim = 4.273000e+05 # m/s
rho_lim = 1.204571e-19 #kg/m *** 3
P_lim = 8.285250e-08 #Pa


## loss definition TO DO: 

def make_loss_fn(funcs: list, weight_phys: float) -> Callable:
    def loss_fn(params: tuple, x: torch.Tensor):
        
        loss = nn.MSELoss()
        
        selected_elements = x[:, [0, 1, 3]]
        magnitudes = torch.sqrt((selected_elements ** 2).sum(dim=1))

        # Define a threshold for the magnitude
        epsilon = 6.0 * R_E / y_bound[1]
        epsilon_const_condition = 3.5 * R_E / y_bound[1]
        #print(float(epsilon), float(epsilon_const_condition))
        # Create a mask where the magnitude is below the threshold
        mask = magnitudes < epsilon
        # Use the mask to select rows from the original tensor
        x_E_Boundary = x[mask]

        x_interior = x[~mask]

        batch_size_in = x_interior.shape[0]
        
        selected_elements_2 = x_E_Boundary[:, [0, 1, 3]]
        magnitudes_2 = torch.sqrt((selected_elements_2 ** 2).sum(dim=1))
        mask2 = magnitudes_2 < epsilon_const_condition
        
        x_inner_boundary = x_E_Boundary[mask2]
        #print(f"1: {x_interior.shape}, 2: {x_E_Boundary.shape}, 3: {x_inner_boundary.shape}")
        #x_E_Boundary = x_E_Boundary[~mask2]
        
### initialise values and main derivatives
        # interior loss
        F = funcs[0]       
        
        F_tensor = F(x_interior, params)
        
        Bx_value = F_tensor[:,0:1].squeeze()
        By_value = F_tensor[:,1:2].squeeze()
        Vx_value = F_tensor[:,2:3].squeeze()
        Vy_value = F_tensor[:,3:4].squeeze()
        rho_value = F_tensor[:,4:5].squeeze()
        p_value = F_tensor[:,5:6].squeeze()
        Bz_value = F_tensor[:,6:7].squeeze()
        Vz_value = F_tensor[:,7:8].squeeze()
        
        dVxdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = Vx_value,
            grad_outputs=torch.ones_like(Vx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        drhodX = torch.autograd.grad(
            inputs = x_interior,
            outputs = rho_value,
            grad_outputs=torch.ones_like(rho_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVydX = torch.autograd.grad(
            inputs = x_interior,
            outputs = Vy_value,
            grad_outputs=torch.ones_like(Vy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBxdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = Bx_value,
            grad_outputs=torch.ones_like(Bx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBydX = torch.autograd.grad(
            inputs = x_interior,
            outputs = By_value,
            grad_outputs=torch.ones_like(By_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dpdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = p_value,
            grad_outputs=torch.ones_like(p_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVzdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = Vz_value,
            grad_outputs=torch.ones_like(Vz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBzdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = Bz_value,
            grad_outputs=torch.ones_like(Bz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVydx_value = dVydX[:, 0]
        dVydy_value = dVydX[:, 1]
        dVydt_value = dVydX[:, 2]
        dVydz_value = dVydX[:, 3]
        dVxdx_value = dVxdX[:, 0]
        dVxdy_value = dVxdX[:, 1]
        dVxdt_value = dVxdX[:, 2]
        dVxdz_value = dVxdX[:, 3]
        drhodx_value = drhodX[:, 0]
        drhody_value = drhodX[:, 1]
        drhodt_value = drhodX[:, 2]
        drhodz_value = drhodX[:, 3]
        dBxdx_value = dBxdX[:, 0]
        dBxdt_value = dBxdX[:, 2]
        dBydy_value = dBydX[:, 1]
        dBydt_value = dBydX[:, 2]
        dpdx_value = dpdX[:, 0]
        dpdy_value = dpdX[:, 1]
        dpdt_value = dpdX[:, 2]
        dpdz_value = dpdX[:, 3]
        dVzdx_value = dVzdX[:, 0]
        dVzdy_value = dVzdX[:, 1]
        dVzdz_value = dVzdX[:, 3]
        dVzdt_value = dVzdX[:, 2]
        dBzdt_value = dBzdX[:, 2]
        dBzdz_value = dBzdX[:, 3]
        
        # New residuals:
### new mass continuity residual
        Vxdrhodx_value = drhodx_value * Vx_value
        rhodVxdx_value =  dVxdx_value * rho_value
        drhoVxdx_value = Vxdrhodx_value + rhodVxdx_value
        
        Vydrhody_value = drhody_value * Vy_value
        rhodVydy_value =  dVydy_value * rho_value
        drhoVydy_value = Vydrhody_value + rhodVydy_value
        
        Vzdrhodz_value = drhodz_value * Vz_value
        rhodVzdz_value = dVzdz_value * rho_value
        drhoVzdz_value = Vzdrhodz_value + rhodVzdz_value
        
        div_rhoV_value = drhoVxdx_value + drhoVydy_value + drhoVzdz_value
        
        grad2_rhoxdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = drhodx_value,
            grad_outputs=torch.ones_like(drhodx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_rhoydX = torch.autograd.grad(
            inputs = x_interior,
            outputs = drhody_value,
            grad_outputs=torch.ones_like(drhody_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_rhozdX  = torch.autograd.grad(
            inputs = x_interior,
            outputs = drhodz_value,
            grad_outputs=torch.ones_like(drhodz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_rhoxdx = grad2_rhoxdX[:, 0]
        grad2_rhoydy = grad2_rhoydX[:, 1]
        grad2_rhozdz = grad2_rhozdX[:, 3]
        
        divgrad_rhodX = grad2_rhoxdx + grad2_rhoydy + grad2_rhozdz
        
        new_mc_interior = drhodt_value + div_rhoV_value  - D_p  * divgrad_rhodX
        new_mc_loss = loss(new_mc_interior, torch.zeros_like(new_mc_interior))

### new momentum residual:
        
        # (v dot del)v
        dir_der_Vx = Vx_value * dVxdx_value + Vy_value * dVxdy_value + Vz_value * dVxdz_value
        dir_der_Vy = Vx_value * dVydx_value + Vy_value * dVydy_value + Vz_value * dVydz_value
        dir_der_Vz = Vx_value * dVzdx_value + Vy_value * dVzdy_value + Vz_value * dVzdz_value
        
        # if you have time generate B_earth at the x_interior positions and minus it below
        
        Bdiffx_value  = Bx_value
        Bdiffy_value  = By_value
        Bdiffz_value  = Bz_value
        
        
        dBdiffxdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = Bdiffx_value,
            grad_outputs=torch.ones_like(Bdiffx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBdiffydX = torch.autograd.grad(
            inputs = x_interior,
            outputs = Bdiffy_value,
            grad_outputs=torch.ones_like(Bdiffy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBdiffzdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = Bdiffz_value,
            grad_outputs=torch.ones_like(Bdiffz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBdiffxdx = dBdiffxdX[:,  0]
        dBdiffxdy = dBdiffxdX[:,  1]
        dBdiffxdz = dBdiffxdX[:,  3]
        dBdiffydx = dBdiffydX[:,  0]
        dBdiffydy = dBdiffydX[:,  1]
        dBdiffydz = dBdiffydX[:,  3]
        dBdiffzdx = dBdiffzdX[:,  0]
        dBdiffzdy = dBdiffzdX[:,  1]
        dBdiffzdz = dBdiffzdX[:,  3]
        
        jx_value = (dBdiffzdy - dBdiffydz)
        jy_value = (dBdiffxdz - dBdiffzdx)
        jz_value = (dBdiffydx - dBdiffxdy)
        
        jcrossB_x = (jy_value * Bz_value) - (jz_value * By_value)
        jcrossB_y = (jz_value * Bx_value) - (jx_value * Bz_value)
        jcrossB_z = (jx_value * By_value) - (jy_value * Bx_value)

        def gravitational_field_torch(coords):
            # Extract x, y, z coordinates from the tensor; ignoring the time component 't'
            x, y, _, z = coords.t()  # Transpose and unpack the coordinates
            
            # Compute the vector r from the Earth's center to the points and its magnitude r
            r = torch.stack((x, y, z), dim=1)
            r_mag = torch.norm(r, dim=1, keepdim=True)  
            # Compute the gravitational field vector g
            g = -G * M / r_mag**3
            g_vector = g * r 
            
            return g_vector
        
        # this scaling is a bit shit but dont know what else to do
        G_vector = gravitational_field_torch(x_interior)
        g_vector_x = G_vector[:, 0:1] / torch.max(G_vector[:, 0:1])
        g_vector_y = G_vector[:, 1:2] / torch.max(G_vector[:, 0:1])
        g_vector_z = G_vector[:, 2:3] / torch.max(G_vector[:, 0:1])
        
        grad2_VxdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = dVxdx_value,
            grad_outputs=torch.ones_like(dVxdx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_VydX = torch.autograd.grad(
            inputs = x_interior,
            outputs = dVydy_value,
            grad_outputs=torch.ones_like(dVydy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_VzdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = dVzdz_value,
            grad_outputs=torch.ones_like(dVzdz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_Vxdx = grad2_VxdX[:, 0]
        grad2_Vxdy = grad2_VxdX[:, 1]
        grad2_Vxdz = grad2_VxdX[:, 3]
        grad2_Vydx = grad2_VydX[:, 0]
        grad2_Vydy = grad2_VydX[:, 1]
        grad2_Vydz = grad2_VydX[:, 3]
        grad2_Vzdx = grad2_VzdX[:, 0]
        grad2_Vzdy = grad2_VzdX[:, 1]
        grad2_Vzdz = grad2_VzdX[:, 3]
        
        phi_x = (grad2_Vxdx + grad2_Vxdy + grad2_Vxdz) * mu_vis
        phi_y = (grad2_Vydx + grad2_Vydy + grad2_Vydz) * mu_vis
        phi_z = (grad2_Vzdx + grad2_Vzdy + grad2_Vzdz) * mu_vis
        
        rho_mean = torch.mean(rho_value)
        
        new_mo_interior_x = dVxdt_value + dir_der_Vx + (dpdx_value / rho_mean) - (jcrossB_x / rho_mean) - g_vector_x - (phi_x / rho_mean)## do next parts also add b-earth and change coordinates
        new_mo_interior_y = dVydt_value + dir_der_Vy + (dpdy_value / rho_mean) - (jcrossB_y / rho_mean) - g_vector_y - (phi_y / rho_mean)
        new_mo_interior_z = dVzdt_value + dir_der_Vz + (dpdz_value / rho_mean) - (jcrossB_z / rho_mean) - g_vector_z - (phi_z / rho_mean)
        
        #print(f"1: {float(torch.max(dVxdt_value))} {float(torch.min(dVxdt_value))} 2: {float(torch.max(dir_der_Vx))} {float(torch.min(dir_der_Vx))} 3: {float(torch.max(dpdx_value))} {float(torch.min(dpdx_value))} 4: {float(torch.max(jcrossB_x))} {float(torch.min(jcrossB_x))}  5: {float(torch.max(g_vector_x))} {float(torch.min(g_vector_x))} 6: {float(torch.max(phi_x))} {float(torch.min(phi_x))}")
        new_mo_interior = new_mo_interior_x +  new_mo_interior_y + new_mo_interior_z
        #print(f"dvdt {torch.mean(dVxdt_value)}, dir_der_V {torch.mean(dir_der_Vx)}, p_term {torch.mean((dpdx_value / rho_value))} jcrossb term {torch.mean((jcrossB_x / rho_value))}, g_vector {torch.mean(g_vector_x)}, phi term {torch.mean((phi_x / rho_value))}")
        new_mo_loss = loss(new_mo_interior, torch.zeros_like(new_mo_interior))
        
### new pressure gradient residual
        
        dir_der_p = Vx_value * dpdx_value + Vy_value * dpdy_value + Vz_value * dpdz_value
        
        div_V = dVxdx_value + dVydy_value + dVzdz_value
        
        gp_div_v = gamma * p_value * div_V
        
        grad2_pxdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = dpdx_value,
            grad_outputs=torch.ones_like(dpdx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_pydX = torch.autograd.grad(
            inputs = x_interior,
            outputs = dpdy_value,
            grad_outputs=torch.ones_like(dpdy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_pzdX  = torch.autograd.grad(
            inputs = x_interior,
            outputs = dpdz_value,
            grad_outputs=torch.ones_like(dpdz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_pxdx = grad2_pxdX[:, 0]
        grad2_pydy = grad2_pydX[:, 1]
        grad2_pzdz = grad2_pzdX[:, 3]
        
        divgrad_pdX = grad2_pxdx + grad2_pydy + grad2_pzdz
        
        new_pg_interior = dpdt_value + dir_der_p + gp_div_v - D_p * divgrad_pdX
        new_pg_loss = loss(new_pg_interior, torch.zeros_like(new_pg_interior))
        
### new field gradient residual
        
        # v cross b
        VcrossB_x = (Vy_value * Bz_value) - (Vz_value * By_value)
        VcrossB_y = (Vz_value * Bx_value) - (Vx_value * Bz_value)
        VcrossB_z = (Vx_value * By_value) - (Vy_value * Bx_value)
        
        dVcrossB_xdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = VcrossB_x,
            grad_outputs=torch.ones_like(VcrossB_x),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVcrossB_ydX = torch.autograd.grad(
            inputs = x_interior,
            outputs = VcrossB_y,
            grad_outputs=torch.ones_like(VcrossB_y),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dVcrossB_zdX = torch.autograd.grad(
            inputs = x_interior,
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
        
        grad2_BxdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = dBxdx_value,
            grad_outputs=torch.ones_like(dBxdx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_BydX = torch.autograd.grad(
            inputs = x_interior,
            outputs = dBydy_value,
            grad_outputs=torch.ones_like(dBydy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_BzdX = torch.autograd.grad(
            inputs = x_interior,
            outputs = dBzdz_value,
            grad_outputs=torch.ones_like(dBzdz_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_Bxdx = grad2_BxdX[:, 0]
        grad2_Bxdy = grad2_BxdX[:, 1]
        grad2_Bxdz = grad2_BxdX[:, 3]
        grad2_Bydx = grad2_BydX[:, 0]
        grad2_Bydy = grad2_BydX[:, 1]
        grad2_Bydz = grad2_BydX[:, 3]
        grad2_Bzdx = grad2_BzdX[:, 0]
        grad2_Bzdy = grad2_BzdX[:, 1]
        grad2_Bzdz = grad2_BzdX[:, 3]
        
        new_fg_interior_x = dBxdt_value - curl_VcrossB_x - eta * (grad2_Bxdx + grad2_Bxdy + grad2_Bxdz)
        new_fg_interior_y = dBydt_value - curl_VcrossB_y - eta * (grad2_Bydx + grad2_Bydy + grad2_Bydz)
        new_fg_interior_z = dBzdt_value - curl_VcrossB_z - eta * (grad2_Bzdx + grad2_Bzdy + grad2_Bzdz)
        
        new_fg_interior = new_fg_interior_x + new_fg_interior_y + new_fg_interior_z
        new_fg_loss = loss(new_fg_interior, torch.zeros_like(new_fg_interior))
        
### Div free B residual
        div_B = dBxdx_value + dBydy_value + dBzdz_value
        div_free_loss = loss(div_B, torch.zeros_like(div_B))
        
### OMNI boundary loss based on boundary data I.E. the omniweb data set 5 mins
        t_interior_scaled = x_interior[:,2:3].squeeze()
        t_interior = t_interior_scaled * t_lim
        sw_size = 200
        sw_indices = torch.randperm(total_tensor.size(0))[:sw_size]
        boundary_sample = total_tensor[sw_indices]
        
        time_tensor_fl_bound = boundary_sample[:,0:1].squeeze() / t_lim
        # the input solar wind bc at x = 1 with y and z as uniform points over that position
        # y_boundary = torch.linspace(-1.0, 1.0, batch_size)
        # z_boundary = torch.linspace(-1.0, 1.0, batch_size)
        # x_boundary = torch.ones(batch_size)
        bc_loss = loss(torch.zeros(sw_size), torch.zeros(sw_size))
        for row in boundary_sample:
            y_boundary = torch.linspace(-1.0, 1.0, sw_size)
            z_boundary = torch.linspace(-1.0, 1.0, sw_size)
            x_boundary = torch.ones(sw_size)
            t_boundary = torch.full_like(y_boundary, row[0] / t_lim)
            X_BOUNDARY = torch.cat((x_boundary.unsqueeze(1), y_boundary.unsqueeze(1), t_boundary.unsqueeze(1), z_boundary.unsqueeze(1)), dim=1)
            F_boundary_tensor = F(X_BOUNDARY, params)
            bx_boundary = torch.full_like(y_boundary,row[1] / bx_lim)
            by_boundary = torch.full_like(y_boundary,row[2] / by_lim)
            vx_boundary = torch.full_like(y_boundary,row[3] / vx_lim)
            vy_boundary = torch.full_like(y_boundary,row[4] / vy_lim)
            rho_boundary = torch.full_like(y_boundary,row[5] / rho_lim)
            p_boundary = torch.full_like(y_boundary,row[6] / P_lim)
            bz_boundary = torch.full_like(y_boundary,row[10] / bz_lim)
            vz_boundary = torch.full_like(y_boundary,row[11] / vz_lim)
            boundary_sample_tensor = torch.cat((bx_boundary.unsqueeze(1), by_boundary.unsqueeze(1), vx_boundary.unsqueeze(1), vy_boundary.unsqueeze(1), rho_boundary.unsqueeze(1),  p_boundary.unsqueeze(1), bz_boundary.unsqueeze(1), vz_boundary.unsqueeze(1)), dim=1)
            bc_loss_i = loss(F_boundary_tensor, boundary_sample_tensor)
            bc_loss = bc_loss + bc_loss_i
        
            
        # y_boundary = boundary_sample[:,8:9].squeeze() / y_bound[1]
        # z_boundary = boundary_sample[:,9:10].squeeze() / z_bound[1]
        # x_boundary = boundary_sample[:,7:8].squeeze() / x_bound[1]
        # X_BOUNDARY = torch.cat((x_boundary.unsqueeze(1), y_boundary.unsqueeze(1), (boundary_sample[:,0:1].squeeze() / t_lim).unsqueeze(1), z_boundary.unsqueeze(1)), dim=1)
        
        # F_boundary_tensor = F(X_BOUNDARY, params) # shape ([300, 4])
        
        # boundary_sample_tensor = torch.cat((boundary_sample[:,1:2] / bx_lim, boundary_sample[:,2:3] / by_lim, boundary_sample[:,3:4] / vx_lim, boundary_sample[:,4:5] / vy_lim, boundary_sample[:,5:6] / rho_lim,  boundary_sample[:,6:7] / P_lim, boundary_sample[:,10:11] / bz_lim, boundary_sample[:, 11:12] / vz_lim), dim=1)
        
        # bc_loss = loss(F_boundary_tensor, boundary_sample_tensor)
        
###  GOES magnometer boundary condition  
        
        G_indices = torch.randperm(G_BC_tensor.size(0))[:batch_size]
        G_boundary_sample = G_BC_tensor[G_indices]
        t_Gboundary = G_boundary_sample[:, 0:1].squeeze() / t_lim
        x_Gboundary = G_boundary_sample[:,1:2].squeeze() / y_lim
        y_Gboundary = G_boundary_sample[:,2:3].squeeze() / y_lim
        z_Gboundary = G_boundary_sample[:,3:4].squeeze() / y_lim
        X_GBOUNDARY = torch.cat((x_Gboundary.unsqueeze(1), y_Gboundary.unsqueeze(1), t_Gboundary.unsqueeze(1), z_Gboundary.unsqueeze(1)), dim=1)
        #print(torch.mean(x_Gboundary))
        F_Gboundary_tensor = F(X_GBOUNDARY, params)
        
        G_boundary_sample_tensor = torch.cat((G_boundary_sample[:,4:5] / bx_lim, G_boundary_sample[:,5:6] / by_lim, G_boundary_sample[:,6:7] / bz_lim),  dim=1)
        G_boundary_result_tensor = torch.cat((F_Gboundary_tensor[:,0:1],F_Gboundary_tensor[:,1:2], F_Gboundary_tensor[:,6:7]), dim=1)
        G_bc_loss = loss(G_boundary_result_tensor, G_boundary_sample_tensor)
        
### Earth boundary condition residual
        io_batch_size = int(floor(B_coords_values_msk.size(0) / 1.5))
        B_indices = torch.randperm(B_coords_values_msk.size(0))[:io_batch_size]
        B_msk_rand = B_coords_values_msk[B_indices]
        x_io_boundary, y_io_boundary, z_io_boundary = B_msk_rand[:,0:1].squeeze() / x_bound[1], B_msk_rand[:,1:2].squeeze() / y_bound[1], B_msk_rand[:,2:3].squeeze() / z_bound[1]
        t_io_ini = torch.FloatTensor(B_msk_rand.shape[0]).uniform_(0.0, 1.0).requires_grad_(True)
        X_iono_boundary = torch.cat((x_io_boundary.unsqueeze(1), y_io_boundary.unsqueeze(1), t_io_ini.unsqueeze(1), z_io_boundary.unsqueeze(1)), dim=1)
        B_tensor_io = torch.cat(((B_msk_rand[:,3:4].squeeze() / bx_lim).unsqueeze(1), (B_msk_rand[:,4:5].squeeze() / by_lim).unsqueeze(1), (B_msk_rand[:,5:6].squeeze() / bz_lim).unsqueeze(1)), dim=1)
        
        iono_F = F(X_iono_boundary, params)
        # print(f"x_max: {float(torch.max(x_io_boundary))}, x_min: {float(torch.min(x_io_boundary))}")
        # print(f"y_max: {float(torch.max(y_io_boundary))}, y_min: {float(torch.min(y_io_boundary))}")
        # print(f"z_max: {float(torch.max(z_io_boundary))}, z_min: {float(torch.min(z_io_boundary))}")
        iono_F_Btensor = torch.cat((iono_F[:,0:1].squeeze().unsqueeze(1), iono_F[:,1:2].squeeze().unsqueeze(1), iono_F[:,6:7].squeeze().unsqueeze(1)), dim=1)
        
        IO_loss = loss(iono_F_Btensor, B_tensor_io)
        
### no grad BC
        batch_no_grad = int(floor(batch_size / 4))
        y_boundary_ngy1 = torch.full((batch_no_grad,), 1.0)
        z_boundary_ngy1 = torch.linspace(-1.0, 1.0, batch_no_grad).requires_grad_(True)
        x_boundary_ngy1 = torch.linspace(-1.0, 1.0, batch_no_grad).requires_grad_(True)
        t_boundary_ngy = torch.FloatTensor(batch_no_grad).uniform_(0.0, 1.0).requires_grad_(True)
        y_boundary_ngy2 = torch.full((batch_no_grad,), -1.0)
        
        X_BOUNDARY_ngy1 = torch.cat((x_boundary_ngy1.unsqueeze(1), y_boundary_ngy1.unsqueeze(1), t_boundary_ngy.unsqueeze(1), z_boundary_ngy1.unsqueeze(1)), dim=1)
        X_BOUNDARY_ngy2 = torch.cat((x_boundary_ngy1.unsqueeze(1), y_boundary_ngy2.unsqueeze(1), t_boundary_ngy.unsqueeze(1), z_boundary_ngy1.unsqueeze(1)), dim=1)
        
        X_BOUNDARY_ngy = torch.cat((X_BOUNDARY_ngy1, X_BOUNDARY_ngy2), dim=0)
        X_BOUNDARY_ngy.requires_grad_(True)
        F_ngy = F(X_BOUNDARY_ngy, params)
        
        dFngydX = torch.autograd.grad(
            inputs = X_BOUNDARY_ngy,
            outputs = F_ngy,
            grad_outputs=torch.ones_like(F_ngy),
            retain_graph=True, 
            create_graph=True
        )[0]
        dFngydy = dFngydX[:, 1]
        y_bc_loss = loss(dFngydy, torch.zeros_like(dFngydy))
        
        z_boundary_ngz1 = torch.full((batch_no_grad,), 1.0)
        y_boundary_ngz1 = torch.linspace(-1.0, 1.0, batch_no_grad).requires_grad_(True)
        z_boundary_ngz2 = torch.full((batch_no_grad,), -1.0)
        
        X_BOUNDARY_ngz1 = torch.cat((x_boundary_ngy1.unsqueeze(1), y_boundary_ngz1.unsqueeze(1), t_boundary_ngy.unsqueeze(1), z_boundary_ngz1.unsqueeze(1)), dim=1)
        X_BOUNDARY_ngz2 = torch.cat((x_boundary_ngy1.unsqueeze(1), y_boundary_ngz1.unsqueeze(1), t_boundary_ngy.unsqueeze(1), z_boundary_ngz2.unsqueeze(1)), dim=1)
        
        X_BOUNDARY_ngz = torch.cat((X_BOUNDARY_ngz1, X_BOUNDARY_ngz2), dim=0)
        X_BOUNDARY_ngz.requires_grad_(True)
        F_ngz = F(X_BOUNDARY_ngz, params)
        
        dFngzdX = torch.autograd.grad(
            inputs = X_BOUNDARY_ngz,
            outputs = F_ngz,
            grad_outputs=torch.ones_like(F_ngz),
            retain_graph=True, 
            create_graph=True
        )[0]
        dFngzdz = dFngzdX[:, 3]
        z_bc_loss = loss(dFngzdz, torch.zeros_like(dFngzdz))
        
        x_boundary_ngx1 = torch.full((batch_no_grad,), -1.0)
        X_BOUNDARY_ngx1 = torch.cat((x_boundary_ngx1.unsqueeze(1), y_boundary_ngz1.unsqueeze(1), t_boundary_ngy.unsqueeze(1), z_boundary_ngy1.unsqueeze(1)), dim=1)
        X_BOUNDARY_ngx1.requires_grad_(True)
        F_ngx = F(X_BOUNDARY_ngx1, params)
        
        dFngxdX = torch.autograd.grad(
            inputs = X_BOUNDARY_ngx1,
            outputs = F_ngx,
            grad_outputs=torch.ones_like(F_ngx),
            retain_graph=True, 
            create_graph=True
        )[0]
        dFngxdx = dFngxdX[:, 0]
        x_bc_loss = loss(dFngxdx, torch.zeros_like(dFngxdx))
        
        nograd_bc = x_bc_loss + y_bc_loss + z_bc_loss
        
### bow shock nose BC
        BSN_indices_BC = torch.randperm(BSN_BC_tensor.size(0))[:batch_size]
        BSN_boundary_sample_BC = BSN_BC_tensor[BSN_indices_BC]
        x_bsn = BSN_boundary_sample_BC[:,1:2].squeeze() / y_lim
        y_bsn = BSN_boundary_sample_BC[:,2:3].squeeze() / y_lim
        z_bsn = BSN_boundary_sample_BC[:,3:4].squeeze() / y_lim
        time_bsn = BSN_boundary_sample_BC[:,0:1].squeeze() / t_lim
        X_bsn = torch.cat((x_bsn.unsqueeze(1), y_bsn.unsqueeze(1), time_bsn.unsqueeze(1), z_bsn.unsqueeze(1)), dim=1)
        
        F_bsn = F(X_bsn, params)
        
        bsn_vels = torch.cat((F_bsn[:,2:3].unsqueeze(1),F_bsn[:,3:4].unsqueeze(1),F_bsn[:,7:8].unsqueeze(1)), dim=1)
        bsn_loss = loss(bsn_vels, torch.zeros_like(bsn_vels))
        
### CRRES validation
        CRRES_indices_BC = torch.randperm(CRRES_BC_tensor.size(0))[:batch_size]
        CRRES_boundary_sample_BC = CRRES_BC_tensor[CRRES_indices_BC]
        x_CRS = CRRES_boundary_sample_BC[:,1:2].squeeze() / x_bound[1]
        y_CRS = CRRES_boundary_sample_BC[:,2:3].squeeze() / y_bound[1]
        z_CRS = CRRES_boundary_sample_BC[:,3:4].squeeze() / z_bound[1]
        time_CRS = CRRES_boundary_sample_BC[:,0:1].squeeze() / t_lim
        B_CRS = CRRES_boundary_sample_BC[:,4:5].squeeze() / bx_lim
        X_CRS = torch.cat((x_CRS.unsqueeze(1), y_CRS.unsqueeze(1), time_CRS.unsqueeze(1), z_CRS.unsqueeze(1)), dim=1)
        F_CRS = F(X_CRS, params)
        
        CRS_FB = torch.sqrt((F_CRS[:,0:1] ** 2) + (F_CRS[:,1:2] ** 2) + (F_CRS[:,6:7] ** 2))
        CRS_loss = loss(CRS_FB.squeeze(), B_CRS.squeeze())
        
### OMNI validation
        OMNI_indices_val = torch.randperm(total_tensor_sw_val.size(0))[:batch_size]
        OMNI_boundary_sample_val = total_tensor_sw_val[OMNI_indices_val]
        x_OMNI_val = OMNI_boundary_sample_val[:,7:8].squeeze() / x_bound[1]
        y_OMNI_val = OMNI_boundary_sample_val[:,8:9].squeeze() / y_bound[1]
        z_OMNI_val = OMNI_boundary_sample_val[:,9:10].squeeze() / z_bound[1]
        time_OMNI_val = OMNI_boundary_sample_val[:,0:1].squeeze() / t_lim
        
        X_OMNI_val = torch.cat((x_OMNI_val.unsqueeze(1), y_OMNI_val.unsqueeze(1), time_OMNI_val.unsqueeze(1), z_OMNI_val.unsqueeze(1)), dim=1)
        F_OMNI_val = F(X_OMNI_val, params)
        
        val_tensor_FO = torch.cat((OMNI_boundary_sample_val[:,1:2] / bx_lim, OMNI_boundary_sample_val[:,2:3] / by_lim, OMNI_boundary_sample_val[:,3:4] / vx_lim, OMNI_boundary_sample_val[:,4:5] / vy_lim, OMNI_boundary_sample_val[:,5:6] / rho_lim,  OMNI_boundary_sample_val[:,6:7] / P_lim, OMNI_boundary_sample_val[:,10:11] / bz_lim, OMNI_boundary_sample_val[:, 11:12] / vz_lim), dim=1)
        
        OMNI_val_loss = loss(F_OMNI_val, val_tensor_FO)
        
### experimental zero v condition
        def generate_points_on_sphere(num_points, radius):
            # Generate random spherical coordinates
            theta = torch.acos(2 * torch.rand(num_points) - 1)  # Polar angle (0 to pi)
            phi = 2 * torch.pi * torch.rand(num_points)  # Azimuthal angle (0 to 2pi)

            # Convert spherical coordinates to Cartesian coordinates
            x = radius * torch.sin(theta) * torch.cos(phi)
            y = radius * torch.sin(theta) * torch.sin(phi)
            z = radius * torch.cos(theta)

            # Stack the coordinates to create the tensor
            points = torch.stack((x, y, z), dim=1)

            return points
        
        earth_points = generate_points_on_sphere(batch_size, 3.5/24)
        x_earth = earth_points[:,0:1].squeeze()
        y_earth = earth_points[:,1:2].squeeze()
        z_earth = earth_points[:,2:3].squeeze()
        t_earth = torch.FloatTensor(batch_size).uniform_(0.0, 1.0).requires_grad_(True)
        earth_inp = torch.cat((x_earth.unsqueeze(1), y_earth.unsqueeze(1), t_earth.unsqueeze(1), z_earth.unsqueeze(1)), dim=1)
        F_earth_exp = F(earth_inp, params)
        vx_earth = F_earth_exp[:,2:3].squeeze()
        vy_earth = F_earth_exp[:,3:4].squeeze()
        vz_earth = F_earth_exp[:,7:8].squeeze()
        
        earth_test = torch.cat((vx_earth.unsqueeze(1), vy_earth.unsqueeze(1), vz_earth.unsqueeze(1)), dim=1)
        
        exp_earth_loss = loss(earth_test, torch.zeros_like(earth_test))
        
### loss final calculation
        physics_loss =  new_mo_loss + new_pg_loss + new_fg_loss + div_free_loss + new_mc_loss 
        data_loss = bc_loss + IO_loss + exp_earth_loss+ CRS_loss + OMNI_val_loss + nograd_bc
        loss_value = weight_phys * physics_loss + data_loss
        ###### TO DO: check if removing certain BC allows for the CRRES loss to converge? maybe rework the ionosphere residual
        
        return loss_value, physics_loss, data_loss, bc_loss, IO_loss, nograd_bc, G_bc_loss, bsn_loss, CRS_loss, new_mc_loss, new_mo_loss, new_pg_loss, new_fg_loss, div_free_loss, OMNI_val_loss
    return loss_fn

## initialise loss and models
if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(24)

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=8)
    parser.add_argument("-d", "--dim-hidden", type=int, default=64)
    parser.add_argument("-b", "--batch-size", type=int, default=4000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=2e-2)
    parser.add_argument("-e", "--num-epochs", type=int, default=1000)

    args = parser.parse_args()
    
    # configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = args.num_epochs
    learning_rate_0 = args.learning_rate
    domain_x = (-y_lim, y_lim)
    domain_y = (-y_lim, y_lim)
    domain_t = (-y_lim, y_lim)
    domain_z = (-y_lim, y_lim)

    # creates a list of models which all output their own parameter to make up the list of targets: bx, by, vx, vy, rho, p
    model = LinearNN(num_inputs=4, num_layers=num_hidden, num_neurons=dim_hidden)
    

    # initialise functions
    funcs = make_forward_fn(model, derivative_order=1)
    params = tuple(model.parameters())
    loss_fn = make_loss_fn(funcs, 1.0)

    # choose optimizer with functional API using functorch
    optimizer =torch.optim.Adam(params, lr=learning_rate_0, eps=1e-6)
    early_stopping = EarlyStopping(tolerance=10, min_delta = 0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=30)
    

    
## running of the model
    # train the model to do: work on making so a loop utilises the params_list and loss function properly
    loss_evolution = []
    data_l_evolution = []
    phys_l_evolution = []
    omni_bc_evo = []
    Earth_bc_evo = []
    no_grad_bc_evo = []
    bsn_bc_evo = []
    CRS_bc_evo = []
    G_bc_evo = []
    mc_evo = []
    mo_evo = []
    pg_evo = []
    fg_evo = []
    Maxwell_evo = []
    OMNI_val_evo = []
    # print(params)
    
    ## add a section here for initial run then add feedback lag layers in the following
    for i in range(num_iter):
        optimizer.zero_grad(set_to_none=False)
        # sample points in the domain randomly for each epoch
        x_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)
        y_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)
        t_dim = torch.FloatTensor(batch_size).uniform_(0.0, 1.0).requires_grad_(True)
        z_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)
        x = torch.cat((x_dim.unsqueeze(1), y_dim.unsqueeze(1), t_dim.unsqueeze(1), z_dim.unsqueeze(1)), dim=1)
        
        # compute the loss with the current parameters
        loss, p_loss, d_loss, bc_loss, Earth_bc_loss, nograd_loss, G_bc_loss, bsn_loss, CRS_loss, mc_loss, mo_loss, pg_loss, fg_loss, max_loss, OMNI_val_loss = loss_fn(params, x)
        # update the parameters with functional optimizer
        
        p_loss.backward()
        d_loss.backward()
        #loss.backward()
        
        optimizer.step()
        #scheduler.step(loss)
            
        #params_list = new_params_list

        #print(f"Iteration {i} with loss {sig_figs(float(loss), 2)} and physics loss {sig_figs(float(p_loss), 2)}, and data loss {sig_figs(float(d_loss), 2)}") #, omni BC: {sig_figs(float(bc_loss), 2)}, Earth losses: {sig_figs(float(Earth_bc_loss), 2)}, {sig_figs(float(Earth_dp_loss), 2)}, bsn loss: {sig_figs(float(bsn_loss), 2)}")
        print(f"Iteration {i} with loss {float(loss)} and physics loss {float(p_loss)}, and data loss {float(d_loss)}")
        print(f"mc loss {float(mc_loss)}, and mo loss {float(mo_loss)}, and pg loss {float(pg_loss)}, fg loss {float(fg_loss)}")
        print(f"SW loss {float(bc_loss)}, iono loss {float(Earth_bc_loss)}, neumann {float(nograd_loss)}")
        #print(f"mc loss {float(mc_loss)}, and mo loss {float(mo_loss)}, and pg loss {float(pg_loss)}, fg loss {float(fg_loss)}")
        loss_evolution.append(float(loss))
        data_l_evolution.append(float(d_loss))
        phys_l_evolution.append(float(p_loss))
        omni_bc_evo.append(float(bc_loss))
        Earth_bc_evo.append(float(Earth_bc_loss))
        no_grad_bc_evo.append(float(nograd_loss))
        bsn_bc_evo.append(float(bsn_loss))
        G_bc_evo.append(float(G_bc_loss))
        CRS_bc_evo.append(float(CRS_loss))
        mc_evo.append(float(mc_loss))
        mo_evo.append(float(mo_loss))
        pg_evo.append(float(pg_loss))
        fg_evo.append(float(fg_loss))
        Maxwell_evo.append(float(max_loss))
        OMNI_val_evo.append(float(OMNI_val_loss))
        early_stopping(loss)
        if early_stopping.early_stop:
            print(f"we are at epoch: {i}")
            break
#print(params)
with open('C:/Users/User/Desktop/PINN_torch/params_newrun.pkl', 'wb') as pickle_file:
    pickle_file.truncate(0)
    pickle.dump(params, pickle_file)
## plots
    def make_time_step(time_index = 450):
        n_points = 50  # Number of points in each dimension for x and y
        #t_start, t_end = 0, 20000000  # Range for t
        time_list = df_val['time'].tolist()
        t_simulation_real = time_list[time_index]
        y_simulation = 0.0
        # Generate evenly spaced points for x and y dimensions
        x_points = torch.linspace(x_bound[0], x_bound[1], n_points) / x_bound[1]
        z_points = torch.linspace(z_bound[0], z_bound[1], n_points) / z_bound[1]
        # Create a 2D grid of points for x and y
        x_grid, z_grid = torch.meshgrid(x_points,  z_points, indexing="xy")
        xz_grid = torch.stack([x_grid, z_grid], dim=-1).reshape(-1, 2)
        # Generate random values for the t dimension
        t_values = torch.full((xz_grid.shape[0], 1), t_simulation_real) / t_lim
        y_values = torch.full((xz_grid.shape[0], 1), y_simulation) / y_lim
        grid_points = torch.cat((xz_grid, y_values, t_values), dim=1)
        rearranged_tensor = grid_points[:, [0, 2, 3, 1]]
        grid_points = rearranged_tensor
        # Set requires_grad=True for the entire grid
        grid_points.requires_grad_(True)
        
        Y_tensor = funcs[0](grid_points, params)
        # B_x_tensor = Y_tensor[:,0:1].squeeze()
        # B_y_tensor = Y_tensor[:,1:2].squeeze()
        # B_z_tensor = Y_tensor[:,6:7].squeeze()
        # V_x_tensor = Y_tensor[:,2:3].squeeze()
        # V_y_tensor = Y_tensor[:,3:4].squeeze()
        # V_z_tensor = Y_tensor[:,7:8].squeeze()
        # x_tensor = grid_points[:, 0]
        # y_tensor = grid_points[:, 1]
        # z_tensor = grid_points[:, 3]
        
        # B_mag_tensor = torch.sqrt(B_x_tensor ** 2 + B_z_tensor ** 2)
        # V_mag_tensor = torch.sqrt(V_x_tensor ** 2 + V_z_tensor ** 2)


        # Convert tensors to NumPy arrays
        # x_array = x_tensor.detach().numpy()
        # y_array = y_tensor.detach().numpy()
        # z_array = z_tensor.detach().numpy()
        # Bx_array = B_x_tensor.detach().numpy()
        # By_array = B_y_tensor.detach().numpy()
        # Bz_array = B_z_tensor.detach().numpy()
        # Vx_array = V_x_tensor.detach().numpy()
        # Vy_array = V_y_tensor.detach().numpy()
        # Vz_array = V_z_tensor.detach().numpy()
        
        # B_array = B_mag_tensor.detach().numpy()
        # v_array = V_mag_tensor.detach().numpy()
        
        # plot with imposed boundary condition reentered
        total_res_tensor = torch.cat((grid_points, Y_tensor), dim=1)
        x_total_res = total_res_tensor[:, 0].squeeze() * x_bound[1]
        y_total_res = total_res_tensor[:, 1].squeeze() * y_bound[1]
        z_total_res = total_res_tensor[:, 3].squeeze() * z_bound[1]
        pos_total_res = torch.cat((x_total_res.unsqueeze(1), y_total_res.unsqueeze(1), z_total_res.unsqueeze(1)), dim=1)
        
        B_io, pos_vector, mask_forplot = magnetic_field(pos_total_res, m, io_bound=1e10)


        Bx_masked = B_io[:, 0]  # All elements in the last dimension, index 0
        By_masked = B_io[:, 1]  # All elements in the last dimension, index 1
        Bz_masked = B_io[:, 2]  # All elements in the last dimension, index 2

        
        B_coords_msk_forplot = torch.cat([pos_vector, Bx_masked.unsqueeze(1), By_masked.unsqueeze(1), Bz_masked.unsqueeze(1)], dim=1)
        
        # scale B_ionosphere
        B_inner_tensor = torch.zeros_like(B_coords_msk_forplot)
        B_inner_tensor[:, 0] = B_coords_msk_forplot[:, 0] / x_bound[1]
        B_inner_tensor[:, 1] = B_coords_msk_forplot[:, 1] / y_bound[1]
        B_inner_tensor[:, 2] = B_coords_msk_forplot[:, 2] / z_bound[1]
        B_inner_tensor[:, 3] = B_coords_msk_forplot[:, 3] / bx_lim
        B_inner_tensor[:, 4] = B_coords_msk_forplot[:, 4] / by_lim
        B_inner_tensor[:, 5] = B_coords_msk_forplot[:, 5] / bz_lim
        
        total_res_tensor = total_res_tensor[mask_forplot]
        
        total_res_tensor[:, 4] = (total_res_tensor[:,4].squeeze() + B_inner_tensor[:, 3].squeeze()) / 2
        total_res_tensor[:, 5] = (total_res_tensor[:,5].squeeze() + B_inner_tensor[:, 4].squeeze()) / 2
        total_res_tensor[:, 10] = (total_res_tensor[:,10].squeeze() + B_inner_tensor[:, 5].squeeze()) / 2
        total_res_tensor[:, 6] = (total_res_tensor[:,6].squeeze() - B_inner_tensor[:, 3].squeeze()) / 2
        total_res_tensor[:, 7] = (total_res_tensor[:,7].squeeze() - B_inner_tensor[:, 4].squeeze()) / 2
        total_res_tensor[:, 11] = (total_res_tensor[:,11].squeeze() - B_inner_tensor[:, 5].squeeze()) / 2
        final_plot_tensor = total_res_tensor
        
        B_x_tensor_n = final_plot_tensor[:,4:5].squeeze()
        #B_y_tensor_n = final_plot_tensor[:,5:6].squeeze()
        B_z_tensor_n = final_plot_tensor[:,10:11].squeeze()
        V_x_tensor_n = final_plot_tensor[:,6:7].squeeze()
        #V_y_tensor_n = final_plot_tensor[:,7:8].squeeze()
        V_z_tensor_n = final_plot_tensor[:,11:12].squeeze()
        x_tensor_n = final_plot_tensor[:, 0:1].squeeze()
        #y_tensor_n = final_plot_tensor[:, 1:2].squeeze()
        z_tensor_n = final_plot_tensor[:, 3:4].squeeze()
        
        B_mag_tensor_n = torch.sqrt(B_x_tensor_n ** 2 + B_z_tensor_n ** 2)
        V_mag_tensor_n = torch.sqrt(V_x_tensor_n ** 2 + V_z_tensor_n ** 2)


        # Convert tensors to NumPy arrays
        x_array_n = x_tensor_n.detach().numpy()
        #y_array_n = y_tensor_n.detach().numpy()
        z_array_n = z_tensor_n.detach().numpy()
        Bx_array_n = B_x_tensor_n.detach().numpy()
        #By_array_n = B_y_tensor_n.detach().numpy()
        Bz_array_n = B_z_tensor_n.detach().numpy()
        Vx_array_n = V_x_tensor_n.detach().numpy()
        #Vy_array_n = V_y_tensor_n.detach().numpy()
        Vz_array_n = V_z_tensor_n.detach().numpy()
        
        norm = np.sqrt(Bx_array_n**2 + Bz_array_n**2)
        B_norm = np.log(B_mag_tensor_n.detach().numpy() + tolerance)
        Bx_norm = Bx_array_n / (norm*20) * np.log(norm + tolerance)
        Bz_norm = Bz_array_n / (norm*20) * np.log(norm + tolerance)
        V_norm = np.log(V_mag_tensor_n.detach().numpy()+ tolerance)
        norm2 = np.sqrt(Vx_array_n ** 2 + Vz_array_n ** 2)
        vx_norm = Vx_array_n / (norm2*20) * np.log(norm2 + tolerance)
        vz_norm = Vz_array_n / (norm2*20) * np.log(norm2 + tolerance)
        # B_array_n = B_mag_tensor_n.detach().numpy()
        # v_array_n = V_mag_tensor_n.detach().numpy()
        #lower_percentile, upper_percentile = 0.1, 99
        # vminb = np.percentile(B_norm, lower_percentile)
        # vmaxb = np.percentile(B_norm, upper_percentile)
        # vminV = np.percentile(V_norm, lower_percentile)
        # vmaxV = np.percentile(V_norm, upper_percentile)
        return x_array_n, z_array_n, Bx_norm, Bz_norm, B_norm, vx_norm, vz_norm, V_norm
    
    # x_array_n, z_array_n, Bx_norm, Bz_norm, B_norm, vx_norm, vz_norm, V_norm = make_time_step(0)
    
    loss_evolution = loss_evolution[2:]
    data_l_evolution = data_l_evolution[2:]
    phys_l_evolution = phys_l_evolution[2:]
    omni_bc_evo = omni_bc_evo[2:]
    Earth_bc_evo = Earth_bc_evo[2:] 
    bsn_bc_evo = bsn_bc_evo[2:]
    G_bc_evo = G_bc_evo[2:]
    mc_evo = mc_evo[2:]
    mo_evo = mo_evo[2:]
    pg_evo = pg_evo[2:]
    fg_evo = fg_evo[2:]
    
    # loss plots
    fig, ax = plt.subplots()
    ax.semilogy(loss_evolution, label='Total Loss')
    ax.semilogy(data_l_evolution, label = 'Data/BC loss')
    ax.semilogy(phys_l_evolution, label = 'Physics based loss')
    ax.set(title="Total Loss evolution MSE", xlabel="# epochs", ylabel="Total Loss MSE")
    fig, ax = plt.subplots()
    ax.semilogy(data_l_evolution, label = 'Data/BC loss')
    ax.set(title="Data Loss evolution MSE", xlabel="# epochs", ylabel="Data Loss MSE")
    fig, ax = plt.subplots()
    ax.semilogy(phys_l_evolution)
    ax.set(title="Physics Loss evolution MSE", xlabel="# epochs", ylabel="Physics Loss MSE")
    # surface plots
    fig, ax3 = plt.subplots()
    ax3.semilogy(omni_bc_evo, label='Upstream Solar Wind BC')
    ax3.semilogy(Earth_bc_evo, label='Ionospheric Near-Earth BC')
    ax3.semilogy(no_grad_bc_evo, label='Neumann BC')
    #ax3.semilogy(bsn_bc_evo, label='Bow Shock Nose BC')
    ax3.semilogy(CRS_bc_evo, label='CRRES Validation loss')
    ax3.semilogy(OMNI_val_evo, label='OMNI within bounds Validation loss')
    ax3.set(title="BC Loss evolution (included in Data loss)", xlabel="# epochs", ylabel="BC Loss")
    ax3.legend()
    fig, ax4 = plt.subplots()
    ax4.semilogy(mc_evo, label='Mass continuity')
    ax4.semilogy(mo_evo, label='Momentum')
    ax4.semilogy(pg_evo, label='Pressure gradient')
    ax4.semilogy(fg_evo, label='Induction')
    ax4.semilogy(Maxwell_evo, label='Divergence free B')
    ax4.set(title="Physics based Loss evolution (included in Physics loss)", xlabel="# epochs", ylabel="BC Loss")
    ax4.legend()
    
    # R_E_plot = 1 / 48
    # # fig, ax = plt.subplots()
    # # scatter = ax.scatter(x_array, z_array, c=B_array, cmap='viridis', s=3)  # Use a colormap of your choice, 'viridis' is just an example
    # # plt.colorbar(scatter, label='B map') 
    # # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # # ax.add_patch(full_circle)
    # # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # # ax.add_patch(semicircle_white)
    # # ax.set(title="2D Map of B magnitude", xlabel="X axis", ylabel="Y axis")
    # # ax.set_aspect('equal', 'box')
    # scale_value = 1
    # # fig, ax = plt.subplots()
    # # quiver = ax.quiver(x_array, z_array, Bx_array, Bz_array, B_array, scale = scale_value / 10, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    # # plt.colorbar(quiver, ax=ax, label='Magnitude of B') 
    # # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # # ax.add_patch(full_circle)
    # # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # # ax.add_patch(semicircle_white)
    # # ax.set(title=f"2D Vector plot of B magnitude at {t_simulation_real} (mins) from IC", xlabel="X GSE", ylabel="Z GSE")
    # # ax.set_aspect('equal', 'box')
    # # ax.set_xlim(min(x_array), max(x_array))
    # # ax.set_ylim(min(z_array), max(z_array))
    
    # # fig2, ax2 = plt.subplots()
    # # quiver2 = ax2.quiver(x_array, z_array, Vx_array, Vz_array, v_array, scale = scale_value, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    # # plt.colorbar(quiver, ax=ax2, label='Magnitude of V solar wind velocity Re/s') 
    # # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # # ax2.add_patch(full_circle)
    # # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # # ax2.add_patch(semicircle_white)
    # # ax2.set(title="2D Vector plot of velocity magnitude at one time", xlabel="X GSE", ylabel="Z GSE")
    # # ax2.set_aspect('equal', 'box')
    # # ax2.set_xlim(min(x_array), max(x_array))
    # # ax2.set_ylim(min(z_array), max(z_array))
    
    # fig, ax = plt.subplots()
    # quiver = ax.quiver(x_array_n, z_array_n, Bx_norm, Bz_norm, B_norm, scale = scale_value * 8, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    # plt.colorbar(quiver, ax=ax, label='Magnitude of B magnetic field strength (nT)') 
    # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # ax.add_patch(full_circle)
    # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # ax.add_patch(semicircle_white)
    # ax.set(title="2D Map of B magnitude at other time", xlabel="X axis", ylabel="Z axis")
    # ax.set_aspect('equal', 'box')
    # ax.set_xlim(min(x_array_n), max(x_array_n))
    # ax.set_ylim(min(z_array_n), max(z_array_n))
    
    # fig, ax = plt.subplots()
    # quiver = ax.quiver(x_array_n, z_array_n, Bx_norm, Bz_norm, B_norm, scale = scale_value * 5, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    # plt.colorbar(quiver, ax=ax, label='Magnitude of B magnetic field strength (nT)') 
    # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # ax.add_patch(full_circle)
    # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # ax.add_patch(semicircle_white)
    # ax.set(title="2D Map of B magnitude at other time", xlabel="X axis", ylabel="Z axis")
    # ax.set_aspect('equal', 'box')
    # ax.set_xlim(min(x_array_n), max(x_array_n))
    # ax.set_ylim(min(z_array_n), max(z_array_n))
    
    # fig, ax = plt.subplots()
    # quiver = ax.quiver(x_array_n, z_array_n, Bx_norm, Bz_norm, B_norm, scale = scale_value * 8, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    # plt.colorbar(quiver, ax=ax, label='Magnitude of B magnetic field strength (nT)') 
    # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # ax.add_patch(full_circle)
    # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # ax.add_patch(semicircle_white)
    # ax.set(title="2D Map of B magnitude at other time", xlabel="X axis", ylabel="Z axis")
    # ax.set_aspect('equal', 'box')
    # ax.set_xlim(min(x_array_n), max(x_array_n))
    # ax.set_ylim(min(z_array_n), max(z_array_n))
    
    # fig, ax = plt.subplots()
    # quiver = ax.quiver(x_array_n, z_array_n, Bx_norm, Bz_norm, B_norm, scale = scale_value * 10, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    # plt.colorbar(quiver, ax=ax, label='Magnitude of B magnetic field strength (nT)') 
    # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # ax.add_patch(full_circle)
    # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # ax.add_patch(semicircle_white)
    # ax.set(title="2D Map of B magnitude at other time", xlabel="X axis", ylabel="Z axis")
    # ax.set_aspect('equal', 'box')
    # ax.set_xlim(min(x_array_n), max(x_array_n))
    # ax.set_ylim(min(z_array_n), max(z_array_n))
    
    
    # fig, ax = plt.subplots()
    # quiver = ax.quiver(x_array_n, z_array_n, vx_norm, vz_norm, V_norm, scale = scale_value * 8, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    # plt.colorbar(quiver, ax=ax, label='Magnitude of B magnetic field strength (nT)') 
    # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # ax.add_patch(full_circle)
    # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # ax.add_patch(semicircle_white)
    # ax.set(title="2D Map of V magnitude at other time", xlabel="X axis", ylabel="Z axis")
    # ax.set_aspect('equal', 'box')
    # ax.set_xlim(min(x_array_n), max(x_array_n))
    # ax.set_ylim(min(z_array_n), max(z_array_n))
    
    # fig, ax = plt.subplots()
    # quiver = ax.quiver(x_array_n, z_array_n, vx_norm, vz_norm, V_norm, scale = scale_value * 5, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    # plt.colorbar(quiver, ax=ax, label='Magnitude of B magnetic field strength (nT)') 
    # full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    # ax.add_patch(full_circle)
    # semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    # ax.add_patch(semicircle_white)
    # ax.set(title="2D Map of V magnitude at other time", xlabel="X axis", ylabel="Z axis")
    # ax.set_aspect('equal', 'box')
    # ax.set_xlim(min(x_array_n), max(x_array_n))
    # ax.set_ylim(min(z_array_n), max(z_array_n))
    
    fig, ax = plt.subplots()
    X, Z, Bx_norm, Bz_norm, B_norm, vx_norm, vz_norm, V_norm = make_time_step(0)
    Q = ax.quiver(X, Z, Bx_norm, Bz_norm, pivot='mid', color='b', scale=5)

    # ax.set_xlim(-1, 7)
    # ax.set_ylim(-1, 7)

    def update_quiver(num, Q, X, Z):
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """

        X, Z, Bx_norm, Bz_norm, B_norm, vx_norm, vz_norm, V_norm = make_time_step(num)

        Q.set_UVC(Bx_norm, Bz_norm)
        print('one step done with ', num, 'num')
        return Q,

    # you need to set blit=False, or the first set of arrows never gets
    # cleared on subsequent frames
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Z), frames=range(1, 600001, 1000), interval=50, blit=False)
    anim.save('C:/Users/User/Desktop/PINN_torch/plasma_anim.mp4', writer='ffmpeg')
    fig.tight_layout()
    plt.show()

## for flux model
def make_tensor_for_fluxmodel(df_flux, spatio_lengths):
    f_time_list = df_flux["time"].tolist()
    L_shell_list = np.arange(4.6, 11.0, 0.1)
    L_list_scld = [(x * R_E)/ y_bound[1] for x in L_shell_list]
    L_shell_i = 6.6 * R_E / y_bound[1]
    grid_points_full = torch.cat((generate_points_on_sphere(spatio_lengths, L_shell_i), torch.full((spatio_lengths, 1), f_time_list[0])), dim=1)
    for ts in f_time_list[1:]:
        grid_points_no_t = generate_points_on_sphere(spatio_lengths, L_shell_i)
        for L_shell in L_list_scld:
            grid_no_t = generate_points_on_sphere(spatio_lengths, L_shell)
            grid_points_no_t = torch.cat((grid_points_no_t, grid_no_t), dim=0)
        grid_points_t = torch.cat((grid_points_no_t, torch.full((grid_points_no_t.shape[0], 1), ts)), dim=1)
        grid_points_full = torch.cat((grid_points_full, grid_points_t), dim=0)
    return grid_points_full
def prod_plasma_df(flux_grid):
    F_flux_tensor = funcs[0](flux_grid, params)
    # Convert tensor to NumPy array
    numpy_array_flux = F_flux_tensor.numpy()
    numpy_array_f_X = flux_grid.numpy()
    # Convert NumPy array to Pandas DataFrame
    df = pd.DataFrame(numpy_array_flux)
    df_flux_X = pd.DataFrame(numpy_array_f_X)
    # Optionally, you can set column names and index
    # For example:
    df.columns = ['Bx', 'By', 'Vx', 'vy', 'rho', 'P', 'Bz', 'Vz']
    df_flux_X.columns=['x', 'y', 'time', 'z']
    
    squared_values = df_flux_X[['x', 'y', 'z']].apply(np.square, axis=1)

    # Sum the squared values for each row
    sum_of_squared_values = squared_values.sum(axis=1)

    # Take the square root of the sum of squared values
    L_shell_array = np.sqrt(sum_of_squared_values)
    df['L'] = L_shell_array
    df['time'] = df_flux_X['time']
    df.to_csv('C:/Users/User/Desktop/PINN_torch/flux_train_dataframe.csv')
def load_params(file_path):
    with open('file_path', 'rb') as pickle_file:
        params = pickle.load(pickle_file)