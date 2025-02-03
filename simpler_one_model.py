## imports dependencies and dataframe
#%matplotlib ipympl
import torch
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
R_E = 6371000.0
D_p = 0.001
gamma  = 5.0 / 3.0
G = 6.67430e-11
M = 5.972e24
mu_0 = 4 * np.pi * 1e-7
eta =  0.002
tolerance = 1e-6

# Convert df to NumPy array to PyTorch tensor with dimensions [num_rows, num_features]
# 2d df
# df = pd.read_csv('C:/Users/User/Desktop/PINN_torch/PINN_data_OMNI_cutdown.csv', index_col=0)
df_val = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_megacut_si_units.csv')
data_numpy_arr = df_val.values
total_tensor = torch.from_numpy(data_numpy_arr).float()

df_CRRES_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/CRRES_GSE_si_units.csv', index_col=0)
CRS_data_numpy_arr = df_CRRES_BC.values
CRRES_BC_tensor = torch.from_numpy(CRS_data_numpy_arr).float()



lim_df = df_val.abs()
x_lim = (-48 * R_E, 24 * R_E)
y_lim = (-24 * R_E, 24 * R_E)
z_lim = (-24 * R_E, 24 * R_E)
t_lim = (0, df_val["time"].max())
bx_lim = lim_df["BX__GSE_T"].max()
by_lim = lim_df["BY__GSE_T"].max()
bz_lim = lim_df["BZ__GSE_T"].max()
vx_lim = lim_df["VX_VELOCITY__GSE_m/s"].max()
vy_lim = lim_df["VY_VELOCITY__GSE_m/s"].max()
vz_lim = lim_df["VZ_VELOCITY__GSE_m/s"].max()
rho_lim = lim_df["PROTON_DENSITY_kg/m^3"].max()
P_lim = lim_df["FLOW_PRESSURE_Pa"].max()

## TO DO: also add animation output function over time steps and see what it predicts

## loss definition TO DO: add other equations to create full physics loss also work on maybe improving the boundary loss maybe scaling? its a bit shit

def make_loss_fn(funcs: list, weight_phys: float) -> Callable:
    def loss_fn(params: tuple, x: torch.Tensor):
        
        loss = nn.MSELoss()
        
        selected_elements = x[:, [0, 1, 3]]
        magnitudes = torch.sqrt((selected_elements[:, 1:3] ** 2).sum(dim=1))

        # Define a threshold for the magnitude
        epsilon = 6.0 * R_E / y_lim
        epsilon_const_condition = 3.5 * R_E / y_lim
        #print(float(epsilon), float(epsilon_const_condition))
        # Create a mask where the magnitude is below the threshold
        mask = magnitudes < epsilon
        # Use the mask to select rows from the original tensor
        x_E_Boundary = x[mask]

        x_interior = x[~mask]

        batch_size_in = x_interior.shape[0]
        
        selected_elements_2 = x_E_Boundary[:, [0, 1, 3]]
        magnitudes_2 = torch.sqrt((selected_elements_2[:, 1:3] ** 2).sum(dim=1))
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
        
        E_indices = torch.randperm(E_BC_tensor.size(0))[:batch_size_in]
        E_boundary_sample = E_BC_tensor[E_indices]
        
        B_earth_y = torch.ones_like(By_value) * 60
        B_earth_x = torch.zeros_like(Bx_value)
        B_earth_z = torch.zeros_like(Bz_value)
        
        Bdiffx_value  = Bx_value - B_earth_x
        Bdiffy_value  = By_value - B_earth_y
        Bdiffz_value  = Bz_value - B_earth_z
        
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
            r = torch.stack((x, y, z), dim=1)  # Stack x, y, z horizontally (dim=1)
            r_mag = torch.norm(r, dim=1, keepdim=True)  # Calculate the magnitude of r
            
            # Compute the gravitational field vector g
            g = -G * M / r_mag**3
            g_vector = g * r # Multiply each element by r to get the vector components
            
            return g_vector
        
        # x_g_lim = torch.min(x_interior[:,0:1])
        # y_g_lim = torch.min(x_interior[:,1:2])
        # z_g_lim = torch.min(x_interior[:,3:4])
        # lim_coords = torch.tensor([[x_g_lim, y_g_lim, t_lim, z_g_lim]])
        # g_lim = gravitational_field_torch(lim_coords)
        # g_lim_x = float(g_lim[0, 0].item())
        # g_lim_y = float(g_lim[0, 1].item())
        # g_lim_z = float(g_lim[0, 2].item())
        
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
        
        mu_vis = 1.4e-3
        
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
        
        
### OMNI boundary loss based on boundary data I.E. the omniweb data set 5 mins
        indices = torch.randperm(total_tensor.size(0))[:batch_size]
        boundary_sample = total_tensor[indices]
        t_boundary = boundary_sample[:, 0:1].squeeze() / t_lim
        x_boundary = boundary_sample[:,7:8].squeeze() / y_lim
        y_boundary = boundary_sample[:,8:9].squeeze() / y_lim
        z_boundary = boundary_sample[:,9:10].squeeze() / y_lim
        X_BOUNDARY = torch.cat((x_boundary.unsqueeze(1), y_boundary.unsqueeze(1), t_boundary.unsqueeze(1), z_boundary.unsqueeze(1)), dim=1)
        
        F_boundary_tensor = F(X_BOUNDARY, params) # shape ([300, 4])
        
        boundary_sample_tensor = torch.cat((boundary_sample[:,1:2] / bx_lim, boundary_sample[:,2:3] / by_lim, boundary_sample[:,3:4] / vx_lim, boundary_sample[:,4:5] / vy_lim, boundary_sample[:,5:6] / rho_lim,  boundary_sample[:,6:7] / P_lim, boundary_sample[:,10:11] / bz_lim, boundary_sample[:, 11:12] / vz_lim), dim=1)
        
    
### CRRES BC
        # CRRES_indices_BC = torch.randperm(CRRES_BC_tensor.size(0))[:batch_size]
        # CRRES_boundary_sample_BC = CRRES_BC_tensor[CRRES_indices_BC]
        # x_CRS = CRRES_boundary_sample_BC[:,1:2].squeeze() / y_lim
        # y_CRS = CRRES_boundary_sample_BC[:,2:3].squeeze() / y_lim
        # z_CRS = CRRES_boundary_sample_BC[:,3:4].squeeze() / y_lim
        # time_CRS = CRRES_boundary_sample_BC[:,0:1].squeeze() / t_lim
        # B_CRS = CRRES_boundary_sample_BC[:,4:5].squeeze() / bx_lim
        # X_CRS = torch.cat((x_CRS.unsqueeze(1), y_CRS.unsqueeze(1), time_CRS.unsqueeze(1), z_CRS.unsqueeze(1)), dim=1)
        # F_CRS = F(X_CRS, params)
        
        # CRS_FB = torch.sqrt((F_CRS[:,0:1] ** 2) + (F_CRS[:,1:2] ** 2) + (F_CRS[:,6:7] ** 2))
        # CRS_loss = loss(CRS_FB.squeeze(), B_CRS.squeeze())
        
### loss final calculation
        physics_loss = new_mc_loss + 2 * new_mo_loss + new_pg_loss + new_fg_loss
        bc_loss = loss(F_boundary_tensor, boundary_sample_tensor)
        #bc_loss = loss(boundary, torch.zeros_like(boundary))
        data_loss = 2 * bc_loss + earth_mag_bc
        loss_value = weight_phys * physics_loss + data_loss
        ###### TO DO: check if removing certain BC allows for the CRRES loss to converge? maybe rework the ionosphere residual
        
        return loss_value, physics_loss, data_loss, earth_mag_bc, new_mc_loss, new_mo_loss, new_pg_loss, new_fg_loss
    return loss_fn

## initialise loss and models
if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(49)

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=5)
    parser.add_argument("-d", "--dim-hidden", type=int, default=200)
    parser.add_argument("-b", "--batch-size", type=int, default=6000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=7e-2)
    parser.add_argument("-e", "--num-epochs", type=int, default=800)

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
    loss_fn = make_loss_fn(funcs, 0.5)

    # choose optimizer with functional API using functorch
    optimizer =torch.optim.Adam(params, lr=learning_rate_0, eps=1e-6)
    early_stopping = EarlyStopping(tolerance=5, min_delta = 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=3)
    

    
## running of the model
    # train the model to do: work on making so a loop utilises the params_list and loss function properly
    loss_evolution = []
    data_l_evolution = []
    phys_l_evolution = []
    omni_bc_evo = []
    Earth_bc_evo = []
    bsn_bc_evo = []
    CRS_bc_evo = []
    G_bc_evo = []
    mc_evo = []
    mo_evo = []
    pg_evo = []
    fg_evo = []
    for i in range(num_iter):
        optimizer.zero_grad(set_to_none=False)
        # sample points in the domain randomly for each epoch
        x_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)
        y_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)
        t_dim = torch.FloatTensor(batch_size).uniform_(0.0, 1.0).requires_grad_(True)
        z_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)
        x = torch.cat((x_dim.unsqueeze(1), y_dim.unsqueeze(1), t_dim.unsqueeze(1), z_dim.unsqueeze(1)), dim=1)
        epsilon_condition = 6.0 * R_E / y_lim
        x_dim_e = torch.FloatTensor(batch_size).uniform_(-epsilon_condition, epsilon_condition).requires_grad_(True)
        y_dim_e = torch.FloatTensor(batch_size).uniform_(-epsilon_condition, epsilon_condition).requires_grad_(True)
        z_dim_e = torch.FloatTensor(batch_size).uniform_(-epsilon_condition, epsilon_condition).requires_grad_(True)
        x_e = torch.cat((x_dim_e.unsqueeze(1), y_dim_e.unsqueeze(1), t_dim.unsqueeze(1), z_dim_e.unsqueeze(1)), dim=1)
        x_full = torch.cat((x, x_e), dim=0)
        
        # compute the loss with the current parameters
        loss, p_loss, d_loss, bc_loss, Earth_bc_loss, G_bc_loss, Earth_dp_loss, bsn_loss, CRS_loss, mc_loss, mo_loss, pg_loss, fg_loss = loss_fn(params, x_full)
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
        #print(f"mc loss {float(mc_loss)}, and mo loss {float(mo_loss)}, and pg loss {float(pg_loss)}, fg loss {float(fg_loss)}")
        loss_evolution.append(float(loss))
        data_l_evolution.append(float(d_loss))
        phys_l_evolution.append(float(p_loss))
        omni_bc_evo.append(float(bc_loss))
        Earth_bc_evo.append(float(Earth_bc_loss + Earth_dp_loss))
        bsn_bc_evo.append(float(bsn_loss))
        G_bc_evo.append(float(G_bc_loss))
        CRS_bc_evo.append(float(CRS_loss))
        mc_evo.append(float(mc_loss))
        mo_evo.append(float(mo_loss))
        pg_evo.append(float(pg_loss))
        fg_evo.append(float(fg_loss))
        early_stopping(loss)
        if early_stopping.early_stop:
            print(f"we are at epoch: {i}")
            break
    
## plots
    n_points = 500  # Number of points in each dimension for x and y
    
    bound_value = 100.0 * R_E
    #t_start, t_end = 0, 20000000  # Range for t
    time_list = df_val2["time"].tolist()
    t_simulation_real = time_list[450]
    y_simulation = 0.0
    # Generate evenly spaced points for x and y dimensions
    x_points = torch.linspace(-bound_value, bound_value, n_points) / y_lim
    z_points = torch.linspace(-bound_value, bound_value, n_points) / y_lim
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
    B_x_tensor = Y_tensor[:,0:1].squeeze()
    B_y_tensor = Y_tensor[:,1:2].squeeze()
    B_z_tensor = Y_tensor[:,6:7].squeeze()
    V_x_tensor = Y_tensor[:,2:3].squeeze()
    V_y_tensor = Y_tensor[:,3:4].squeeze()
    V_z_tensor = Y_tensor[:,7:8].squeeze()
    x_tensor = grid_points[:, 0]
    y_tensor = grid_points[:, 1]
    z_tensor = grid_points[:, 3]
    
    B_mag_tensor = torch.sqrt(B_x_tensor ** 2 + B_z_tensor ** 2)
    V_mag_tensor = torch.sqrt(V_x_tensor ** 2 + V_z_tensor ** 2)


    # Convert tensors to NumPy arrays
    x_array = x_tensor.detach().numpy()
    y_array = y_tensor.detach().numpy()
    z_array = z_tensor.detach().numpy()
    Bx_array = B_x_tensor.detach().numpy()
    By_array = B_y_tensor.detach().numpy()
    Bz_array = B_z_tensor.detach().numpy()
    Vx_array = V_x_tensor.detach().numpy()
    Vy_array = V_y_tensor.detach().numpy()
    Vz_array = V_z_tensor.detach().numpy()
    
    B_array = B_mag_tensor.detach().numpy()
    v_array = V_mag_tensor.detach().numpy()
    
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
    ax3.semilogy(bsn_bc_evo, label='Bow Shock Nose BC')
    ax3.semilogy(CRS_bc_evo, label='CRRES BC')
    ax3.set(title="BC Loss evolution (included in Data loss)", xlabel="# epochs", ylabel="BC Loss")
    ax3.legend()
    fig, ax4 = plt.subplots()
    ax4.semilogy(mc_evo, label='Mass continuity')
    ax4.semilogy(mo_evo, label='Momentum')
    ax4.semilogy(pg_evo, label='Pressure gradient')
    ax4.semilogy(fg_evo, label='Induction')
    ax4.set(title="Physics based Loss evolution (included in Physics loss)", xlabel="# epochs", ylabel="BC Loss")
    ax4.legend()
    
    R_E_plot = R_E / y_lim
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_array, z_array, c=B_array, cmap='viridis', s=3)  # Use a colormap of your choice, 'viridis' is just an example
    plt.colorbar(scatter, label='B map') 
    full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    ax.add_patch(full_circle)
    semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    ax.add_patch(semicircle_white)
    ax.set(title="2D Map of B magnitude", xlabel="X axis", ylabel="Y axis")
    ax.set_aspect('equal', 'box')
    
    fig, ax = plt.subplots()
    quiver = ax.quiver(x_array, z_array, Bx_array, Bz_array, B_array, scale = 100, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    plt.colorbar(quiver, ax=ax, label='Magnitude of B') 
    full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    ax.add_patch(full_circle)
    semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    ax.add_patch(semicircle_white)
    ax.set(title=f"2D Vector plot of B magnitude at {t_simulation_real} (mins) from IC", xlabel="X GSE", ylabel="Z GSE")
    ax.set_aspect('equal', 'box')
    ax.set_xlim(min(x_array), max(x_array))
    ax.set_ylim(min(z_array), max(z_array))
    
    fig2, ax2 = plt.subplots()
    quiver2 = ax2.quiver(x_array, z_array, Vx_array, Vz_array, v_array, scale = 100, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    plt.colorbar(quiver, ax=ax2, label='Magnitude of V solar wind velocity Re/s') 
    full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    ax2.add_patch(full_circle)
    semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    ax2.add_patch(semicircle_white)
    ax2.set(title="2D Vector plot of velocity magnitude at one time", xlabel="X GSE", ylabel="Z GSE")
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim(min(x_array), max(x_array))
    ax2.set_ylim(min(z_array), max(z_array))
    
    fig, ax = plt.subplots()
    quiver = ax.quiver(x_array, z_array, Bx_array, Bz_array, B_array, scale = 100, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    plt.colorbar(quiver, ax=ax, label='Magnitude of B magnetic field strength (nT)') 
    full_circle = patches.Circle((0, 0), radius=R_E_plot, color='black')
    ax.add_patch(full_circle)
    semicircle_white = patches.Wedge((0, 0), R_E_plot, -90, 90, color='white')
    ax.add_patch(semicircle_white)
    ax.set(title="2D Map of B magnitude at other time", xlabel="X axis", ylabel="Z axis")
    ax.set_aspect('equal', 'box')
    ax.set_xlim(min(x_array), max(x_array))
    ax.set_ylim(min(z_array), max(z_array))
    
    plt.show()
