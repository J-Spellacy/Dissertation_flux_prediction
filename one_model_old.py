## second document
## imports dependencies and dataframe
#%matplotlib ipympl
import torch
import torch.nn as nn
import torchopt
from typing import Callable
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Support_funcs_one_model import make_forward_fn, LinearNN, model_creation
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
from math import floor, log10

def sig_figs(num: float, precision: int):
        return round(num, -int(floor(log10(abs(num)))) + (precision - 1))

# Convert df to NumPy array to PyTorch tensor with dimensions [num_rows, num_features]
# 2d df
# df = pd.read_csv('C:/Users/User/Desktop/PINN_torch/PINN_data_OMNI_cutdown.csv', index_col=0)
df_val = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_cutdown.csv', index_col=0)
data_numpy_arr = df_val.values
total_tensor = torch.from_numpy(data_numpy_arr).float()

df_GOES_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/combined_ephem_mag.csv')
GOES_data_numpy_arr = df_GOES_BC.values
G_BC_tensor = torch.from_numpy(GOES_data_numpy_arr).float()

df_EARTH_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/GSE_ver.csv')
EARTH_data_numpy_arr = df_EARTH_BC.values
E_BC_tensor = torch.from_numpy(EARTH_data_numpy_arr).float()
## TO DO: ADD eta and gamma (non ideal induction), add more equations including maybe momentum?, also add animation output function over time steps and see what it predicts, add more BCs such as GOES MAG and PLASMAPAUSE AND PLASMASPHERE CONSTRAINTS

## loss definition TO DO: add other equations to create full physics loss also work on maybe improving the boundary loss maybe scaling? its a bit shit

def make_loss_fn(funcs: list, weight_phys: float) -> Callable:
    def loss_fn(params: tuple, x: torch.Tensor):

        # Calculate the square root of the sum of squares for the first three parameters
        magnitudes = torch.sqrt((x[:, :3] ** 2).sum(dim=1))

        # Define a threshold for the magnitude
        threshold = 6.0

        # Create a mask where the magnitude is below the threshold
        mask = magnitudes < threshold

        # Use the mask to select rows from the original tensor
        x_E_Boundary = x[mask]

        x_interior = x[~mask]
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
        
        mu_0 = 4 * np.pi * 1e-7
        eta =  0.0001
        D_p = 0.001
        gamma  = 5.0 / 3.0
        G = 6.67430e-11
        M = 5.972e24
        R_E = 6371000.0
        
        def scale_sample(sample: torch.Tensor, scaler_min = -1.0, scaler_max = 1.0) -> torch.Tensor:
            sample_max, sample_min = sample.max(), sample.min()
            scaled_sample = (sample - sample_min)/(sample_max - sample_min)*(scaler_max - scaler_min) + scaler_min
            return scaled_sample
        
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
        dBxdy_value = dBxdX[:, 1]
        dBxdt_value = dBxdX[:, 2]
        dBxdz_value = dBxdX[:, 3]
        dBydx_value = dBydX[:, 0]
        dBydy_value = dBydX[:, 1]
        dBydt_value = dBydX[:, 2]
        dBydz_value = dBydX[:, 3]
        dpdx_value = dpdX[:, 0]
        dpdy_value = dpdX[:, 1]
        dpdt_value = dpdX[:, 2]
        dpdz_value = dpdX[:, 3]
        dVzdx_value = dVzdX[:, 0]
        dVzdy_value = dVzdX[:, 1]
        dVzdz_value = dVzdX[:, 3]
        dVzdt_value = dVzdX[:, 2]
        dBzdx_value = dBzdX[:, 0]
        dBzdy_value = dBzdX[:, 1]
        dBzdt_value = dBzdX[:, 2]
        dBzdz_value = dBzdX[:, 3]
        
        # mass continuity equation
        
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
        
        induction_interior_x = dBxdt_value - curl_VcrossB_x# curl goes here
        induction_interior_y = dBydt_value - curl_VcrossB_y
        induction_interior_z = dBzdt_value - curl_VcrossB_z
        
        induction_interior = torch.sqrt(induction_interior_x ** 2 + induction_interior_y ** 2 + induction_interior_z ** 2)
        
        # New residuals:
        # new mass continuity
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
            inputs = x,
            outputs = drhodx_value,
            grad_outputs=torch.ones_like(drhodx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_rhoydX = torch.autograd.grad(
            inputs = x,
            outputs = drhody_value,
            grad_outputs=torch.ones_like(drhody_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_rhozdX  = torch.autograd.grad(
            inputs = x,
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

        # new momentum residual:
        
        # (v dot del)v
        dir_der_Vx = Vx_value * dVxdx_value + Vy_value * dVxdy_value + Vz_value * dVxdz_value
        dir_der_Vy = Vx_value * dVydx_value + Vy_value * dVydy_value + Vz_value * dVydz_value
        dir_der_Vz = Vx_value * dVzdx_value + Vy_value * dVzdy_value + Vz_value * dVzdz_value
        
        E_indices = torch.randperm(E_BC_tensor.size(0))[:batch_size]
        E_boundary_sample = E_BC_tensor[E_indices]
        B_earth_x = scale_sample(E_boundary_sample[:,1:2])#.squeeze().unsqueeze(1)
        B_earth_y = scale_sample(E_boundary_sample[:,2:3])#.squeeze().unsqueeze(1)
        B_earth_z = scale_sample(E_boundary_sample[:,3:4])#.squeeze().unsqueeze(1)
        
        Bdiffx_value  = Bx_value - B_earth_x
        Bdiffy_value  = By_value - B_earth_y
        Bdiffz_value  = Bz_value - B_earth_z
        
        dBdiffxdX = torch.autograd.grad(
            inputs = x,
            outputs = Bdiffx_value,
            grad_outputs=torch.ones_like(Bdiffx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBdiffydX = torch.autograd.grad(
            inputs = x,
            outputs = Bdiffy_value,
            grad_outputs=torch.ones_like(Bdiffy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        dBdiffzdX = torch.autograd.grad(
            inputs = x,
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
        
        jx_value = (1/mu_0) * (dBdiffzdy - dBdiffydz)
        jy_value = (1/mu_0) * (dBdiffxdz - dBdiffzdx)
        jz_value = (1 / mu_0) * (dBdiffydx - dBdiffxdy)
        
        jcrossB_x = (jy_value * Bz_value) - (jz_value * By_value)
        jcrossB_y = (jz_value * Bx_value) - (jx_value * Bz_value)
        jcrossB_z = (jx_value * By_value) - (jy_value * Bx_value)

        def gravitational_field_torch(coords):
            # Extract x, y, z coordinates from the tensor; ignoring the time component 't'
            x, y, _, z = coords.t()  # Transpose and unpack the coordinates
            
            x, y, z = x * R_E, y * R_E, z * R_E
            
            # Compute the vector r from the Earth's center to the points and its magnitude r
            r = torch.stack((x, y, z), dim=1)  # Stack x, y, z horizontally (dim=1)
            r_mag = torch.norm(r, dim=1, keepdim=True)  # Calculate the magnitude of r
            
            # Compute the gravitational field vector g
            g = -G * M / r_mag**3
            g_vector = g * r 
            g_vector =  g_vector / R_E# Multiply each element by r to get the vector components
            
            return g_vector
        
        G_vector = gravitational_field_torch(x)
        g_vector_x = G_vector[:, 0:1]
        g_vector_y = G_vector[:, 1:2]
        g_vector_z = G_vector[:, 2:3]
        
        grad2_VxdX = torch.autograd.grad(
            inputs = x,
            outputs = dVxdx_value,
            grad_outputs=torch.ones_like(dVxdx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_VydX = torch.autograd.grad(
            inputs = x,
            outputs = dVydy_value,
            grad_outputs=torch.ones_like(dVydy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_VzdX = torch.autograd.grad(
            inputs = x,
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
        
        phi_x = grad2_Vxdx + grad2_Vxdy + grad2_Vxdz
        phi_y = grad2_Vydx + grad2_Vydy + grad2_Vydz
        phi_z = grad2_Vzdx + grad2_Vzdy + grad2_Vzdz
        
        new_mo_interior_x = dVxdt_value + dir_der_Vx + dpdx_value / rho_value - jcrossB_x / rho_value - g_vector_x - phi_x / rho_value## do next parts also add b-earth and change coordinates
        new_mo_interior_y = dVydt_value + dir_der_Vy + dpdy_value / rho_value - jcrossB_y / rho_value - g_vector_y - phi_y / rho_value
        new_mo_interior_z = dVzdt_value + dir_der_Vz + dpdz_value / rho_value - jcrossB_z / rho_value - g_vector_z - phi_z / rho_value
        
        new_mo_interior = new_mo_interior_x +  new_mo_interior_y + new_mo_interior_z
        
        # new pressure gradient residual
        
        dir_der_p = Vx_value * dpdx_value + Vy_value * dpdy_value + Vz_value * dpdz_value
        
        div_V = dVxdx_value + dVydy_value + dVzdz_value
        
        gp_div_v = gamma * p_value * div_V
        
        grad2_pxdX = torch.autograd.grad(
            inputs = x,
            outputs = dpdx_value,
            grad_outputs=torch.ones_like(dpdx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_pydX = torch.autograd.grad(
            inputs = x,
            outputs = dpdy_value,
            grad_outputs=torch.ones_like(dpdy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_pzdX  = torch.autograd.grad(
            inputs = x,
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
        
        # new field gradient residual
        
        # v cross b
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
        
        grad2_BxdX = torch.autograd.grad(
            inputs = x,
            outputs = dBxdx_value,
            grad_outputs=torch.ones_like(dBxdx_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_BydX = torch.autograd.grad(
            inputs = x,
            outputs = dBydy_value,
            grad_outputs=torch.ones_like(dBydy_value),
            retain_graph=True, 
            create_graph=True
        )[0]
        
        grad2_BzdX = torch.autograd.grad(
            inputs = x,
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
        
        # sums physics interior equation residuals
        #interior = mc_interior + Gauss_interior + induction_interior
        print(f"new_mc_interior: {sig_figs(torch.mean(new_mc_interior), 2)} + new_mo_interior {sig_figs(torch.mean(new_mo_interior), 2)} + new_pg_interior {sig_figs(torch.mean(new_pg_interior), 2)} + new_fg_interior {sig_figs(torch.mean(new_fg_interior), 2)}")
        interior = new_mc_interior + new_mo_interior + new_pg_interior + new_fg_interior
        interior = scale_sample(interior)
        # boundary loss based on boundary data I.E. the omniweb data set 5 mins
        indices = torch.randperm(total_tensor.size(0))[:batch_size]
        boundary_sample = total_tensor[indices]
        new_max, new_min = 1.0, -1.0
        newt_max, newt_min = 1.0, 0.0
        t_boundary = boundary_sample[:, 0:1].squeeze()
        tb_max, tb_min = t_boundary.max(), t_boundary.min()
        t_boundary = (t_boundary  - tb_min)/(tb_max - tb_min)*(newt_max - newt_min) + newt_min
        x_boundary = boundary_sample[:,7:8].squeeze()
        xb_max, xb_min = x_boundary.max(), x_boundary.min()
        x_boundary = (x_boundary  - xb_min)/(xb_max - xb_min)*(new_max - new_min) + new_min
        y_boundary = boundary_sample[:,8:9].squeeze()
        yb_max, yb_min = y_boundary.max(), y_boundary.min()
        y_boundary = (y_boundary  - yb_min)/(yb_max - yb_min)*(new_max - new_min) + new_min
        z_boundary = boundary_sample[:,9:10].squeeze()
        zb_max, zb_min = z_boundary.max(), z_boundary.min()
        z_boundary = (z_boundary  - zb_min)/(zb_max - zb_min)*(new_max - new_min) + new_min
        X_BOUNDARY = torch.cat((x_boundary.unsqueeze(1), y_boundary.unsqueeze(1), t_boundary.unsqueeze(1), z_boundary.unsqueeze(1)), dim=1)
        
        F_boundary_tensor = F(X_BOUNDARY, params) # shape ([300, 4])
        
        boundary_sample_tensor = torch.cat((scale_sample(boundary_sample[:,1:2]), scale_sample(boundary_sample[:,2:3]), scale_sample(boundary_sample[:,3:4]), scale_sample(boundary_sample[:,4:5]), scale_sample(boundary_sample[:,5:6]),  scale_sample(boundary_sample[:,6:7]), scale_sample(boundary_sample[:,10:11]), scale_sample(boundary_sample[:, 11:12])), dim=1)
        
        # GOES magnometer boundary condition  
        
        G_indices = torch.randperm(G_BC_tensor.size(0))[:batch_size]
        G_boundary_sample = G_BC_tensor[G_indices]
        t_Gboundary = G_boundary_sample[:, 0:1].squeeze()
        tb_Gmax, tb_Gmin = t_Gboundary.max(), t_Gboundary.min()
        t_Gboundary = (t_Gboundary  - tb_Gmin)/(tb_Gmax - tb_Gmin)*(newt_max - newt_min) + newt_min
        x_Gboundary = G_boundary_sample[:,1:2].squeeze()
        xb_Gmax, xb_Gmin = x_Gboundary.max(), x_Gboundary.min()
        x_Gboundary = (x_Gboundary  - xb_Gmin)/(xb_Gmax - xb_Gmin)*(new_max - new_min) + new_min
        y_Gboundary = G_boundary_sample[:,2:3].squeeze()
        yb_Gmax, yb_Gmin = y_Gboundary.max(), y_Gboundary.min()
        y_Gboundary = (y_Gboundary  - yb_Gmin)/(yb_Gmax - yb_Gmin)*(new_max - new_min) + new_min
        z_Gboundary = G_boundary_sample[:,3:4].squeeze()
        zb_Gmax, zb_Gmin = z_Gboundary.max(), z_Gboundary.min()
        z_Gboundary = (z_Gboundary  - zb_Gmin)/(zb_Gmax - zb_Gmin)*(new_max - new_min) + new_min
        X_GBOUNDARY = torch.cat((x_Gboundary.unsqueeze(1), y_Gboundary.unsqueeze(1), t_Gboundary.unsqueeze(1), z_Gboundary.unsqueeze(1)), dim=1)
        F_Gboundary_tensor = F(X_GBOUNDARY, params)
        
        G_boundary_sample_tensor = torch.cat((scale_sample(G_boundary_sample[:,4:5]), scale_sample(G_boundary_sample[:,5:6]), scale_sample(G_boundary_sample[:,6:7])),  dim=1)
        G_boundary_result_tensor = torch.cat((F_Gboundary_tensor[:,0:1],F_Gboundary_tensor[:,1:2], F_Gboundary_tensor[:,6:7]), dim=1)
        
        # boundary loss based on 
        
        loss = nn.MSELoss()
        physics_loss = loss(interior, torch.zeros_like(interior))
        bc_loss = loss(F_boundary_tensor, boundary_sample_tensor)
        #bc_loss = loss(boundary, torch.zeros_like(boundary))
        G_bc_loss = loss(G_boundary_result_tensor, G_boundary_sample_tensor)
        loss_value = weight_phys * physics_loss + bc_loss + G_bc_loss
        data_loss = bc_loss + G_bc_loss
        
        return loss_value, physics_loss, data_loss
    return loss_fn

## initialise loss and models
if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(random.randint(1, 50))

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=2)
    parser.add_argument("-d", "--dim-hidden", type=int, default=120)
    parser.add_argument("-b", "--batch-size", type=int, default=5000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-2)
    parser.add_argument("-e", "--num-epochs", type=int, default=250)

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
    model = LinearNN(num_inputs=4, num_layers=num_hidden, num_neurons=dim_hidden)
    

    # initialise functions
    funcs = make_forward_fn(model, derivative_order=1)
    params = tuple(model.parameters())
    loss_fn = make_loss_fn(funcs, 0.8)

    # choose optimizer with functional API using functorch
    optimizer =torch.optim.Adam(params, lr=learning_rate_0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    

    
## running of the model
    # train the model to do: work on making so a loop utilises the params_list and loss function properly
    loss_evolution = []
    data_l_evolution = []
    phys_l_evolution = []
    for i in range(num_iter):
        optimizer.zero_grad(set_to_none=False)
        # sample points in the domain randomly for each epoch
        x_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)#.unsqueeze(1)
        y_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)#.unsqueeze(1)
        t_dim = torch.FloatTensor(batch_size).uniform_(0, 1).requires_grad_(True)#.unsqueeze(1)
        z_dim = torch.FloatTensor(batch_size).uniform_(-1.0, 1.0).requires_grad_(True)
        x = torch.cat((x_dim.unsqueeze(1), y_dim.unsqueeze(1), t_dim.unsqueeze(1), z_dim.unsqueeze(1)), dim=1)
        
        # compute the loss with the current parameters
        loss, p_loss, d_loss = loss_fn(params, x)
        # update the parameters with functional optimizer
        
        loss.backward()
        
        optimizer.step()
        scheduler.step(loss)

        #params_list = new_params_list

        print(f"Iteration {i} with loss {sig_figs(float(loss), 2)} and physics loss {sig_figs(float(p_loss), 2)}, and data loss {sig_figs(float(d_loss), 2)}")
        loss_evolution.append(float(loss))
        data_l_evolution.append(float(d_loss))
        phys_l_evolution.append(float(p_loss))
    
## plots
    n_points = 50  # Number of points in each dimension for x and y
    
    bound_value = 60.0
    real_bounds_min, real_bounds_max = -bound_value, bound_value
    x_start, x_end = real_bounds_min, real_bounds_max  # Range for x
    z_start, z_end = real_bounds_min, real_bounds_max  # Range for y
    #t_start, t_end = 0, 20000000  # Range for t
    time_list = df_val["time"].tolist()
    t_simulation_real = random.choice(time_list)
    t_simulation = t_simulation_real/(max(time_list))
    y_simulation = random.uniform(-1.0, 1.0)
    # Generate evenly spaced points for x and y dimensions
    x_points = torch.linspace(x_start, x_end, n_points)
    x_points_scaled = x_points/df_val["X_(S/C)__GSE_Re"].max()
    z_points = torch.linspace(z_start, z_end, n_points)
    z_points_scaled = z_points/df_val["Z_(S/C)__GSE_Re"].max()
    # Create a 2D grid of points for x and y
    x_grid, z_grid = torch.meshgrid(x_points_scaled,  z_points_scaled, indexing="xy")
    xz_grid = torch.stack([x_grid, z_grid], dim=-1).reshape(-1, 2)
    # Generate random values for the t dimension, scaled to the range [t_start, t_end]
    t_values = torch.full((xz_grid.shape[0], 1), t_simulation)
    y_values = torch.full((xz_grid.shape[0], 1), y_simulation)
    #t_values = torch.rand(xy_grid.shape[0], 1) * (t_end - t_start) + t_start
    grid_points = torch.cat((xz_grid, y_values, t_values), dim=1)
    rearranged_tensor = grid_points[:, [0, 2, 3, 1]]
    grid_points = rearranged_tensor
    # Set requires_grad=True for the entire grid
    grid_points.requires_grad_(True)

    def unscale(scaled_out: torch.Tensor, real_min, real_max) -> torch.Tensor:
        s_o_min, s_o_max = scaled_out.min(), scaled_out.max()
        unscaled_out = (scaled_out - s_o_min)/(s_o_max - s_o_min)*(real_max - real_min) + real_min
        return unscaled_out
    

    
    Y_tensor = funcs[0](grid_points, params)
    B_x_tensor = unscale(Y_tensor[:,0:1].squeeze(),df_val["BX__GSE_nT"].min(),df_val["BX__GSE_nT"].max())
    B_y_tensor = unscale(Y_tensor[:,1:2].squeeze(),df_val["BY__GSE_nT"].min(),df_val["BY__GSE_nT"].max())
    B_z_tensor = unscale(Y_tensor[:,6:7].squeeze(), df_val["BZ__GSE_nT"].min(), df_val["BZ__GSE_nT"].max())
    B_mag_tensor = torch.sqrt(B_x_tensor ** 2 + B_z_tensor ** 2)
    x_tensor = unscale(grid_points[:, 0],real_bounds_min,real_bounds_max)
    y_tensor = unscale(grid_points[:, 1],real_bounds_min,real_bounds_max)
    z_tensor = unscale(grid_points[:, 3],real_bounds_min,real_bounds_max)
    #t_tensor = torch.full((B_mag_tensor.shape[0], 1), t_simulation_real).squeeze()
    
    # create a tensor of all the values being plotted and cut off all values out of the plotting bound:
    
    
    # Convert tensors to NumPy arrays
    x_array = x_tensor.detach().numpy()
    y_array = y_tensor.detach().numpy()
    z_array = z_tensor.detach().numpy()
    Bx_array = B_x_tensor.detach().numpy()
    By_array = B_y_tensor.detach().numpy()
    Bz_array = B_z_tensor.detach().numpy()
    #t_array = t_tensor.detach().numpy()
    B_array = B_mag_tensor.detach().numpy()
    
    fig, ax = plt.subplots()
    ax.semilogy(loss_evolution)
    ax.set(title="Total Loss evolution", xlabel="# epochs", ylabel="Total Loss")
    fig, ax = plt.subplots()
    ax.semilogy(data_l_evolution)
    ax.set(title="Data Loss evolution", xlabel="# epochs", ylabel="Data Loss")
    fig, ax = plt.subplots()
    ax.semilogy(phys_l_evolution)
    ax.set(title="Physics Loss evolution", xlabel="# epochs", ylabel="Physics Loss")
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_array, z_array, c=B_array, cmap='viridis', s=1)  # Use a colormap of your choice, 'viridis' is just an example
    plt.colorbar(scatter, label='B map') 
    full_circle = patches.Circle((0, 0), radius=1, color='black')
    ax.add_patch(full_circle)
    semicircle_white = patches.Wedge((0, 0), 1, -90, 90, color='white')
    ax.add_patch(semicircle_white)
    ax.set(title="2D Map of B magnitude with mass continuity, Gauss and MSE of bcs", xlabel="X axis", ylabel="Y axis")
    ax.set_aspect('equal', 'box')
    
    fig, ax = plt.subplots()
    quiver = ax.quiver(x_array, z_array, Bx_array, Bz_array, B_array, scale = 5, scale_units='xy', angles='xy', cmap='viridis')  # Use a colormap of your choice, 'viridis' is just an example
    plt.colorbar(quiver, ax=ax, label='Magnitude of B') 
    full_circle = patches.Circle((0, 0), radius=1, color='black')
    ax.add_patch(full_circle)
    semicircle_white = patches.Wedge((0, 0), 1, -90, 90, color='white')
    ax.add_patch(semicircle_white)
    ax.set(title="2D Map of B magnitude at other time", xlabel="X axis", ylabel="Y axis")
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-bound_value, bound_value)
    ax.set_ylim(-bound_value, bound_value)
        
    plt.show()

## new animation function
    def animate_2d_plot(start_time_index: int, end_time_index: int, n_points: int = 300, space_bound_value: float = 50.0, df: pd.DataFrame = df_val):
        fig, ax = plt.subplots(figsize=(12, 8))
        param_max = 50  # Assuming a reasonable scale factor for visibility

        # Initializing the quiver; ensure it is visible by setting scale
        quiver = ax.quiver([], [], [], [], scale=50, scale_units='xy', angles='xy', cmap='viridis')
        
        time_list = np.linspace(start_time_index, end_time_index, 100)  # Fewer frames for testing
        real_bounds_min, real_bounds_max = -space_bound_value, space_bound_value
        
        # Adjust these points and ranges as necessary to ensure they fall within visible axis limits
        x_points = torch.linspace(real_bounds_min, real_bounds_max, n_points)
        z_points = torch.linspace(real_bounds_min, real_bounds_max, n_points)
        x_grid, z_grid = torch.meshgrid(x_points, z_points, indexing="xy")
        xz_grid = torch.stack([x_grid, z_grid], dim=-1).reshape(-1, 2)

        def update(frame):
            ax.clear()
            
            # Simulated data update
            time_frame = time_list[frame]
            x_array = xz_grid[:, 0].numpy()
            z_array = xz_grid[:, 1].numpy()
            Bx_array = np.sin(x_array) * np.cos(time_frame)  # Example simulation
            Bz_array = np.cos(z_array) * np.sin(time_frame)  # Example simulation
            B_array = np.sqrt(Bx_array**2 + Bz_array**2)  # Magnitude for coloring
            
            # Update quiver
            quiver.set_offsets(np.column_stack([x_array, z_array]))
            quiver.set_UVC(Bx_array, Bz_array, B_array)
            
            # Resetting the plot limits and labels every frame
            ax.set_xlim(real_bounds_min, real_bounds_max)
            ax.set_ylim(real_bounds_min, real_bounds_max)
            ax.set_xlabel('X (GSE)')
            ax.set_ylabel('Z (GSE)')
            ax.set_title(f"2D plot of plasma magnetic field at {time_frame:.2f} minutes from initial conditions")

        ani = FuncAnimation(fig, update, frames=len(time_list), repeat=False)
        plt.show()

    # Dummy DataFrame
    df_val = pd.DataFrame({
        "BX__GSE_nT": np.random.rand(100),
        "BY__GSE_nT": np.random.rand(100),
        "BZ__GSE_nT": np.random.rand(100),
        "X_(S/C)__GSE_Re": np.random.rand(100),
        "Z_(S/C)__GSE_Re": np.random.rand(100)
    })

    animate_2d_plot(0, 100000001, df=df_val)