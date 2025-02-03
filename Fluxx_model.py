## second document
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
from Fluxx_Support_funcs import make_forward_fn, FluxNN, concatenate_previous_time_steps
import Support_funcs_one_model
from Support_funcs_one_model import sig_figs, scale_sample, scale_sample_t, unscale, unscale_not_t
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
E_R = 6371000.0
mu_0 = 4 * np.pi * 1e-7
def magnetic_field(r_vec, m, io_bound):
    # Compute the magnitude of position vectors
    r = torch.norm(r_vec, dim=-1)

    # Create a mask for points within the bounds
    mask = (r <= io_bound)

    # Filter points using the mask
    r_vec_filtered = r_vec[mask]

    # Compute magnetic field B for filtered points
    B_filtered = torch.zeros_like(r_vec_filtered)  # Initialize B_filtered

    # Compute magnetic field B only for filtered points
    for i in range(r_vec_filtered.shape[0]):
        # Check if r is below E_R
        if r[i] < E_R:
            B_filtered[i] = 0  # Set B to 0 if r is below E_R
        else:
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

## import GOES or LANL FLUX data here 
# df_train = pd.read_csv('C:/Users/User/Desktop/PINN_torch/flux_train_dataframe.csv') # dataframe with data in format like x
# df_train_numpy_arr = df_train.values
# df_train_tensor = torch.from_numpy(df_train_numpy_arr).float()


#df_val_flux = pd.read_csv('C:/Users/User/Desktop/PINN_torch/flux_model_CRRES_GSE.csv') # goes dataframe with columns time flux (2 MeV)

# cutdown version
df_val_flux = pd.read_csv('C:/Users/User/Desktop/PINN_torch/cutdown_CRRES.csv')

df_val_flux_numpy_arr = df_val_flux.values
df_val_flux_tensor = torch.from_numpy(df_val_flux_numpy_arr).float()

# x = Bx, By, Vx, Vy, n, P, Bz, Vz, x, y, time, z
# x = 0,  1,  2,  3,  4, 5, 6,  7,  8, 9, 10, 11
# df_flux_val = time, L, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, X, Y, Z
# df_flux_val = 0,    1, 2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12, 13,  14,  15,  16,  17,  18,  19,20,21

## limits set
R_E_m = 1 # 6371000.0
x_plasma_bound = (-24 * R_E_m, 24 * R_E_m)
y_plasma_bound = (-24 * R_E_m, 24 * R_E_m)
z_plasma_bound = (-24 * R_E_m, 24 * R_E_m)
time_plasma_lim = 1.046453e+09
f1_lim = 5.631738e+00
f2_lim = 5.379618e+00
f3_lim = 5.328973e+00
f4_lim = 4.975668e+00
f5_lim = 4.463170e+00
f6_lim = 4.460717e+00
f7_lim = 4.213025e+00
f8_lim = 3.993964e+00
f9_lim = 3.870470e+00
f10_lim = 3.737422e+00
f11_lim = 3.666408e+00
f12_lim = 3.527674e+00
f13_lim = 3.607707e+00
f14_lim = 3.654025e+00
f15_lim = 3.651197e+00
f16_lim = 3.651843e+00
f17_lim = 3.507399e+00
## loss definition 

def make_loss_fn(funcs: list, weight_phys: float) -> Callable:
    def loss_fn(params: tuple, x: torch.Tensor, Batch_dims_in: torch.Tensor):
        
        Batch_dims = Batch_dims_in[max_tau:]
        loss = nn.MSELoss()
        # interior loss
        F = funcs[0]       
        
        F_tensor = F(x, params)
        F_f1 = F_tensor[:,0:1].squeeze()
        F_f2 = F_tensor[:,1:2].squeeze()
        F_f3 = F_tensor[:,2:3].squeeze()
        F_f4 = F_tensor[:,3:4].squeeze()
        F_f5 = F_tensor[:,4:5].squeeze()
        F_f6 = F_tensor[:,5:6].squeeze()
        F_f7 = F_tensor[:,6:7].squeeze()
        F_f8 = F_tensor[:,7:8].squeeze()
        F_f9 = F_tensor[:,8:9].squeeze()
        F_f10 = F_tensor[:,9:10].squeeze()
        F_f11 = F_tensor[:,10:11].squeeze()
        F_f12 = F_tensor[:,11:12].squeeze()
        F_f13 = F_tensor[:,12:13].squeeze()
        F_f14 = F_tensor[:,13:14].squeeze()
        F_f15 = F_tensor[:,14:15].squeeze()
        F_f16 = F_tensor[:,15:16].squeeze()
        F_f17 = F_tensor[:,16:17].squeeze()
        # F_tensor = flux
        val_tensor_f1 = Batch_dims[:,2:3].squeeze() / f1_lim
        val_tensor_f2 = Batch_dims[:,3:4].squeeze() / f2_lim
        val_tensor_f3 = Batch_dims[:,4:5].squeeze() / f3_lim
        val_tensor_f4 = Batch_dims[:,5:6].squeeze() / f4_lim
        val_tensor_f5 = Batch_dims[:,6:7].squeeze() / f5_lim
        val_tensor_f6 = Batch_dims[:,7:8].squeeze() / f6_lim
        val_tensor_f7 = Batch_dims[:,8:9].squeeze() / f7_lim
        val_tensor_f8 = Batch_dims[:,9:10].squeeze() / f8_lim
        val_tensor_f9 = Batch_dims[:,10:11].squeeze() / f9_lim
        val_tensor_f10 = Batch_dims[:,11:12].squeeze() / f10_lim
        val_tensor_f11 = Batch_dims[:,12:13].squeeze() / f11_lim
        val_tensor_f12 = Batch_dims[:,13:14].squeeze() / f12_lim
        val_tensor_f13 = Batch_dims[:,14:15].squeeze() / f13_lim
        val_tensor_f14 = Batch_dims[:,15:16].squeeze() / f14_lim
        val_tensor_f15 = Batch_dims[:,16:17].squeeze() / f15_lim
        val_tensor_f16 = Batch_dims[:,17:18].squeeze() / f16_lim
        val_tensor_f17 = Batch_dims[:,18:19].squeeze() / f17_lim
        
        loss_f1 = loss(F_f1, val_tensor_f1)
        loss_f2 = loss(F_f2, val_tensor_f2)
        loss_f3 = loss(F_f3, val_tensor_f3)
        loss_f4 = loss(F_f4, val_tensor_f4)
        loss_f5 = loss(F_f5, val_tensor_f5)
        loss_f6 = loss(F_f6, val_tensor_f6)
        loss_f7 = loss(F_f7, val_tensor_f7)
        loss_f8 = loss(F_f8, val_tensor_f8)
        loss_f9 = loss(F_f9, val_tensor_f9)
        loss_f10 = loss(F_f10, val_tensor_f10)
        loss_f11 = loss(F_f11, val_tensor_f11)
        loss_f12 = loss(F_f12, val_tensor_f12)
        loss_f13 = loss(F_f13, val_tensor_f13)
        loss_f14 = loss(F_f14, val_tensor_f14)
        loss_f15 = loss(F_f15, val_tensor_f15)
        loss_f16 = loss(F_f16, val_tensor_f16)
        loss_f17 = loss(F_f17, val_tensor_f17)
        loss_f_list = [
            loss_f1,
            loss_f2,
            loss_f3,
            loss_f4,
            loss_f5,
            loss_f6,
            loss_f7,
            loss_f8,
            loss_f9,
            loss_f10,
            loss_f11,
            loss_f12,
            loss_f13,
            loss_f14,
            loss_f15,
            loss_f16,
            loss_f17
        ]
        
        concatenated_tensor = torch.stack(loss_f_list)

        # Sum all elements across all tensors
        total_loss = torch.sum(concatenated_tensor)
        # tau_0 = 2.67 # days
        # Energy = 2.0 # MeV
        loss_fout_list = [
            float(loss_f1),
            float(loss_f2),
            float(loss_f3),
            float(loss_f4),
            float(loss_f5),
            float(loss_f6),
            float(loss_f7),
            float(loss_f8),
            float(loss_f9),
            float(loss_f10),
            float(loss_f11),
            float(loss_f12),
            float(loss_f13),
            float(loss_f14),
            float(loss_f15),
            float(loss_f16),
            float(loss_f17)
        ]
        # psd = F_tensor[:,0:1].squeeze()
        # C_value = F_tensor[:,1:2].squeeze()
        # alpha_value = F_tensor[:,2:3].squeeze()
        # gamma1 = F_tensor[:,3:4].squeeze()
        # gamma2 = F_tensor[:,4:5].squeeze()
        # third_term_approx = F_tensor[:,5:6].squeeze()
        
        # dpsddX = torch.autograd.grad(
        #     inputs = x,
        #     outputs = psd,
        #     grad_outputs=torch.ones_like(psd),
        #     retain_graph=True, 
        #     create_graph=True
        # )[0]
        
        # dpsddt_value = dpsddX[:, 9]
        # dpsddL_value = dpsddX[:, 8]
        
        # vx_value = x[:,2:3].squeeze()
        # vy_value = x[:,3:4].squeeze()
        # vz_value = x[:,7:8].squeeze()
        # v_value = torch.sqrt((vx_value ** 2) + (vy_value ** 2) + (vz_value ** 2))
        # Bz_value = x[:,6:7].squeeze()
        # L_value = x[:,8:9].squeeze()
        # v_0 = float(torch.mean(v_value))
        
        
        # D_0_1st = C_value * (v_value / v_0) ** gamma1 ## needs filling with the real equation of input params
        # D_0_2nd = (1 + ((vx_value * Bz_value + torch.abs(vx_value * Bz_value)) / alpha_value) ** 2) ** gamma2
        
        # D_0 = D_0_1st * D_0_2nd * third_term_approx
        # D_ll = D_0 * ((6.6 / L_value) ** 10)
        # tau = tau_0 * ((6.6 / L_value) ** 10)
        
        # psd_by_L = (D_ll / (L_value ** 2)) * dpsddL_value
        
        # dpsdXbyL = torch.autograd.grad(
        #     inputs = L_value,
        #     outputs = psd_by_L,
        #     grad_outputs=torch.ones_like(psd_by_L),
        #     retain_graph=True, 
        #     create_graph=True
        # )[0]
        
        # dpsdDLL_dL_value = dpsdXbyL[:, 0]
        
        # diffusion_residual = dpsddt_value - (L_value ** 2) * dpsdDLL_dL_value + psd / tau
        
        # diffusion_loss = loss(diffusion_residual, torch.zeros_like(diffusion_residual))
        
        # def electron_momentum_from_energy(energy_mev):
        #     # Constants
        #     c = 3.0e8  # Speed of light in m/s
        #     m0c2 = 0.511  # Rest mass energy of electron in MeV
        #     MeV_to_J = 1.60218e-13  # Conversion factor from MeV to J (1eV = 1.60218e-19 J, 1MeV = 1.60218e-13 J)
        #     # Convert energy from MeV to Joules
        #     total_energy_j = (energy_mev + m0c2) * MeV_to_J  # Total energy in Joules
        #     # Calculate momentum
        #     momentum = ((total_energy_j**2 - (m0c2 * MeV_to_J)**2)**0.5) / c
        #     return momentum
        
        # rela_momentum = electron_momentum_from_energy(Energy)
        # dflux = psd * (rela_momentum ** 2)
        # flux = dflux * Energy
        # log_flux = torch.log10(flux)
        # Flux_indices = torch.randperm(df_val_flux_tensor.size(0))[:int(flux.size(0))]
        # flux_sample = df_val_flux_tensor[Flux_indices]
        # data_flux = flux_sample[:,1:2].squeeze()
        
        # MSE_residual = loss(log_flux.squeeze(), data_flux.squeeze())
        
        # loss_value = diffusion_loss + MSE_residual
        return total_loss, loss_fout_list, loss_f_list
    return loss_fn

## initialise loss and models
if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(30)

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=6)
    parser.add_argument("-d", "--dim-hidden", type=int, default=120)
    parser.add_argument("-b", "--batch-size", type=int, default=400)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("-e", "--num-epochs", type=int, default=10000)

    args = parser.parse_args()
    
    # configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = args.num_epochs
    max_tau = 12
    tolerance = 1e-8
    learning_rate_0 = args.learning_rate
    # creates a list of models which all output their own parameter to make up the list of targets: bx, by, vx, vy, rho, p
    model = FluxNN(num_inputs=19 * (max_tau + 1), num_layers=num_hidden, num_neurons=dim_hidden)
    model_plasma = Support_funcs_one_model.LinearNN(num_inputs=4, num_layers=8, num_neurons=64)

    # initialise functions
    funcs = make_forward_fn(model, derivative_order=1)
    funcs_plasma = Support_funcs_one_model.make_forward_fn(model_plasma, derivative_order=1)
    params = tuple(model.parameters())
    with open('C:/Users/User/Desktop/PINN_torch/params_newrun.pkl', 'rb') as pickle_file:
        params_plasma =  pickle.load(pickle_file)
    loss_fn = make_loss_fn(funcs, 0.8)

    # choose optimizer with functional API using functorch
    optimizer =torch.optim.Adam(params, lr=learning_rate_0)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    

    
## running of the model
    # train the model to do: work on making so a loop utilises the params_list and loss function properly
    loss_evolution = []
    indi_loss_evo = []
    for i in range(num_iter):
        optimizer.zero_grad(set_to_none=False)
        f_indices = torch.randperm(df_val_flux_tensor.size(0))[:batch_size]
        CRRES_batch = df_val_flux_tensor[f_indices]
        x_batch = CRRES_batch[:,19:20].squeeze() / x_plasma_bound[1]
        y_batch = CRRES_batch[:,20:21].squeeze() / y_plasma_bound[1]
        t_batch = CRRES_batch[:,0:1].squeeze() / time_plasma_lim
        z_batch = CRRES_batch[:,21:22].squeeze() / z_plasma_bound[1]
        #print(f"x: {x_batch.squeeze().shape}, y: {y_batch.squeeze().shape}, z: {z_batch.squeeze().shape}, t: {t_batch.squeeze().shape}")
        X_batch_prep = torch.cat((x_batch.unsqueeze(1), y_batch.unsqueeze(1), t_batch.unsqueeze(1), z_batch.unsqueeze(1)), dim=1)
        X_batch_plasma = funcs_plasma[0](X_batch_prep, params_plasma)
        #X_batch_plasma = torch.zeros_like(X_batch_plasma_n)
        X_batch = torch.cat((X_batch_plasma, X_batch_prep), dim=1)
        
        # imposing ionospheric BC:
        total_res_tensor = torch.zeros_like(X_batch)
        total_res_tensor[:,0] = X_batch[:,8]
        total_res_tensor[:,1] = X_batch[:,9]
        total_res_tensor[:,2] = X_batch[:,10]
        total_res_tensor[:,3] = X_batch[:,11]
        total_res_tensor[:,4] = X_batch[:,0]
        total_res_tensor[:,5] = X_batch[:,1]
        total_res_tensor[:,6] = X_batch[:,2]
        total_res_tensor[:,7] = X_batch[:,3]
        total_res_tensor[:,8] = X_batch[:,4]
        total_res_tensor[:,9] = X_batch[:,5]
        total_res_tensor[:,10] = X_batch[:,6]
        total_res_tensor[:,11] = X_batch[:,7]
        # tensor:  x, y,  t,  z, bx, by, vx, vy, rho, p, bz, vz
        # indices: 0, 1,  2,  3,  4,  5, 6,  7,  8,   9, 10, 11
        x_total_res = total_res_tensor[:, 0].squeeze() * x_plasma_bound[1] * 6371000.0
        y_total_res = total_res_tensor[:, 1].squeeze() * y_plasma_bound[1] * 6371000.0
        z_total_res = total_res_tensor[:, 3].squeeze() * z_plasma_bound[1] * 6371000.0
        pos_total_res = torch.cat((x_total_res.unsqueeze(1), y_total_res.unsqueeze(1), z_total_res.unsqueeze(1)), dim=1)
        m = torch.tensor([0, 0, 7.8e22], dtype=torch.float32)
        B_io, pos_vector, mask_forplot = magnetic_field(pos_total_res, m, io_bound=1e10)

        Bx_masked = B_io[:, 0]  # All elements in the last dimension, index 0
        By_masked = B_io[:, 1]  # All elements in the last dimension, index 1
        Bz_masked = B_io[:, 2]  # All elements in the last dimension, index 2

        B_coords_msk_forplot = torch.cat([pos_vector, Bx_masked.unsqueeze(1), By_masked.unsqueeze(1), Bz_masked.unsqueeze(1)], dim=1)
        bx_lim = 4.4183e-05 # T
        by_lim = 4.4183e-05 # T
        bz_lim = 5.9763e-05
        # scale B_ionosphere
        B_inner_tensor = torch.zeros_like(B_coords_msk_forplot)
        B_inner_tensor[:, 0] = total_res_tensor[:, 0].squeeze()
        B_inner_tensor[:, 1] = total_res_tensor[:, 1].squeeze()
        B_inner_tensor[:, 2] = total_res_tensor[:, 3].squeeze()
        B_inner_tensor[:, 3] = B_coords_msk_forplot[:, 3] / bx_lim
        B_inner_tensor[:, 4] = B_coords_msk_forplot[:, 4] / by_lim
        B_inner_tensor[:, 5] = B_coords_msk_forplot[:, 5] / bz_lim
        total_res_tensor = total_res_tensor[mask_forplot]

        total_res_tensor[:, 4] = (total_res_tensor[:,4].squeeze() + B_inner_tensor[:, 3].squeeze()) / 2
        total_res_tensor[:, 5] = (total_res_tensor[:,5].squeeze() + B_inner_tensor[:, 4].squeeze()) / 2
        total_res_tensor[:, 10] = (total_res_tensor[:,10].squeeze() + B_inner_tensor[:, 5].squeeze()) / 2
        
        X_batch= torch.zeros_like(total_res_tensor)
        X_batch[:,0] = total_res_tensor[:,4]
        X_batch[:,1] = total_res_tensor[:,5]
        X_batch[:,2] = total_res_tensor[:,6]
        X_batch[:,3] = total_res_tensor[:,7]
        X_batch[:,4] = total_res_tensor[:,8]
        X_batch[:,5] = total_res_tensor[:,9]
        X_batch[:,6] = total_res_tensor[:,10]
        X_batch[:,7] = total_res_tensor[:,11]
        X_batch[:,8] = total_res_tensor[:,0]
        X_batch[:,9] = total_res_tensor[:,1]
        X_batch[:,10] = total_res_tensor[:,2]
        X_batch[:,11] = total_res_tensor[:,3]
        
        Temp_batch = CRRES_batch[:,22:23].squeeze() / 1.060000e+06
        AE_batch = CRRES_batch[:,23:24].squeeze() / 1.934500e+03
        symh_batch = CRRES_batch[:,24:25].squeeze() / 5.581818e+01
        asyh_batch = CRRES_batch[:,25:26].squeeze() / 1.602727e+02
        tenmev_batch = CRRES_batch[:,26:27].squeeze() / 4.325000e+02
        thirtymev_batch = CRRES_batch[:,27:28].squeeze() / 6.020000e+01
        sixtymev_batch = CRRES_batch[:,28:29].squeeze() / 2.250000e+01
        omni_batch = torch.cat((Temp_batch.unsqueeze(1), AE_batch.unsqueeze(1), symh_batch.unsqueeze(1), asyh_batch.unsqueeze(1), tenmev_batch.unsqueeze(1), thirtymev_batch.unsqueeze(1), sixtymev_batch.unsqueeze(1)), dim=1)
        X_batch_full = torch.cat((X_batch, omni_batch), dim=1)
        # print(X_batch.shape)
        # print(omni_batch.shape)
        X_batch = concatenate_previous_time_steps(X_batch_full, max_time_step=max_tau)
        #print(X_batch.shape)


        

        loss, loss_list, loss_list_real = loss_fn(params, X_batch, CRRES_batch)
        # update the parameters with functional optimizer
        
        loss.backward()

        optimizer.step()
        #scheduler.step(loss)

        #params_list = new_params_list

        print(f"Iteration {i} with loss {float(loss)}")
        loss_evolution.append(float(loss))
        indi_loss_evo.append(loss_list)
    # loss plots
    evolution_transpose = list(map(list, zip(*indi_loss_evo)))
    fig, ax = plt.subplots()
    ax.semilogy(loss_evolution)
    ax.set(title="Total Loss evolution", xlabel="# epochs", ylabel="Total Loss MSE")
    fig2, ax2 = plt.subplots()
    for i, param_values in enumerate(evolution_transpose):
        ax2.plot(range(len(param_values)), param_values, label=f'loss of flux energy {i+1}')
    ax2.set(title="Flux energy channel Losses over training", xlabel="# epochs", ylabel="Loss MSE")
    ax2.legend()
      
    ## making validation plot:
    
    # plot of one channel over time compared with the prediction on that channel:
    f1_overt = df_val_flux_tensor[:,2:3][max_tau:].squeeze().detach().numpy() #/ f1_lim
    f10_overt = df_val_flux_tensor[:,11:12][max_tau:].squeeze().detach().numpy()
    f15_overt = df_val_flux_tensor[:,16:17][max_tau:].squeeze().detach().numpy()
    f1_t = df_val_flux_tensor[:,0:1].squeeze()
    L_forplot = df_val_flux_tensor[:,1:2][max_tau:].squeeze().detach().numpy()
    f1_t_forplot = f1_t[max_tau:].detach().numpy()
    f1_t_forpred = f1_t / time_plasma_lim
    f1_x = df_val_flux_tensor[:,19:20].squeeze() / x_plasma_bound[1]
    f1_y = df_val_flux_tensor[:,20:21].squeeze() / y_plasma_bound[1]
    f1_z = df_val_flux_tensor[:,21:22].squeeze() / z_plasma_bound[1]
    # generates the X based on f1 over time
    X_plot_prep = torch.cat((f1_x.unsqueeze(1), f1_y.unsqueeze(1), f1_t_forpred.unsqueeze(1), f1_z.unsqueeze(1)), dim=1)
    X_pred_plasma = funcs_plasma[0](X_plot_prep, params_plasma)
    
    #X_pred_plasma = torch.zeros_like(X_pred_plasma)
    X_full_pred = torch.cat((X_pred_plasma, X_plot_prep), dim=1)
    X_batch = X_full_pred
    
    total_res_tensor = torch.zeros_like(X_batch)
    total_res_tensor[:,0] = X_batch[:,8]
    total_res_tensor[:,1] = X_batch[:,9]
    total_res_tensor[:,2] = X_batch[:,10]
    total_res_tensor[:,3] = X_batch[:,11]
    total_res_tensor[:,4] = X_batch[:,0]
    total_res_tensor[:,5] = X_batch[:,1]
    total_res_tensor[:,6] = X_batch[:,2]
    total_res_tensor[:,7] = X_batch[:,3]
    total_res_tensor[:,8] = X_batch[:,4]
    total_res_tensor[:,9] = X_batch[:,5]
    total_res_tensor[:,10] = X_batch[:,6]
    total_res_tensor[:,11] = X_batch[:,7]
    # tensor:  x, y,  t,  z, bx, by, vx, vy, rho, p, bz, vz
    # indices: 0, 1,  2,  3,  4,  5, 6,  7,  8,   9, 10, 11
    x_total_res = total_res_tensor[:, 0].squeeze() * x_plasma_bound[1] * 6371000.0
    y_total_res = total_res_tensor[:, 1].squeeze() * y_plasma_bound[1] * 6371000.0
    z_total_res = total_res_tensor[:, 3].squeeze() * z_plasma_bound[1] * 6371000.0
    pos_total_res = torch.cat((x_total_res.unsqueeze(1), y_total_res.unsqueeze(1), z_total_res.unsqueeze(1)), dim=1)
    m = torch.tensor([0, 0, 7.8e22], dtype=torch.float32)
    B_io, pos_vector, mask_forplot = magnetic_field(pos_total_res, m, io_bound=1e10)

    Bx_masked = B_io[:, 0]  # All elements in the last dimension, index 0
    By_masked = B_io[:, 1]  # All elements in the last dimension, index 1
    Bz_masked = B_io[:, 2]  # All elements in the last dimension, index 2

    B_coords_msk_forplot = torch.cat([pos_vector, Bx_masked.unsqueeze(1), By_masked.unsqueeze(1), Bz_masked.unsqueeze(1)], dim=1)
    bx_lim = 4.4183e-05 # T
    by_lim = 4.4183e-05 # T
    bz_lim = 5.9763e-05
    # scale B_ionosphere
    B_inner_tensor = torch.zeros_like(B_coords_msk_forplot)
    B_inner_tensor[:, 0] = B_coords_msk_forplot[:, 0] / (x_plasma_bound[1] * 6371000.0)
    B_inner_tensor[:, 1] = B_coords_msk_forplot[:, 1] / (y_plasma_bound[1] * 6371000.0)
    B_inner_tensor[:, 2] = B_coords_msk_forplot[:, 2] / (z_plasma_bound[1] * 6371000.0)
    B_inner_tensor[:, 3] = B_coords_msk_forplot[:, 3] / bx_lim
    B_inner_tensor[:, 4] = B_coords_msk_forplot[:, 4] / by_lim
    B_inner_tensor[:, 5] = B_coords_msk_forplot[:, 5] / bz_lim
    total_res_tensor = total_res_tensor[mask_forplot]
    
    total_res_tensor[:, 4] = (total_res_tensor[:,4].squeeze() + B_inner_tensor[:, 3].squeeze()) / 2
    total_res_tensor[:, 5] = (total_res_tensor[:,5].squeeze() + B_inner_tensor[:, 4].squeeze()) / 2
    total_res_tensor[:, 10] = (total_res_tensor[:,10].squeeze() + B_inner_tensor[:, 5].squeeze()) / 2
    
    X_batch= torch.zeros_like(total_res_tensor)
    X_batch[:,0] = total_res_tensor[:,4]
    X_batch[:,1] = total_res_tensor[:,5]
    X_batch[:,2] = total_res_tensor[:,6]
    X_batch[:,3] = total_res_tensor[:,7]
    X_batch[:,4] = total_res_tensor[:,8]
    X_batch[:,5] = total_res_tensor[:,9]
    X_batch[:,6] = total_res_tensor[:,10]
    X_batch[:,7] = total_res_tensor[:,11]
    X_batch[:,8] = total_res_tensor[:,0]
    X_batch[:,9] = total_res_tensor[:,1]
    X_batch[:,10] = total_res_tensor[:,2]
    X_batch[:,11] = total_res_tensor[:,3]
    X_full_pred = X_batch
    Bz_array = X_pred_plasma[:,6:7][max_tau:].squeeze().detach().numpy()
    Temp_batch = df_val_flux_tensor[:,22:23].squeeze() / 1.060000e+06
    AE_batch = df_val_flux_tensor[:,23:24].squeeze() / 1.934500e+03
    symh_batch = df_val_flux_tensor[:,24:25].squeeze() / 5.581818e+01
    asyh_batch = df_val_flux_tensor[:,25:26].squeeze() / 1.602727e+02
    tenmev_batch = df_val_flux_tensor[:,26:27].squeeze() / 4.325000e+02
    thirtymev_batch = df_val_flux_tensor[:,27:28].squeeze() / 6.020000e+01
    sixtymev_batch = df_val_flux_tensor[:,28:29].squeeze() / 2.250000e+01
    omni_batch = torch.cat((Temp_batch.unsqueeze(1), AE_batch.unsqueeze(1), symh_batch.unsqueeze(1), asyh_batch.unsqueeze(1), tenmev_batch.unsqueeze(1), thirtymev_batch.unsqueeze(1), sixtymev_batch.unsqueeze(1)), dim=1)
    X_full_pred = torch.cat((X_full_pred, omni_batch), dim=1)
    X_full_pred = concatenate_previous_time_steps(X_full_pred, max_tau)
    # generates predicted unscaled back up
    pred_tensor = funcs[0](X_full_pred, params)
    f1_pred = pred_tensor[:,0:1].squeeze() * f1_lim
    f10_pred = pred_tensor[:,9:10].squeeze() * f10_lim
    f15_pred = pred_tensor[:,14:15].squeeze() * f15_lim
    f1_pred_forplot = f1_pred.detach().numpy()
    f10_pred_forplot = f10_pred.detach().numpy()
    f15_pred_forplot = f15_pred.detach().numpy()
    # Convert time_array to a pandas Series with a datetime index
    time_series = pd.Series(index=pd.to_datetime(f1_t_forplot, unit='s'))

    # Convert parameter_array to a pandas Series with the same index
    parameter_series_flux_r = pd.Series(f15_overt, index=time_series.index)
    parameter_series_flux_pred = pd.Series(f15_pred_forplot, index=time_series.index)
    parameter_series_L = pd.Series(L_forplot, index=time_series.index)

    # Resample the parameter Series to hourly intervals, taking the mean
    hourly_average_r_f = parameter_series_flux_r.resample('D').mean()
    hourly_average_flux_pred = parameter_series_flux_pred.resample('D').mean()
    hourly_average_L = parameter_series_L.resample('D').mean()

    def create_map_tensors(index_row):
        # random_row = df_val_flux.iloc[index_row]
        
        # specific_row_series = df_val_flux.iloc[index_row]
        # specific_row_dataframe = specific_row_series.to_frame().T
        specific_row_dataframe = df_val_flux.sample(n=1)
        df_val_flux_numpy_arr = specific_row_dataframe.values
        flux_tensor_fullmap = torch.from_numpy(df_val_flux_numpy_arr).float()
        
        time_fullmap_value = float(flux_tensor_fullmap[:,0:1].squeeze())
        Temp_fullmap_value = float(flux_tensor_fullmap[:,22:23].squeeze()) / 1.060000e+06
        AE_fullmap_value = float(flux_tensor_fullmap[:,23:24].squeeze()) / 1.934500e+03
        symh_fullmap_value = float(flux_tensor_fullmap[:,24:25].squeeze()) / 5.581818e+01
        asyh_fullmap_value = float(flux_tensor_fullmap[:,25:26].squeeze()) / 1.602727e+02
        tenmev_fullmap_value = float(flux_tensor_fullmap[:,26:27].squeeze()) / 4.325000e+02
        thirtymev_fullmap_value = float(flux_tensor_fullmap[:,27:28].squeeze()) / 6.020000e+01
        sixtymev_fullmap_value = float(flux_tensor_fullmap[:,28:29].squeeze()) / 2.250000e+01
        
        n_points = 200
        
        
        x_points = torch.linspace(-1.0, 1.0, n_points)
        z_points = torch.linspace(-1.0, 1.0, n_points)
        # Create a 2D grid of points for x and y
        x_grid, z_grid = torch.meshgrid(x_points,  z_points, indexing="xy")
        xz_grid = torch.stack([x_grid, z_grid], dim=-1).reshape(-1, 2)
        # Generate random values for the t dimension
        t_values = torch.full((xz_grid.shape[0], 1), time_fullmap_value) / time_plasma_lim
        y_values = torch.full((xz_grid.shape[0], 1), index_row)
        grid_points = torch.cat((xz_grid, y_values, t_values), dim=1)
        rearranged_tensor = grid_points[:, [0, 2, 3, 1]]
        grid_points = rearranged_tensor
        plasma_fullmap = funcs_plasma[0](grid_points, params_plasma)
        
        X_full_pred = torch.cat((plasma_fullmap, grid_points), dim=1)
        X_batch = X_full_pred
        
        total_res_tensor = torch.zeros_like(X_batch)
        total_res_tensor[:,0] = X_batch[:,8]
        total_res_tensor[:,1] = X_batch[:,9]
        total_res_tensor[:,2] = X_batch[:,10]
        total_res_tensor[:,3] = X_batch[:,11]
        total_res_tensor[:,4] = X_batch[:,0]
        total_res_tensor[:,5] = X_batch[:,1]
        total_res_tensor[:,6] = X_batch[:,2]
        total_res_tensor[:,7] = X_batch[:,3]
        total_res_tensor[:,8] = X_batch[:,4]
        total_res_tensor[:,9] = X_batch[:,5]
        total_res_tensor[:,10] = X_batch[:,6]
        total_res_tensor[:,11] = X_batch[:,7]
        # tensor:  x, y,  t,  z, bx, by, vx, vy, rho, p, bz, vz
        # indices: 0, 1,  2,  3,  4,  5, 6,  7,  8,   9, 10, 11
        x_total_res = total_res_tensor[:, 0].squeeze() * x_plasma_bound[1] * 6371000.0
        y_total_res = total_res_tensor[:, 1].squeeze() * y_plasma_bound[1] * 6371000.0
        z_total_res = total_res_tensor[:, 3].squeeze() * z_plasma_bound[1] * 6371000.0
        pos_total_res = torch.cat((x_total_res.unsqueeze(1), y_total_res.unsqueeze(1), z_total_res.unsqueeze(1)), dim=1)
        m = torch.tensor([0, 0, 7.8e22], dtype=torch.float32)
        B_io, pos_vector, mask_forplot = magnetic_field(pos_total_res, m, io_bound=1e10)

        Bx_masked = B_io[:, 0]  # All elements in the last dimension, index 0
        By_masked = B_io[:, 1]  # All elements in the last dimension, index 1
        Bz_masked = B_io[:, 2]  # All elements in the last dimension, index 2

        B_coords_msk_forplot = torch.cat([pos_vector, Bx_masked.unsqueeze(1), By_masked.unsqueeze(1), Bz_masked.unsqueeze(1)], dim=1)
        bx_lim = 4.4183e-05 # T
        by_lim = 4.4183e-05 # T
        bz_lim = 5.9763e-05
        # scale B_ionosphere
        B_inner_tensor = torch.zeros_like(B_coords_msk_forplot)
        B_inner_tensor[:, 0] = B_coords_msk_forplot[:, 0] / (x_plasma_bound[1] * 6371000.0)
        B_inner_tensor[:, 1] = B_coords_msk_forplot[:, 1] / (y_plasma_bound[1] * 6371000.0)
        B_inner_tensor[:, 2] = B_coords_msk_forplot[:, 2] / (z_plasma_bound[1] * 6371000.0)
        B_inner_tensor[:, 3] = B_coords_msk_forplot[:, 3] / bx_lim
        B_inner_tensor[:, 4] = B_coords_msk_forplot[:, 4] / by_lim
        B_inner_tensor[:, 5] = B_coords_msk_forplot[:, 5] / bz_lim
        total_res_tensor = total_res_tensor[mask_forplot]
        
        total_res_tensor[:, 4] = (total_res_tensor[:,4].squeeze() + B_inner_tensor[:, 3].squeeze()) / 2
        total_res_tensor[:, 5] = (total_res_tensor[:,5].squeeze() + B_inner_tensor[:, 4].squeeze()) / 2
        total_res_tensor[:, 10] = (total_res_tensor[:,10].squeeze() + B_inner_tensor[:, 5].squeeze()) / 2
        
        X_batch= torch.zeros_like(total_res_tensor)
        X_batch[:,0] = total_res_tensor[:,4]
        X_batch[:,1] = total_res_tensor[:,5]
        X_batch[:,2] = total_res_tensor[:,6]
        X_batch[:,3] = total_res_tensor[:,7]
        X_batch[:,4] = total_res_tensor[:,8]
        X_batch[:,5] = total_res_tensor[:,9]
        X_batch[:,6] = total_res_tensor[:,10]
        X_batch[:,7] = total_res_tensor[:,11]
        X_batch[:,8] = total_res_tensor[:,0]
        X_batch[:,9] = total_res_tensor[:,1]
        X_batch[:,10] = total_res_tensor[:,2]
        X_batch[:,11] = total_res_tensor[:,3]
        X_full_pred = X_batch
        Temp_batch = torch.full_like(X_batch[:,11].squeeze(), Temp_fullmap_value)
        AE_batch = torch.full_like(X_batch[:,11].squeeze(), AE_fullmap_value)
        symh_batch = torch.full_like(X_batch[:,11].squeeze(), symh_fullmap_value)
        asyh_batch = torch.full_like(X_batch[:,11].squeeze(), asyh_fullmap_value)
        tenmev_batch = torch.full_like(X_batch[:,11].squeeze(), tenmev_fullmap_value)
        thirtymev_batch = torch.full_like(X_batch[:,11].squeeze(), thirtymev_fullmap_value)
        sixtymev_batch = torch.full_like(X_batch[:,11].squeeze(), sixtymev_fullmap_value)
        omni_batch = torch.cat((Temp_batch.unsqueeze(1), AE_batch.unsqueeze(1), symh_batch.unsqueeze(1), asyh_batch.unsqueeze(1), tenmev_batch.unsqueeze(1), thirtymev_batch.unsqueeze(1), sixtymev_batch.unsqueeze(1)), dim=1)
        X_full_pred = torch.cat((X_full_pred, omni_batch), dim=1)
        X_full_pred = concatenate_previous_time_steps(X_full_pred, max_tau)
        # generates predicted unscaled back up
        return X_full_pred, x_total_res, z_total_res
    
    
    
    
### plots
    fig2, ax2 = plt.subplots(figsize=(24, 8))
    ax2.plot(hourly_average_r_f.index, hourly_average_r_f, label='real flux average', color='black')
    ax2.plot(hourly_average_flux_pred.index, hourly_average_flux_pred, label='Flux prediction average', color='red', alpha=0.8)
    ax2.grid(True)
    # ax2.set_xlabel('Date')
    ax2.set_xticks([])
    ax2.set_ylabel('Log Average Flux')
    ax2.set_title('Real log average hourly Flux from CRRES vs prediction made with P.I.N.N at one measured energy channel')  # Adjust title as needed
    ax2.legend()
    
    # # fig3, ax3 = plt.subplots(figsize=(24, 3))
    # # scatter = ax3.scatter(f1_t_forplot, L_forplot, c=f1_overt, vmin = np.min(f1_overt), vmax = np.max(f1_overt))
    # # ax3.grid(True)
    # # ax3.set_xlabel('time (s) from IC')
    # # ax3.set_ylabel('Real log Average Flux')
    # # ax3.legend()
    
    # # fig5, ax5 = plt.subplots(figsize=(24, 3))
    # # scatter2 = ax5.scatter(f1_t_forplot, L_forplot, c=f1_pred_forplot, vmin = np.min(f1_overt), vmax = np.max(f1_overt))
    # # ax5.grid(True)
    # # ax5.set_xlabel('time (s) from IC')
    # # ax5.set_ylabel('P.I.N.N model predicted log Average flux')
    # # ax5.legend()
    
    # # Create a figure with 2 subplots arranged vertically
    # #fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, figsize=(36, 3))
    # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, figsize=(36, 3))

    # # Plot the first scatterplot
    # scatter1 = ax1.scatter(f1_t_forplot, L_forplot, c=f1_overt, cmap='viridis', marker='s', vmin = np.min(f1_overt), vmax = np.max(f1_overt))
    # ax1.grid(True)
    # ax1.set_xlabel('time (s) from IC')
    # #ax1.set_ylabel('Obs 148 keV')

    # # Plot the second scatterplot
    # scatter2 = ax2.scatter(f1_t_forplot, L_forplot, c=f1_pred_forplot, cmap='viridis', marker='s', vmin = np.min(f1_overt), vmax = np.max(f1_overt))
    # ax2.grid(True)
    # ax2.set_xlabel('time (s) from IC')
    # #ax2.set_ylabel('Pred 148 keV')
    
    # scatter3 = ax3.scatter(f1_t_forplot, L_forplot, c=f10_overt, cmap='viridis', marker='s', vmin = np.min(f10_overt), vmax = np.max(f10_overt))
    # ax3.grid(True)
    # ax3.set_xlabel('time (s) from IC')
    # #ax3.set_ylabel('Obs')

    # # Plot the second scatterplot
    # scatter4 = ax4.scatter(f1_t_forplot, L_forplot, c=f10_pred_forplot, cmap='viridis', marker='s', vmin = np.min(f10_overt), vmax = np.max(f10_overt))
    # ax4.grid(True)
    # ax4.set_xlabel('time (s) from IC')
    # #ax4.set_ylabel('Pred')

    # scatter5 = ax5.scatter(f1_t_forplot, L_forplot, c=f15_overt, cmap='viridis', marker='s', vmin = np.min(f15_overt), vmax = np.max(f15_overt))
    # ax5.grid(True)
    # ax5.set_xlabel('time (s) from IC')
    # #ax5.set_ylabel('Obs')

    # # Plot the second scatterplot
    # scatter6 = ax6.scatter(f1_t_forplot, L_forplot, c=f15_pred_forplot, cmap='viridis', marker='s', vmin = np.min(f15_overt), vmax = np.max(f15_overt))
    # ax6.grid(True)
    # ax6.set_xlabel('time (s) from IC')
    # #ax6.set_ylabel('Pred')
    
    # ax1.set_title('Measured vs. Predicted at 148 keV')
    # ax3.set_title('Measured vs. Predicted at 876 keV')
    # ax5.set_title('Measured vs. Predicted at 1368 keV')
    
    # # Add a colorbar for both subplots
    # cbar1 = plt.colorbar(scatter1, ax=ax1)
    # cbar1.set_label('log(Flux)')

    # cbar2 = plt.colorbar(scatter2, ax=ax2)
    # cbar2.set_label('log(Flux)')
    
    # cbar3 = plt.colorbar(scatter3, ax=ax3)
    # cbar3.set_label('log(Flux)')

    # cbar4 = plt.colorbar(scatter4, ax=ax4)
    # cbar4.set_label('log(Flux)')
    
    # cbar5 = plt.colorbar(scatter5, ax=ax5)
    # cbar5.set_label('log(Flux)')

    # cbar6 = plt.colorbar(scatter6, ax=ax6)
    # cbar6.set_label('log(Flux)')
    
    # fig4, ax4 = plt.subplots()
    # ax4.plot(f1_t_forplot, Bz_array, label='Bz from plasma model', color='black')
    # #ax4.plot(f1_t_forplot, f1_pred_forplot, label='Flux prediction', color='red', alpha=0.8)
    # ax4.grid(True)
    # ax4.set_xlabel('Date')
    # ax4.set_ylabel('Log Average Flux')
    # ax4.set_title('Real log average hourly Flux from CRRES vs prediction made with P.I.N.N at one measured energy channel')  # Adjust title as needed
    # ax4.legend()
    
    y_list = [-1.0,-0.5,0.0,0.5, 1.0]
    for i in y_list:
        y_value = i
        X_full_pred, x_total_res, z_total_res = create_map_tensors(y_value)
        pred_tensor = funcs[0](X_full_pred, params)
        f1_pred = pred_tensor[:,0:1].squeeze() * f1_lim
        f10_pred = pred_tensor[:,9:10].squeeze() * f10_lim
        f15_pred = pred_tensor[:,14:15].squeeze() * f15_lim
        f1_pred_forplot = f1_pred.detach().numpy()
        f10_pred_forplot = f10_pred.detach().numpy()
        f15_pred_forplot = f15_pred.detach().numpy()
        width, height = 6, 6
        
        fig4, ax4 = plt.subplots(figsize=(width, height))
        scatter_flux = ax4.scatter(x_total_res[max_tau:].detach().numpy(), z_total_res[max_tau:].detach().numpy(), c=f15_pred_forplot, cmap='Spectral', s=3)
        plt.colorbar(scatter_flux, label='Flux (Log Average 1/cm^2*s*sr*kev)')
        #ax4.plot(f1_t_forplot, f1_pred_forplot, label='Flux prediction', color='red', alpha=0.8)
        full_circle = patches.Circle((0, 0), radius=6371000.0, color='black')
        ax4.add_patch(full_circle)
        semicircle_white = patches.Wedge((0, 0), 6371000.0, -90, 90, color='white')
        ax4.add_patch(semicircle_white)
        ax4.set(title=f"2D Map of Flux (Log Average) 1368 KeV at y-slice {i * y_plasma_bound[1] * 6371000.0} from IC", xlabel="X GSE (m)", ylabel="Z GSE (m)")  # Adjust title as needed
        ax4.set_aspect('equal', 'box')
    
    
    plt.show()  
    
    
    variance = np.var(hourly_average_r_f)
    MSE = np.mean((hourly_average_r_f - hourly_average_flux_pred) ** 2)
    PE = 1- (MSE/variance)
    print(PE)
    
    import matplotlib.animation as animation

    # Function to update the plot for each frame of the animation
    def update_plot(frame):
        y_value = frame / 100.0 - 1.0  # Adjust the step and start point as needed
        X_full_pred, x_total_res, z_total_res = create_map_tensors(y_value)
        pred_tensor = funcs[0](X_full_pred, params)
        f15_pred = pred_tensor[:,14:15].squeeze() * f15_lim
        f15_pred_forplot = f15_pred.detach().numpy()
        
        scatter_flux.set_offsets(np.column_stack((x_total_res[max_tau:].detach().numpy(), z_total_res[max_tau:].detach().numpy())))
        scatter_flux.set_array(f15_pred_forplot)
        ax4.set_title(f"2D Map of Flux (Log Average) 1368 KeV at y-slice {y_value * y_plasma_bound[1] * 6371000.0} from IC")
        print('1 step done')

    # Create the figure and axes outside the loop
    fig4, ax4 = plt.subplots(figsize=(width, height))
    X_full_pred, x_total_res, z_total_res = create_map_tensors(-1.0)
    pred_tensor = funcs[0](X_full_pred, params)
    f15_pred = pred_tensor[:,14:15].squeeze() * f15_lim
    f15_pred_forplot = f15_pred.detach().numpy()
    scatter_flux = ax4.scatter(x_total_res[max_tau:].detach().numpy(), z_total_res[max_tau:].detach().numpy(), c=f15_pred_forplot, cmap='Spectral', s=3)
    plt.colorbar(scatter_flux, label='Flux (Log Average 1/cm^2*s*sr*keV)')
    # full_circle = patches.Circle((0, 0), radius=6371000.0, color='black')
    # ax4.add_patch(full_circle)
    # semicircle_white = patches.Wedge((0, 0), 6371000.0, -90, 90, color='white')
    # ax4.add_patch(semicircle_white)
    ax4.set(title="2D Map of Flux (Log Average) 1368 KeV", xlabel="X GSE (m)", ylabel="Z GSE (m)")
    ax4.set_aspect('equal', 'box')

    # Create the animation
    ani = animation.FuncAnimation(fig4, update_plot, frames=range(200), interval=200)
    ani.save('C:/Users/User/Desktop/PINN_torch/flux_animation.mp4', writer='ffmpeg')