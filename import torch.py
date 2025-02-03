import torch
import pandas as pd

df_val = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_cutdown.csv', index_col=0)
batch_size = 30
tolerance = 1e-8
#learning_rate_0 = args.learning_rate
domain_x = (df_val['X_(S/C)__GSE_Re'].min(), df_val['X_(S/C)__GSE_Re'].max())
domain_y = (df_val['Y_(S/C)__GSE_Re'].min(), df_val['Y_(S/C)__GSE_Re'].max())
domain_t = (df_val['time'].min(), df_val['time'].max())
domain_z = (df_val['Z_(S/C)__GSE_Re'].min(), df_val['Z_(S/C)__GSE_Re'].max())
new_max, new_min = -1.0, 1.0
x_dim = torch.FloatTensor(batch_size).uniform_(domain_x[0], domain_x[1]).requires_grad_(True)#.unsqueeze(1)
x_max, x_min = x_dim.max(), x_dim.min()
scaled_minmax = (x_dim  - x_min)/(x_max - x_min)*(new_max - new_min) + new_min
x_dim = scaled_minmax
y_dim = torch.FloatTensor(batch_size).uniform_(domain_y[0], domain_y[1]).requires_grad_(True)#.unsqueeze(1)
y_max, y_min = y_dim.max(), y_dim.min()
scaled_minmax = (y_dim  - y_min)/(y_max - y_min)*(new_max - new_min) + new_min
y_dim = scaled_minmax
t_dim = torch.FloatTensor(batch_size).uniform_(domain_t[0], domain_t[1]).requires_grad_(True)#.unsqueeze(1)
t_max, t_min = t_dim.max(), t_dim.min()
scaled_minmax = (t_dim  - t_min)/(t_max - t_min)*(new_max - new_min) + new_min
t_dim = scaled_minmax
z_dim = torch.FloatTensor(batch_size).uniform_(domain_z[0], domain_z[1]).requires_grad_(True)
z_max, z_min = z_dim.max(), z_dim.min()
scaled_minmax = (z_dim  - z_min)/(z_max - z_min)*(new_max - new_min) + new_min
z_dim = scaled_minmax
x = torch.cat((x_dim.unsqueeze(1), y_dim.unsqueeze(1), t_dim.unsqueeze(1), z_dim.unsqueeze(1)), dim=1)
df_GOES_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/combined_ephem_mag.csv')
GOES_data_numpy_arr = df_GOES_BC.values
G_BC_tensor = torch.from_numpy(GOES_data_numpy_arr).float()
indices = torch.randperm(G_BC_tensor.size(0))[:batch_size]
boundary_sample = G_BC_tensor[indices]
new_max, new_min = 1.0, -1.0
newt_max, newt_min = 1.0, 0.0
t_boundary = boundary_sample[:, 0:1].squeeze()

import torch

# Generate a random tensor of shape [300, 4]
tensor = 10 * torch.randn(300, 4)

selected_elements = tensor[:, [0, 1, 3]]

# Calculate the square root of the sum of squares for the first three parameters
magnitudes = torch.sqrt((selected_elements[:, 1:4] ** 2).sum(dim=1))

# Define a threshold for the magnitude
threshold = 6.0

# Create a mask where the magnitude is below the threshold
mask = magnitudes < threshold

# Use the mask to select rows from the original tensor
filtered_tensor = tensor[mask]

remaining_tensor = tensor[~mask]
df_EARTH_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/GSE_ver.csv')
df_GOES_BC = pd.read_csv('C:/Users/User/Desktop/PINN_torch/combined_ephem_mag.csv')
# Output
# print("Filtered Tensor:")
# print(filtered_tensor)
# print("Filtered Tensor Shape:", filtered_tensor.shape)
# print("other Tensor Shape:", remaining_tensor.shape)

for col in df_EARTH_BC.columns:
    try:
        df_EARTH_BC[col] = pd.to_numeric(df_EARTH_BC[col], errors='raise')
    except ValueError:
        print(f"Column {col} cannot be converted to numeric.{col}")
        
#print(df_EARTH_BC.head())
print(df_val['X_(S/C)__GSE_Re'].max())


# Bx_Gboundary_value = F_Gboundary_tensor[:,0:1].squeeze() - scale_sample(G_boundary_sample[:,4:5].squeeze())
        # By_Gboundary_value = F_Gboundary_tensor[:,1:2].squeeze() - scale_sample(G_boundary_sample[:,5:6].squeeze())
        # Bz_Gboundary_value = F_Gboundary_tensor[:,6:7].squeeze() - scale_sample(G_boundary_sample[:,6:7].squeeze())
        
        # G_boundary_sum = Bx_Gboundary_value + By_Gboundary_value + Bz_Gboundary_value
        # G_boundary = G_boundary_sum  /  3

#bc_loss = loss(F_boundary_tensor, Sample_tensor)
# Bx_boundary_value =  F_boundary_tensor[:,0:1].squeeze() - scale_sample(boundary_sample[:,1:2].squeeze())
# By_boundary_value = F_boundary_tensor[:,1:2].squeeze() - scale_sample(boundary_sample[:,2:3].squeeze())
# Vx_boundary_value = F_boundary_tensor[:,2:3].squeeze() - scale_sample(boundary_sample[:,3:4].squeeze())
# Vy_boundary_value = F_boundary_tensor[:,3:4].squeeze() - scale_sample(boundary_sample[:,4:5].squeeze())
# rho_boundary_value = F_boundary_tensor[:,4:5].squeeze() - scale_sample(boundary_sample[:,5:6].squeeze())
# p_boundary_value = F_boundary_tensor[:,5:6].squeeze() - scale_sample(boundary_sample[:,6:7].squeeze())
# Bz_boundary_value = F_boundary_tensor[:,6:7].squeeze() - scale_sample(boundary_sample[:,10:11].squeeze())
# Vz_boundary_value = F_boundary_tensor[:,7:8].squeeze() - scale_sample(boundary_sample[:, 11:12].squeeze())

# boundary_sum = Bx_boundary_value + By_boundary_value + Vx_boundary_value + Vy_boundary_value + rho_boundary_value + p_boundary_value + Bz_boundary_value + Vz_boundary_value
# boundary = boundary_sum / 6
