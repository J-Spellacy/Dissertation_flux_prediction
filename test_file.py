import pandas as pd
from one_model_unscaled import make_tensor_for_fluxmodel
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
R_E = 6371000.0
#df = pd.read_csv('C:/Users/User/Desktop/PINN_torch/combined_ephem_mag_si_units.csv', index_col=0)
df = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_megacut_si_units.csv')
#print(6.0 / (df["Y_(S/C)__GSE_m"].max() / R_E))
df_val_flux = pd.read_csv('C:/Users/User/Desktop/PINN_torch/CRRES_GSE_si_units.csv')
#r = make_tensor_for_fluxmodel(df_flux=df_val_flux, spatio_lengths=2)
print(df.abs().min())
# print(df_val_flux['B_T'].max())
#print(6371000.0 * 24)

# Ensure PyTorch uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
R_E = 6371000.0
M_m = 7.8e22
M_re = M_m /(R_E ** 2)
mu0 = (4 * np.pi * 1e-7) #/ 6371000.0 # Permeability of free space
m = torch.tensor([0, 0, M_m], dtype=torch.float32, device=device)  # Dipole moment (pointing in z-direction)
grid_size = 200
bound = 3.5 * R_E
range_x = (-bound, bound)
range_y = (-bound, bound)
range_z = (-bound, bound)

# Create a grid
x = torch.linspace(range_x[0], range_x[1], grid_size, device=device)
y = torch.linspace(range_y[0], range_y[1], grid_size, device=device)
z = torch.linspace(range_z[0], range_z[1], grid_size, device=device)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
#small_const = 1e-4
threshold = 3.5 * R_E
earth_radius_bound = R_E
# Calculate the magnetic field
def magnetic_field(X, Y, Z, m):
    r_vec = torch.stack((X, Y, Z), dim=-1)
    r = torch.norm(r_vec, dim=-1)
    r_hat = r_vec / r.unsqueeze(-1)
    m_dot_r_hat = torch.sum(m * r_hat, dim=-1)
    term1 = 3 * m_dot_r_hat.unsqueeze(-1) * r_hat
    term2 = m.view(1, 1, 1, 3)
    B = (mu0 / (4 * np.pi * r.unsqueeze(-1)**3)) * (term1 - term2)
    combined_mask = (r <= threshold) & (r >= earth_radius_bound)
    return B, combined_mask

B, mask = magnetic_field(X, Y, Z, m)

Bx = B[..., 0]  # All elements in the last dimension, index 0
By = B[..., 1]  # All elements in the last dimension, index 1
Bz = B[..., 2]  # All elements in the last dimension, index 2

coords = torch.stack([X, Y, Z], dim=-1)

Bx_masked = Bx[mask]
By_masked = By[mask]
Bz_masked = Bz[mask]
coords_masked = torch.stack([X[mask], Y[mask], Z[mask]], dim=1)
# Now flatten the coordinates tensor to shape [125000, 3] (for 50*50*50 grid points)
coords_flat = coords.reshape(-1, 3)
# Flatten the Bx, By, Bz tensors
Bx_flat = Bx.flatten()
By_flat = By.flatten()
Bz_flat = Bz.flatten()

B_coords_values_msk = torch.cat([coords_masked, Bx_masked.unsqueeze(1), By_masked.unsqueeze(1), Bz_masked.unsqueeze(1)], dim=1)
# By_coords_values_msk = torch.cat([coords_masked, By_masked.unsqueeze(1)], dim=1)
# Bz_coords_values_msk = torch.cat([coords_masked, Bz_masked.unsqueeze(1)], dim=1)
# You can create a tensor for each component with coordinates
# Combining coordinates with Bx values
Bx_coords_values = torch.cat([coords_flat, Bx_flat.unsqueeze(1)], dim=1)  # Adds Bx values as the fourth column
By_coords_values = torch.cat([coords_flat, By_flat.unsqueeze(1)], dim=1)
Bz_coords_values = torch.cat([coords_flat, Bz_flat.unsqueeze(1)], dim=1)

max_values = torch.max(B_coords_values_msk, dim=0)
print(Bx_coords_values.shape)
print(coords_flat.shape)
# print(max_values)
import matplotlib.pyplot as plt
import torch.optim as optim
  # Example grid size, adjust based on your actual dimensions
Bx_reconstructed = Bx_flat.view(grid_size, grid_size, grid_size)

# Select a mid-plane slice, e.g., along the z-axis
z_slice_index = grid_size // 2
Bx_slice = Bx_reconstructed[:, :, z_slice_index]

# Plot the slice
plt.figure(figsize=(10, 8))
plt.imshow(Bx_slice.cpu().numpy(), cmap='viridis', origin='lower')
plt.colorbar(label='Bx (magnetic field X-component)')
plt.title('Reconstructed Slice of the Bx Component at Z-Midplane')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
# Assume Bx is already defined and has shape (50, 50, 50)
# Let's plot the middle slice along the Z-axis

# z_slice_index = Bx.shape[2] // 2  # Middle index along Z-axis
# plt.imshow(Bx[:, :, z_slice_index], cmap='viridis', origin='lower')
# plt.colorbar(label='Bx (magnetic field X-component)')
# plt.title('Slice of the Bx Component at Z-Midplane')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
# Generate random data
# def generate_data(batch_size, grid_size, time_steps):
#     # Dimensions: (batch_size, depth, height, width, time, channels)
#     # Channels could include different features; we'll assume 1 for simplicity.
#     x = torch.rand(batch_size, grid_size, grid_size, grid_size, time_steps, 1, requires_grad=True)
#     y_velocity = torch.rand(batch_size, grid_size, grid_size, grid_size, 3)
#     y_field = torch.rand(batch_size, grid_size, grid_size, grid_size, 3)

#     return x, y_velocity, y_field
           
# batch_size = 20
# grid_size = 50  # Spatial dimensions: 5x5x5
# time_steps = 20

# x_train, y_train_velocity, y_train_field = generate_data(batch_size, grid_size, time_steps)
# print(x_train.shape)

# class VectorPredictionCNN(nn.Module):
#     def __init__(self):
#         super(VectorPredictionCNN, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv3d(32, 6, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         # x shape: (batch, channels, depth, height, width, time)
#         # We merge batch and time into one dimension for processing
#         batch_size, depth, height, width, time, channels = x.shape
#         x = x.view(batch_size * time, channels, depth, height, width)
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)  # This will have 6 channels at the output
#         # Split the two vector predictions
#         x = x.view(batch_size, time, 6, depth, height, width).permute(0, 3, 4, 5, 1, 2)
#         velocity = x[..., :3]
#         field = x[..., 3:]
#         return velocity, field

# model = VectorPredictionCNN()

# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Basic training loop
# for epoch in range(5):  # Run for 5 epochs for simplicity
#     optimizer.zero_grad()
#     predicted_velocity, predicted_field = model(x_train)
#     loss_velocity = criterion(predicted_velocity, y_train_velocity)
#     loss_field = criterion(predicted_field, y_train_field)
#     loss = loss_velocity + loss_field
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
# predicted_velocity, predicted_field = model(x_train)
# print("Predicted Velocity:", predicted_velocity.shape)
# print("Predicted Field:", predicted_field.shape)
    

# grad_BxdX = torch.autograd.grad(
#             inputs = x_train,
#             outputs = predicted_velocity,
#             grad_outputs=torch.ones_like(predicted_velocity),
#             retain_graph=True, 
#             create_graph=True
#         )[0]