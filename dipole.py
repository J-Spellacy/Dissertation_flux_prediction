
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure PyTorch uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
R_E = 6371000.0
M_m = 7.8e22
M_re = M_m /(R_E ** 2)
mu0 = (4 * np.pi * 1e-7) #/ 6371000.0 # Permeability of free space
m = torch.tensor([0, 0, M_m], dtype=torch.float32, device=device)  # Dipole moment (pointing in z-direction)
grid_size = 50
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
threshold = R_E

# Calculate the magnetic field
def magnetic_field(X, Y, Z, m):
    r_vec = torch.stack((X, Y, Z), dim=-1)
    print(r_vec.shape)
    r = torch.norm(r_vec, dim=-1)
    r_hat = r_vec / r.unsqueeze(-1)
    m_dot_r_hat = torch.sum(m * r_hat, dim=-1)
    term1 = 3 * m_dot_r_hat.unsqueeze(-1) * r_hat
    term2 = m.view(1, 1, 1, 3)
    B = (mu0 / (4 * np.pi * r.unsqueeze(-1)**3)) * (term1 - term2)
    mask = r < threshold  # Find where r is less than the threshold
    B[mask] = 0
    return B

B = magnetic_field(X, Y, Z, m)
print(torch.max(B), torch.min(B), torch.mean(B))
# print(X.shape)
# print(Y.shape)
# print(Z.shape)
# Magnitude of B (for visualization)
epsilon = 1e-10

B_magnitude = torch.sqrt(torch.sum(B**2, dim=-1))
B_log_magnitude = torch.log(B_magnitude + epsilon)
# B_normalized = (B / B_magnitude[..., None]) * B_log_magnitude[..., None]

# Helper function to plot a plane
lower_percentile, upper_percentile = 5, 99.9
vmin = np.percentile(B_log_magnitude.cpu().numpy(), lower_percentile)
vmax = np.percentile(B_log_magnitude.cpu().numpy(), upper_percentile)

def plot_plane_in_window(plane_data, title, xlabel, ylabel, scale=250):
    X, Y, U, V, C = plane_data
    fig, ax = plt.subplots(figsize=(8, 8))  # Create a new figure for each plot
    norm = np.sqrt(U**2 + V**2)
    U_norm = U / norm * np.log(norm + epsilon)
    V_norm = V / norm * np.log(norm + epsilon)
    # Compute log-scaled magnitudes for coloring
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    U_np = U_norm.cpu().numpy()
    V_np = V_norm.cpu().numpy()
    log_C = np.log(C.cpu().numpy() + epsilon)  # Ensure non-negative by adding epsilon before log

    # Plot vectors using the original magnitudes for vector lengths
    Q = ax.quiver(X_np, Y_np, U_np, V_np, log_C, scale=scale, cmap='viridis_r')#, clim=(vmin, vmax))
    #Q = ax.quiver(X.cpu().numpy(), Y.cpu().numpy(), U_norm.cpu().numpy(), V_norm.cpu().numpy(), C.cpu().numpy(), scale=scale)
    ax.set_title(title)
    ax.set_xlabel('X GSE (Re)')
    ax.set_ylabel('Y GSE (Re)')
    ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio to be equal
    cbar = fig.colorbar(Q, ax=ax, aspect=40)
    cbar.set_label('Log Field Magnitude log10(T)')
    plt.show()

# Data for each plane
plane_xy = (X[:, :, grid_size//2], Y[:, :, grid_size//2], B[:, :, grid_size//2, 0], B[:, :, grid_size//2, 1], B_magnitude[:, :, grid_size//2])
plane_xz = (X[:, grid_size//2, :], Z[:, grid_size//2, :], B[:, grid_size//2, :, 0], B[:, grid_size//2, :, 2], B_magnitude[:, grid_size//2, :])
plane_yz = (Y[grid_size//2, :, :], Z[grid_size//2, :, :], B[grid_size//2, :, :, 1], B[grid_size//2, :, :, 2], B_magnitude[grid_size//2, :, :])

# Create separate plots for each plane in new windows
plot_plane_in_window(plane_xy, 'XY plane at Z=0', 'X', 'Y')
plot_plane_in_window(plane_xz, 'XZ plane at Y=0', 'X', 'Z')
plot_plane_in_window(plane_yz, 'YZ plane at X=0', 'Y', 'Z')

# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # Helper function to plot a plane with appropriate normalization
# def plot_plane(ax, X, Y, U, V, C, title, xlabel, ylabel, scale=500):
#     norm = np.sqrt(U**2 + V**2)
#     U_norm = U / norm * np.log(norm + epsilon)
#     V_norm = V / norm * np.log(norm + epsilon)
#     Q = ax.quiver(X.cpu().numpy(), Y.cpu().numpy(), U_norm.cpu().numpy(), V_norm.cpu().numpy(), C.cpu().numpy(), scale=scale)
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xlim(-threshold, threshold)
#     ax.set_ylim(-threshold,threshold)
#     return Q

# # Plot each plane
# Q1 = plot_plane(axes[0], X[:, :, grid_size//2], Y[:, :, grid_size//2], B[:, :, grid_size//2, 0], B[:, :, grid_size//2, 1], B_magnitude[:, :, grid_size//2], 'XY plane at Z=0', 'X', 'Y')
# Q2 = plot_plane(axes[1], X[:, grid_size//2, :], Z[:, grid_size//2, :], B[:, grid_size//2, :, 0], B[:, grid_size//2, :, 2], B_magnitude[:, grid_size//2, :], 'XZ plane at Y=0', 'X', 'Z')
# Q3 = plot_plane(axes[2], Y[grid_size//2, :, :], Z[grid_size//2, :, :], B[grid_size//2, :, :, 1], B[grid_size//2, :, :, 2], B_magnitude[grid_size//2, :, :], 'YZ plane at X=0', 'Y', 'Z')

# # Add color bars
# cbar1 = fig.colorbar(Q1, ax=axes[0], aspect=40)
# cbar1.set_label('Field Magnitude (T)')
# cbar2 = fig.colorbar(Q2, ax=axes[1], aspect=40)
# cbar2.set_label('Field Magnitude (T)')
# cbar3 = fig.colorbar(Q3, ax=axes[2], aspect=40)
# cbar3.set_label('Field Magnitude (T)')

# plt.tight_layout()
# plt.show()

# # XY plane
# quiver0 = axes[0].quiver(X[:, :, grid_size//2].cpu().numpy(), Y[:, :, grid_size//2].cpu().numpy(), B[:, :, grid_size//2, 0].cpu().numpy(), B[:, :, grid_size//2, 1].cpu().numpy(), B_magnitude[:, :, grid_size//2].cpu().numpy())
# plt.colorbar(quiver0, ax=axes[0], label='Magnitude of B magnetic field strength (T)')
# axes[0].set_title('XY plane at Z=0')
# axes[0].set_xlabel('X')
# axes[0].set_ylabel('Y')

# # XZ plane
# quiver1 = axes[1].quiver(X[:, grid_size//2, :].cpu().numpy(), Z[:, grid_size//2, :].cpu().numpy(), B[:, grid_size//2, :, 0].cpu().numpy(), B[:, grid_size//2, :, 2].cpu().numpy(), B_magnitude[:, grid_size//2, :].cpu().numpy())
# plt.colorbar(quiver1, ax=axes[1], label='Magnitude of B magnetic field strength (T)')
# axes[1].set_title('XZ plane at Y=0')
# axes[1].set_xlabel('X')
# axes[1].set_ylabel('Z')

# # YZ plane
# quiver2 = axes[2].quiver(Y[grid_size//2, :, :].cpu().numpy(), Z[grid_size//2, :, :].cpu().numpy(), B[grid_size//2, :, :, 1].cpu().numpy(), B[grid_size//2, :, :, 2].cpu().numpy(), B_magnitude[grid_size//2, :, :].cpu().numpy())
# plt.colorbar(quiver2, ax=axes[2], label='Magnitude of B magnetic field strength (T)')
# axes[2].set_title('YZ plane at X=0')
# axes[2].set_xlabel('Y')
# axes[2].set_ylabel('Z')

# plt.tight_layout()
# plt.show()