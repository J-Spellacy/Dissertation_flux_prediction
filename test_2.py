import torch
import pandas as pd
import numpy as np

# Example DataFrame
df_val = pd.DataFrame({
    'Feature1': np.random.randn(1000),  # 100 random features
    'Feature2': np.random.randn(1000)
}, index=np.linspace(0, 999, 1000))  # Numerical index from 0 to 99

# Example Tensor
tensor = torch.tensor([10.3, 20, 30, 40, 50, 60, 70, 80, 90, 100, 45, 60, 800, 8000, 6, 3, 596, 78, 49, 37, 649507, 5847, 758, 6873, 33, 3,3,46, 7, 8,99]).float()
tensor = torch.squeeze(tensor)  # Ensure tensor is squeezed

# Convert tensor to numpy array or list for iteration
tensor_values = tensor.numpy()

# DataFrame to store selected rows
selected_rows = pd.DataFrame()

# Iterate over each value in the tensor
for selector_val in tensor_values:
    # Find the closest index in DataFrame
    closest_index = df_val.index[np.argmin(np.abs(df_val.index - selector_val))]
    # Select the row and add to selected_rows DataFrame
    selected_row = df_val.loc[closest_index]
    selected_rows = selected_rows._append(selected_row)

# print("Selected Rows based on closest index values:")
# print(selected_rows)
df_val = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_megacut_si_units.csv', index_col=0)

num_points = 1000
range_x = (-24, 24)
range_y = (-24, 24)
range_z = (-24, 24)
x_iono = torch.rand(num_points) * (range_x[1] - range_x[0]) + range_x[0]
y_iono = torch.rand(num_points) * (range_y[1] - range_y[0]) + range_y[0]
z_iono = torch.rand(num_points) * (range_z[1] - range_z[0]) + range_z[0]
r_vec = torch.stack((x_iono, y_iono, z_iono), dim=-1)
print(r_vec.shape)



