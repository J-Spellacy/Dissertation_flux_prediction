import pandas as pd
import torch
oni = pd.read_csv('C:/Users/User/Desktop/PINN_torch/cutdown_CRRES.csv')
df_val_flux = pd.read_csv('C:/Users/User/Desktop/PINN_torch/cutdown_CRRES.csv')
print(df_val_flux)

# common_index = oni.index.intersection(df_val_flux.index)
# df2_trimmed = oni.loc[common_index]
# df2_trimmed[['TEMPERATURE_K',   '5-M_AE_nT', 'SYM/H_INDEX_nT',  'ASY/H_INDEX_nT',  'PROTONS>10_MEV_1/(SQcm-ster-s)',  'PROTONS>30_MEV_1/(SQcm-ster-s)',  'PROTONS>60_MEV_1/(SQcm-ster-s)..']] = df_val_flux[['TEMPERATURE_K',   '5-M_AE_nT', 'SYM/H_INDEX_nT',  'ASY/H_INDEX_nT',  'PROTONS>10_MEV_1/(SQcm-ster-s)',  'PROTONS>30_MEV_1/(SQcm-ster-s)',  'PROTONS>60_MEV_1/(SQcm-ster-s)..']]

# df2_trimmed.dropna(axis=1, inplace=True)
# print(df2_trimmed.max())

#df2_trimmed.to_csv('C:/Users/User/Desktop/PINN_torch/cutdown_CRRES.csv',index=False)

def concatenate_previous_time_steps(input_tensor, max_time_step):
    """
    Concatenate each time step with the previous one up to a maximum time step.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, num_features).
        max_time_step (int): Maximum number of time steps to concatenate with the previous one.

    Returns:
        torch.Tensor: Concatenated tensor of shape (batch_size - max_time_step, num_features * (max_time_step + 1)).
    """
    batch_size, num_features = input_tensor.shape

    # Initialize a list to store concatenated tensors
    concatenated_tensors = []

    # Concatenate each time step with the previous one up to max_time_step
    for t in range(max_time_step, batch_size):
        previous_time_steps = input_tensor[t - max_time_step:t + 1]  # Get previous time steps
        concatenated_tensors.append(previous_time_steps.flatten())  # Flatten and append

    # Stack the concatenated tensors along a new batch dimension
    return torch.stack(concatenated_tensors)

# input_tensor = torch.randn(20, 5)  # Example input tensor of shape (batch_size, num_features)
# max_time_step = 3  # Example maximum number of time steps
# concatenated_tensor = concatenate_previous_time_steps(input_tensor, max_time_step)
# #print(concatenated_tensor.shape)  # Shape of the concatenated tensor


