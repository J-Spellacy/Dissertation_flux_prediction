import torch
import torch.nn as nn
from typing import Callable
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Support_funcs_one_model import make_forward_fn, LinearNN, model_creation, sig_figs, scale_sample, scale_sample_t, unscale, unscale_not_t, EarlyStopping, convNN
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
from math import floor, log10

omni_data = pd.read_csv('C:/Users/User/Desktop/PINN_torch/val_PINN_data_OMNI_cutdown_si_units.csv', index_col=0)
#print(omni_data.max())      
R_E_km = 6371
R_E_m = 6371000.0

velocity_factor = 1 / R_E_km
df=omni_data
def Reps_to_mps(x):
    return x * R_E_m

def convert_density(n_per_cc):
    """
    Convert density from number per cubic centimeter to kilograms per cubic meter.
    Assuming the density is of protons.

    :param n_per_cc: Density in number per cubic centimeter
    :return: Density in kilograms per cubic meter
    """
    mass_of_proton = 1.67e-27  # kg
    return n_per_cc * mass_of_proton * 1e6

def convert_pressure(nPa):
    """
    Convert pressure from nanoPascals to Pascals.

    :param nPa: Pressure in nanoPascals
    :return: Pressure in Pascals
    """
    return nPa * 1e-9

def convert_magnetic_field_strength(nT):
    """
    Convert magnetic field strength from nanoTesla to Tesla.

    :param nT: Magnetic field strength in nanoTesla
    :return: Magnetic field strength in Tesla
    """
    return nT * 1e-9

def convert_mins_sec(time):
    return time * 60

def convert_Re_m(distance):
    return distance * R_E_m


# df[['BX__GSE_T', 'BY__GSE_T', 'BZ__GSE_T']] = df[['X_(BSN)__GSE_Re',  'Y_(BSN)__GSE_Re',  'Z_(BSN)__GSE_Re']].apply(convert_magnetic_field_strength)
# df.drop(['GSE_X_nT', 'GSE_Y_nT', 'GSE_Z_nT'], axis=1, inplace=True)

# df[['B_T']] = df[['B__nT']].apply(convert_magnetic_field_strength)
# df.drop(['B__nT'], axis=1, inplace=True)

# df[['GSE_X',  'GSE_Y',  'GSE_Z']] = df[['GSE_X',  'GSE_Y',  'GSE_Z']].apply(convert_Re_m)
# # df.drop(['X_(BSN)__GSE_Re',  'Y_(BSN)__GSE_Re',  'Z_(BSN)__GSE_Re'], axis=1, inplace=True)

# new_order = ['time', 'GSE_X',  'GSE_Y',  'GSE_Z', 'B_T']

# # Reindex the DataFrame according to the new order
# df = df[new_order]
df = df[df['X_(S/C)__GSE_m'] < 6371000.0 * 24]
# df.to_csv('C:/Users/User/Desktop/PINN_torch/CRRES_GSE_si_units.csv')
df['time'] = df['time_s']
df.set_index('time_s', inplace=True)
df['group'] = df.index // 1000

# Now group by this new column and calculate the mean for each group
resampled_df = df.groupby('group').mean()
resampled_df.set_index('time', inplace=True)
print(df)
print(resampled_df)

resampled_df.to_csv('C:/Users/User/Desktop/PINN_torch/val_omni_sw_lessthan_24.csv', index=True)