import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_o3/"
norm_path = "../coupling_folder/norm_files/"

spData_val_input = combine_arrays(make_nn_input(load_data(month = 11, year = 1, data_path = data_path)), \
                                  make_nn_input(load_data(month = 12, year = 1, data_path = data_path)))
print(spData_val_input.shape)

spData_val_input = reshape_input(spData_val_input)
print(spData_val_input.shape)

inp_sub = np.loadtxt(norm_path + "inp_sub.txt")[:, np.newaxis]
inp_div = np.loadtxt(norm_path + "inp_div.txt")[:, np.newaxis]
print(inp_sub.shape)
print(inp_div.shape)

X_val = normalize_input_val(X_val = spData_val_input, inp_sub = inp_sub, inp_div = inp_div, save_files = True)

print("finished")

