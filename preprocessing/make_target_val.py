import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_o3/"

spData_val_target = combine_arrays(make_nn_target(load_data(month = 11, year = 1, data_path = data_path)), \
                                   make_nn_target(load_data(month = 12, year = 1, data_path = data_path)))
print(spData_val_target.shape)

spData_val_target = reshape_target(spData_val_target)
print(spData_val_target.shape)

y_val = normalize_target_val(y_val_original = spData_val_target, save_files = True)

print("finished")