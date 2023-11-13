import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_o3/"
save_path = "../offlinetesteval/testing_data/"

spData_test_input = combine_arrays(make_nn_input(load_data(month = 9, year = 1, data_path = data_path), family = "relative", subsample = False))
print(spData_test_input.shape)

test_input = spData_test_input[:,0:336,:,:]
print(test_input.shape)

with open(save_path + "test_input.npy", 'wb') as f:
    np.save(f, np.float32(test_input))

print("finished")
