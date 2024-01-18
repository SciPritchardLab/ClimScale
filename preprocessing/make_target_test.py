import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_o3/"
save_path = "../offlinetesteval/testing_data/"

spData_test_target = combine_arrays(make_nn_target(load_data(month = 9, year = 1, data_path = data_path), subsample = False))
print(spData_test_target.shape)

test_target = spData_test_target[:,0:336,:,:]
print(test_target.shape)

with open(save_path + "test_target.npy", 'wb') as f:
    np.save(f, np.float32(test_target))

print("finished")