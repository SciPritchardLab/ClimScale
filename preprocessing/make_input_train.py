import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_o3/"

spData_train_input = combine_arrays(make_nn_input(load_data(month = 2, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 3, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 4, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 5, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 6, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 7, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 8, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 9, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 10, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 11, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 12, year = 0, data_path = data_path)), \
                                    make_nn_input(load_data(month = 1, year = 1, data_path = data_path)))
print(spData_train_input.shape)

spData_train_input = reshape_input(spData_train_input)
print(spData_train_input.shape)

X_train, inp_sub, inp_div = normalize_input_train(X_train = spData_train_input, save_files = True)

print("finished")
