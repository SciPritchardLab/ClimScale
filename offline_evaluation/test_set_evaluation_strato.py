import xarray as xr
import numpy as np
import os
import re
import math
import scipy.integrate as integrate
from tensorflow import keras

from tqdm import tqdm

proj_name = "specific"
subsampling = 3

def sample_indices(size, spacing, fixed = True):
    numIndices = np.round(size/spacing)
    if fixed:
        indices = np.array([int(x) for x in np.round(np.linspace(1,size,int(numIndices)))])-1
    else:
        indices = list(range(size))
        np.random.shuffle(indices)
        indices = indices[0:int(numIndices)]
    return indices

data_path = "testing_data/"
norm_path = "norm_files/"
model_path = "../coupling_folder/h5_models/"

num_models = 330

inpsub = np.loadtxt(norm_path + "inp_sub.txt")
inpdiv = np.loadtxt(norm_path + "inp_div.txt")

heatScale = 1004
moistScale = 2.5e6
outscale = np.concatenate((np.repeat(heatScale, 30), np.repeat(moistScale, 30)))

with open(data_path + 'test_input.npy', 'rb') as f:
    test_input = np.load(f)[:,sample_indices(336, subsampling),:,:]
    
with open(data_path + 'test_target.npy', 'rb') as f:
    test_target = np.load(f)[:,sample_indices(336, subsampling),:,:]
    
assert test_input.shape[1]==test_target.shape[1]

timesteps = test_input.shape[1]
    
nn_input = (test_input-inpsub[:,np.newaxis,np.newaxis,np.newaxis])/inpdiv[:,np.newaxis,np.newaxis,np.newaxis]

assert test_input.shape[1] == test_target.shape[1]

# ERROR WEIGHTS

error_weight_path = "/ocean/projects/atm200007p/jlin96/new_final_families/"

with open(error_weight_path + 'offline_error_weights_strato.npy', 'rb') as f:
    error_weights = np.load(f)

def get_prediction(proj_name, model_rank):
    f_load = model_path + '%s_model_%03d.h5'%(proj_name, model_rank)
    model = keras.models.load_model(f_load, compile=False)
    unrolled = np.reshape(nn_input, (64, -1)).transpose()
    prediction = model.predict(unrolled).transpose()/outscale[:,np.newaxis]
    prediction = np.reshape(prediction, (60, timesteps, 64, 128))
    return prediction

def squared_error(prediction, target):
    se = (prediction-target)**2
    se_T = se[0:30,:,:,:]
    se_Q = se[30:60,:,:,:]
    return se_T, se_Q

def weight_error(se):
    return se*error_weights

def root_error(wse):
    return np.sum(wse)**.5

def get_rmse(prediction, target):
    se_T, se_Q = squared_error(prediction, target)
    rmse_T = root_error(weight_error(se_T))
    rmse_Q = root_error(weight_error(se_Q))
    return rmse_T, rmse_Q

rmse = []
for i in tqdm(range(num_models)):
    model_rank = i+1
    rmse_T, rmse_Q = get_rmse(get_prediction(proj_name, model_rank), test_target)
    rmse.append([rmse_T, rmse_Q])
rmse = np.array(rmse)

save_path = "offline_errors/"
with open(save_path + "rmse_strato.npy", 'wb') as f:
    np.save(f, np.float32(rmse))

# open a file for writing
with open('confirmation_strato.txt', 'w') as f:
    # write some text to the file
    f.write('RMSE strato finished calculating!\n')