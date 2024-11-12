import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
import gc

from preprocessing_functions import *

print('imported packages')

config_name = 'nodropout'

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_clean/"
sp_data = load_data(month = 9, year = 1, data_path = data_path)

print('loaded SP data')

num_timesteps = 336

sp_data_train_input = np.load('../training/training_data/train_input.npy')
sp_data_train_target = np.load('../training/training_data/train_target.npy')

sp_data_train_target = np.concatenate((sp_data_train_target[:,:30], \
                                        np.zeros((sp_data_train_target.shape[0], 5)), \
                                        sp_data_train_target[:,30:]), axis = 1)

inp_sub = np.loadtxt('../coupling_folder/norm_files/inp_sub.txt')[None,:]
inp_div = np.loadtxt('../coupling_folder/norm_files/inp_div.txt')[None,:]
out_scale = np.loadtxt('../coupling_folder/norm_files/out_scale.txt')[None,:]

out_scale = np.concatenate((out_scale[:,:30], \
                            np.ones((out_scale.shape[0], 5)), \
                            out_scale[:,30:]), axis = 1)

print('loaded normalization files')

sp_data_train_input = (sp_data_train_input * inp_div) + inp_sub
sp_data_train_target = sp_data_train_target/out_scale

print('unnormalized training data')

sp_data_test_input = np.load('testing_data/test_input.npy')
sp_data_test_target = np.load('testing_data/test_target.npy')

dp = sp_data['gw'] * (sp_data["P0"] * sp_data["hyai"] + sp_data['hybi']*sp_data['NNPSBSP']).diff(dim = "ilev")
error_weights = dp[:336,:,:,:].values
error_weights_norm = np.swapaxes((error_weights/np.sum(error_weights)), 1, 2)

print('created error weights')

heating_true = sp_data_test_target[:,:30,:,:]
moistening_true = sp_data_test_target[:,30:,:,:]

def get_heating_rmse(heating_pred):
    return np.sum(((heating_pred - heating_true)**2)*error_weights_norm)**.5

def get_moistening_rmse(moistening_pred):
    return np.sum(((moistening_pred - moistening_true)**2)*error_weights_norm)**.5

average_heating_rmse = np.average(((np.mean(sp_data_train_target, axis = 0)[None,0:30,None,None] - sp_data_test_target[:,0:30,:,:])**2), weights = error_weights_norm)**.5
average_moistening_rmse = 1000*np.average(((np.mean(sp_data_train_target, axis = 0)[None,30:60,None,None] - sp_data_test_target[:,30:60,:,:])**2), weights = error_weights_norm)**.5

print('calculated average')

bias = np.ones((sp_data_train_input.shape[0], 1))
train_input_with_bias = np.concatenate((sp_data_train_input, bias), axis=1)
train_input_with_bias_transpose = np.transpose(train_input_with_bias)
X_transpose_X = train_input_with_bias_transpose@train_input_with_bias
X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
X_transpose_y = train_input_with_bias_transpose@sp_data_train_target
mlr_weights = X_transpose_X_inverse@X_transpose_y

reshaped_test_input = reshape_input(sp_data_test_input).transpose()
bias = np.ones((reshaped_test_input.shape[0], 1))
test_input_with_bias = np.concatenate((reshaped_test_input, bias), axis=1)

mlr_preds = reverse_reshape(test_input_with_bias@mlr_weights, sp_data_test_target.shape)
mlr_preds_heating = mlr_preds[:,:30,:,:]
mlr_preds_moistening = mlr_preds[:,30:,:,:]

mlr_heating_rmse = get_heating_rmse(mlr_preds_heating)
mlr_moistening_rmse = get_moistening_rmse(mlr_preds_moistening)*1000

print('calculated mlr')

unablated_test_input = np.load('/ocean/projects/atm200007p/jlin96/nnspreadtesting_good/unablated/offline_evaluation/testing_data/test_input.npy')

def reverse_reshape(reshaped_arr, original_shape):
    '''
    reshaped_arr should be num_samples x features for this function to work properly
    '''
    arr = reshaped_arr.transpose().reshape(60, original_shape[0], original_shape[2], original_shape[3], order='F')
    ans = arr.transpose(1,0,2,3)
    print(ans.shape)
    return ans

def reshape_unablated_input(nn_input):
    nn_input = nn_input.transpose(1,0,2,3)
    ans = nn_input.ravel(order = 'F').reshape(60,-1,order = 'F')
    print(ans.shape)
    return ans

martingale_preds = reverse_reshape(reshape_unablated_input(unablated_test_input[:,60:120,:,:]).transpose(), sp_data_test_target.shape)
martingale_preds_heating = martingale_preds[:,:30,:,:]
martingale_preds_moistening = martingale_preds[:,30:,:,:]

martingale_heating_rmse = get_heating_rmse(martingale_preds_heating)
martingale_moistening_rmse = get_moistening_rmse(martingale_preds_moistening)*1000

print('calculated martingale')

rmse = np.load('offline_test_error/rmse.npy')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

line_heating, = axs[0].plot(np.arange(rmse.shape[0]), rmse[:, 0], color = 'red', label='heating')
axs[0].set_title('heating RMSE')
axs[0].set_xlabel('model rank')
axs[0].set_ylabel('K/s')
axs[0].set_ylim(1e-5, 5e-5)
axs[0].grid(True, which="both", ls="--")
axs[0].axhline(y = average_heating_rmse, color='black', linestyle='--', label='average heating RMSE')
axs[0].axhline(y = mlr_heating_rmse, color='green', linestyle='--', label='MLR heating RMSE')
axs[0].axhline(y = martingale_heating_rmse, color='purple', linestyle='--', label='martingale heating RMSE')

line_moistening, = axs[1].plot(np.arange(rmse.shape[0]), rmse[:, 1], color = 'blue', label='moistening')
axs[1].set_title('moistening RMSE')
axs[1].set_xlabel('model rank')
axs[1].set_ylabel('g/kg')
axs[1].set_ylim(1e-5, 5e-5)
axs[1].grid(True, which="both", ls="--")
line_average = axs[1].axhline(y = average_moistening_rmse, color='black', linestyle='--', label='average moistening RMSE')
line_mlr = axs[1].axhline(y = mlr_moistening_rmse, color='green', linestyle='--', label='MLR moistening RMSE')
line_martingale = axs[1].axhline(y = martingale_moistening_rmse, color='purple', linestyle='--', label='martingale moistening RMSE')
axs[1].legend([line_heating, line_moistening, line_average, line_mlr, line_martingale], ['heating', 'moistening', 'average', 'MLR', 'martingale'])

fig.suptitle('offline test error by model rank, {} configuration'.format(config_name))

plt.tight_layout()
fig.savefig('offline_test_error/offline_test_error.png')
