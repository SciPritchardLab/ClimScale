import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import os
import gc
from preprocessing_functions import *
print('imported packages')

config_dir = 'mae'
config_name = config_dir

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_clean/"
data_path_cold = "/ocean/projects/atm200007p/jlin96/longSPrun_clean_cold/"
data_path_warm = "/ocean/projects/atm200007p/jlin96/longSPrun_clean_warm/"
save_path = "offline_test_error/"

sp_data = load_data(month = 9, year = 1, data_path = data_path)
sp_data_cold = load_data(month = 7, year = 0, data_path = data_path_cold)
sp_data_warm = load_data(month = 7, year = 0, data_path = data_path_warm)

dp = sp_data['gw'] * (sp_data["P0"] * sp_data["hyai"] + sp_data['hybi']*sp_data['NNPSBSP']).diff(dim = "ilev")
dp_cold = sp_data_cold['gw'] * (sp_data_cold["P0"] * sp_data_cold["hyai"] + sp_data_cold['hybi']*sp_data_cold['NNPSBSP']).diff(dim = "ilev")
dp_warm = sp_data_warm['gw'] * (sp_data_warm["P0"] * sp_data_warm["hyai"] + sp_data_warm['hybi']*sp_data_warm['NNPSBSP']).diff(dim = "ilev")

num_timesteps = 336
batch_size = 229376
num_models = len([x for x in os.listdir('../coupling_folder/h5_models') if '_model_' in x])

inp_sub = np.loadtxt('../coupling_folder/norm_files/inp_sub.txt')[None,:]
inp_div = np.loadtxt('../coupling_folder/norm_files/inp_div.txt')[None,:]
out_scale = np.loadtxt('../coupling_folder/norm_files/out_scale.txt')[None,:]

def get_diffs(nn_model_path, nn_input, nn_test_target):
    nn_model = keras.models.load_model(nn_model_path, compile = False)
    tf_fn = tf.function(nn_model, jit_compile=False)
    nn_predict = np.concatenate([tf_fn(nn_input[idx*batch_size:(idx+1)*batch_size,:]) for idx in range(int(nn_input.shape[0]/batch_size))], axis = 0)/out_scale
    nn_predict = np.concatenate((nn_predict[:,0:30], np.zeros((nn_predict.shape[0], 5)), nn_predict[:,30:]), axis = 1)
    nn_predict_unshaped = reverse_reshape(nn_predict, nn_test_target.shape)
    heating_predict = nn_predict_unshaped[:,0:30,:,:]
    moistening_predict = nn_predict_unshaped[:,30:60,:,:]
    diff_predict_unshaped = nn_predict_unshaped - nn_test_target
    heating_diff = diff_predict_unshaped[:,0:30,:,:]
    moistening_diff = diff_predict_unshaped[:,30:60,:,:]
    keras.backend.clear_session()
    gc.collect()
    return heating_diff, moistening_diff

def run_inference(nn_input, nn_test_target, error_weights_norm):
    rmse = []
    for i in tqdm(range(num_models)):
        model_rank = i+1
        model_name = config_dir + "_model_" + str(model_rank).zfill(3)
        nn_model_path = f'../coupling_folder/h5_models/{config_dir}_model_{str(model_rank).zfill(3)}.h5'
        heating_diff, moistening_diff = get_diffs(nn_model_path, nn_input, nn_test_target)
        assert heating_diff.shape == error_weights_norm.shape, "Shape mismatch between heating_diff and error_weights_norm"
        assert moistening_diff.shape == error_weights_norm.shape, "Shape mismatch between moistening_diff and error_weights_norm"
        heating_rmse = np.sum((heating_diff**2) * error_weights_norm)**.5
        del heating_diff
        moistening_rmse = np.sum((moistening_diff**2) * error_weights_norm)**.5
        del moistening_diff
        # print(model_name)
        # print('heating rmse: ')
        # print(heating_rmse)
        # print('moistening rmse: ')
        # print(moistening_rmse)
        rmse.append([heating_rmse, moistening_rmse*1000])
    return rmse

print('loading data for inference')

sp_data_test_input = np.load('testing_data/test_input.npy')
sp_data_test_target = np.load('testing_data/test_target.npy')
heating_true = sp_data_test_target[:,:30,:,:]
moistening_true = sp_data_test_target[:,30:,:,:]
reshaped_input = (reshape_input(sp_data_test_input).transpose() - inp_sub)/inp_div
error_weights = dp[:num_timesteps,:,:,:].values
error_weights_norm = np.swapaxes((error_weights/np.sum(error_weights)), 1, 2)
print('created weights and reference data')
print('reshaped_input.shape')
print(reshaped_input.shape)
print('heating_true.shape')
print(heating_true.shape)
print('moistening_true.shape')
print(moistening_true.shape)
print('error_weights_norm.shape')
print(error_weights_norm.shape)

print('starting inference')
rmse = run_inference(reshaped_input, sp_data_test_target, error_weights_norm)
print('finished inference')

with open(os.path.join(save_path, "rmse.npy"), 'wb') as f:
    np.save(f, np.float32(rmse))

del sp_data_test_input
del sp_data_test_target
del reshaped_input
del error_weights
del error_weights_norm
del rmse
gc.collect()

sp_data_test_input_2x = np.load('testing_data_2x/test_input_2x.npy')
sp_data_test_target_2x = np.load('testing_data_2x/test_target_2x.npy')
heating_true_2x = sp_data_test_target_2x[:,:30,:,:]
moistening_true_2x = sp_data_test_target_2x[:,30:,:,:]
reshaped_input_2x = (reshape_input(sp_data_test_input_2x).transpose() - inp_sub)/inp_div
error_weights_2x = dp[:num_timesteps*2,:,:,:].values
error_weights_norm_2x = np.swapaxes((error_weights_2x/np.sum(error_weights_2x)), 1, 2)
print('created weights and reference data (2x)')
print('reshaped_input_2x.shape')
print(reshaped_input_2x.shape)
print('heating_true_2x.shape')
print(heating_true_2x.shape)
print('moistening_true_2x.shape')
print(moistening_true_2x.shape)
print('error_weights_norm_2x.shape')
print(error_weights_norm_2x.shape)

print('starting inference')
rmse_2x = run_inference(reshaped_input_2x, sp_data_test_target_2x, error_weights_norm_2x)
print('finished inference')

with open(os.path.join(save_path, "rmse_2x.npy"), 'wb') as f:
    np.save(f, np.float32(rmse_2x))

del sp_data_test_input_2x
del sp_data_test_target_2x
del reshaped_input_2x
del error_weights_2x
del error_weights_norm_2x
del rmse_2x
gc.collect()

sp_data_test_input_4x = np.load('testing_data_4x/test_input_4x.npy')
sp_data_test_target_4x = np.load('testing_data_4x/test_target_4x.npy')
heating_true_4x = sp_data_test_target_4x[:,:30,:,:]
moistening_true_4x = sp_data_test_target_4x[:,30:,:,:]
reshaped_input_4x = (reshape_input(sp_data_test_input_4x).transpose() - inp_sub)/inp_div
error_weights_4x = dp[:num_timesteps*4,:,:,:].values
error_weights_norm_4x = np.swapaxes((error_weights_4x/np.sum(error_weights_4x)), 1, 2)
print('created weights and reference data (4x)')
print('reshaped_input_4x.shape')
print(reshaped_input_4x.shape)
print('heating_true_4x.shape')
print(heating_true_4x.shape)
print('moistening_true_4x.shape')
print(moistening_true_4x.shape)
print('error_weights_norm_4x.shape')
print(error_weights_norm_4x.shape)
rmse_4x = []

print('starting inference')
rmse_4x = run_inference(reshaped_input_4x, sp_data_test_target_4x, error_weights_norm_4x)
print('finished inference')

with open(os.path.join(save_path, "rmse_4x.npy"), 'wb') as f:
    np.save(f, np.float32(rmse_4x))

del sp_data_test_input_4x
del sp_data_test_target_4x
del reshaped_input_4x
del error_weights_4x
del error_weights_norm_4x
del rmse_4x
gc.collect()

sp_data_test_input_multi = np.load('testing_data_multi/test_input_multi.npy')
sp_data_test_target_multi = np.load('testing_data_multi/test_target_multi.npy')
heating_true_multi = sp_data_test_target_multi[:,:30,:,:]
moistening_true_multi = sp_data_test_target_multi[:,30:,:,:]
reshaped_input_multi = (reshape_input(sp_data_test_input_multi).transpose() - inp_sub)/inp_div
error_weights = dp[:int(num_timesteps/3),:,:,:].values
error_weights_cold = dp_cold[:int(num_timesteps/3),:,:,:].values
error_weights_warm = dp_warm[:int(num_timesteps/3),:,:,:].values
error_weights_multi = np.concatenate([error_weights_cold, error_weights, error_weights_warm], axis = 0)
error_weights_norm_multi = np.swapaxes((error_weights_multi/np.sum(error_weights_multi)), 1, 2)
print('created weights and reference data (multi)')
print('reshaped_input_multi.shape')
print(reshaped_input_multi.shape)
print('heating_true_multi.shape')
print(heating_true_multi.shape)
print('moistening_true_multi.shape')
print(moistening_true_multi.shape)
print('error_weights_norm_multi.shape')
print(error_weights_norm_multi.shape)
rmse_multi = []

print('starting inference')
rmse_multi = run_inference(reshaped_input_multi, sp_data_test_target_multi, error_weights_norm_multi)
print('finished inference')

with open(os.path.join(save_path, "rmse_multi.npy"), 'wb') as f:
    np.save(f, np.float32(rmse_multi))

del sp_data_test_input_multi
del sp_data_test_target_multi
del reshaped_input_multi
del error_weights
del error_weights_cold
del error_weights_warm
del error_weights_multi
del error_weights_norm_multi
del rmse_multi
gc.collect()

print("All processing completed successfully.")
