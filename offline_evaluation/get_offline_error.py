import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from tqdm import tqdm
import gc

from preprocessing_functions import *

print('imported packages')

config_dir = 'standard'
config_name = config_dir

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_clean/"
sp_data = load_data(month = 9, year = 1, data_path = data_path)

num_timesteps = 336
num_models = len([x for x in os.listdir('../coupling_folder/h5_models') if '_model_' in x])

sp_data_test_input = np.load('testing_data/test_input.npy')
sp_data_test_target = np.load('testing_data/test_target.npy')

inp_sub = np.loadtxt('../coupling_folder/norm_files/inp_sub.txt')[None,:]
inp_div = np.loadtxt('../coupling_folder/norm_files/inp_div.txt')[None,:]
out_scale = np.loadtxt('../coupling_folder/norm_files/out_scale.txt')[None,:]

reshaped_input = (reshape_input(sp_data_test_input).transpose() - inp_sub)/inp_div

dp = sp_data['gw'] * (sp_data["P0"] * sp_data["hyai"] + sp_data['hybi']*sp_data['NNPSBSP']).diff(dim = "ilev")
error_weights = dp[:336,:,:,:].values
error_weights_norm = np.swapaxes((error_weights/np.sum(error_weights)), 1, 2)

heating_true = sp_data_test_target[:,:30,:,:]
moistening_true = sp_data_test_target[:,30:,:,:]

print('created weights and reference data')

print(heating_true.shape)
print(moistening_true.shape)

def get_diffs(nn_model_path):
    nn_model = keras.models.load_model(nn_model_path, compile = False)
    nn_predict = nn_model.predict(reshaped_input)/out_scale
    nn_predict = np.concatenate((nn_predict[:,0:30], np.zeros((nn_predict.shape[0], 5)), nn_predict[:,30:]), axis = 1)
    nn_predict_unshaped = reverse_reshape(nn_predict, sp_data_test_target.shape)
    heating_predict = nn_predict_unshaped[:,0:30,:,:]
    moistening_predict = nn_predict_unshaped[:,30:60,:,:]
    diff_predict_unshaped = nn_predict_unshaped - sp_data_test_target
    heating_diff = diff_predict_unshaped[:,0:30,:,:]
    moistening_diff = diff_predict_unshaped[:,30:60,:,:]
    return heating_diff, moistening_diff

rmse = []
save_path = "offline_test_error/"

for i in tqdm(range(num_models)):
    keras.backend.clear_session()
    gc.collect()
    model_rank = i+1
    model_name = config_dir + "_model_" + str(model_rank).zfill(3)
    model_path = f'../coupling_folder/h5_models/{config_dir}_model_{str(model_rank).zfill(3)}.h5'
    heating_diff, moistening_diff = get_diffs(f'../coupling_folder/h5_models/{config_dir}_model_{str(model_rank).zfill(3)}.h5')
    heating_rmse = np.sum((heating_diff**2) * error_weights_norm)**.5
    del heating_diff
    moistening_rmse = np.sum((moistening_diff**2) * error_weights_norm)**.5
    del moistening_diff
    print(model_name)
    print('heating rmse: ')
    print(heating_rmse)
    print('moistening rmse: ')
    print(moistening_rmse)
    rmse.append([heating_rmse, moistening_rmse*1000])

rmse = np.array(rmse)

with open(save_path + "rmse.npy", 'wb') as f:
    np.save(f, np.float32(rmse))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(np.arange(rmse.shape[0]), rmse[:, 0], color = 'red', label='heating')
axs[0].set_title('heating RMSE')
axs[0].set_xlabel('model rank')
axs[0].set_ylabel('K/s')
axs[0].set_ylim(1e-5, 5e-5)
axs[0].grid(True, which="both", ls="--")

axs[1].plot(np.arange(rmse.shape[0]), rmse[:, 1], color = 'blue', label='moistening')
axs[1].set_title('moistening RMSE')
axs[1].set_xlabel('model rank')
axs[1].set_ylabel('g/kg')
axs[1].set_ylim(1e-5, 5e-5)
axs[1].grid(True, which="both", ls="--")

fig.suptitle('offline test error by model rank, {} configuration'.format(config_name))

plt.tight_layout()
fig.savefig('offline_test_error/offline_test_error.png')

print("finished")
