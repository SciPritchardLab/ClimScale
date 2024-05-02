import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable

import os
import re
import math
from tqdm import tqdm
import sys

import pickle

config_subdir = sys.argv[1]
config_name = config_subdir

sp_path = "/ocean/projects/atm200007p/jlin96/longSPrun_clean/trim_dir/trimmed/"
num_runs = len([x for x in os.listdir("../coupled_results/") if '_model_' in x])

def ls(data_path = ""):
    return os.popen(" ".join(["ls", data_path])).read().splitlines()

def load_data(data_path):
    datasets = ls(data_path)
    datasets = [data_path + x for x in datasets if "h1.000" in x]
    return xr.open_mfdataset(datasets, \
                             compat = 'override', \
                             join = 'override', \
                             coords = "minimal")

sp_data = load_data(sp_path)
sp_data_year = sp_data.isel(time = slice(0, 365))
sp_data_lag = sp_data.isel(time = slice(31, 365+31))

month_length = np.array([len(sp_data_year.groupby("time.month")[x]["time"]) for x in np.array(range(12))+1])
month_sum = np.insert(np.cumsum(month_length), 0, 0)

def get_monthly_mean(data):
    monthly_mean = np.concatenate((np.mean(data[month_sum[0]:month_sum[1],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[1]:month_sum[2],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[2]:month_sum[3],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[3]:month_sum[4],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[4]:month_sum[5],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[5]:month_sum[6],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[6]:month_sum[7],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[7]:month_sum[8],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[8]:month_sum[9],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[9]:month_sum[10],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[10]:month_sum[11],:,:,:], axis = 0)[None,:,:,:],
                                   np.mean(data[month_sum[11]:month_sum[12],:,:,:], axis = 0)[None,:,:,:]), axis = 0)
    return monthly_mean

sp_temp = get_monthly_mean(sp_data_year["NNTBSP"].values)
sp_hum = get_monthly_mean(sp_data_year["NNQBSP"].values)
sp_temp_lag = get_monthly_mean(sp_data_lag["NNTBSP"].values)
sp_hum_lag = get_monthly_mean(sp_data_lag["NNQBSP"].values)

dp = sp_data_year['gw'] * (sp_data_year["P0"] * sp_data_year["hyai"] + sp_data_year['hybi']*sp_data_year['NNPSBSP']).diff(dim = "ilev")
error_weights = dp.groupby('time.month').mean('time')
error_weights = error_weights.values/(error_weights.sum(dim = ['lat', 'ilev', 'lon']).values[:,None,None,None])
error_weights = np.transpose(error_weights, (0,2,1,3))

lagged_temp = np.sqrt(np.sum(((sp_temp_lag - sp_temp)**2)*error_weights, axis = (1,2,3)))
lagged_hum = np.sqrt(np.sum(((sp_hum_lag - sp_hum)**2)*error_weights, axis = (1,2,3)))

offline_test_error = np.load('../offline_evaluation/offline_test_error/rmse.npy')

def peek(config_subdir, number, var):
    folder = "../coupled_results/"
    path = folder + config_subdir + "_model_%03d"%number + "/"
    h1List = os.popen(" ".join(["ls", path + "*.h1.0000*"])).read().splitlines()
    dataset = xr.open_mfdataset(h1List)
    if var == "NNTBSP":
        arr = dataset["NNTBSP"]
    elif var == "NNQBSP":
        arr = dataset["NNQBSP"]
    return arr

def get_diff(config_subdir, number, var):
    arr = peek(config_subdir, number, var)
    end_length = np.arange(1,13)[arr.shape[0] >= month_sum[1:]][-1] #such that only complete months are added
    arr = np.array(arr.groupby("time.month").mean("time"))
    if var == "NNTBSP":
        sp_vals = sp_temp[:end_length,:,:,:]
    elif var == "NNQBSP":
        sp_vals = sp_hum[:end_length,:,:,:]
    se = (sp_vals-arr[:end_length,:,:,:])**2
    wse = error_weights[:end_length,:,:,:]*se
    return np.sum(wse, axis = (1,2,3))**.5

def monthcheck(config_subdir, number, var):
    arr = peek(config_subdir, number, var)
    if arr.shape[0] < 31: # check to see first month integrated
        return False
    return True

def load_run(config_subdir, var, num_runs = num_runs):
    prognostic_runs = {}
    for i in tqdm(range(num_runs)):
        modelrank = i + 1
        if monthcheck(config_subdir, modelrank, var):
            prognostic_runs[modelrank] = get_diff(config_subdir, modelrank, var)
        else:
            prognostic_runs[modelrank] = np.array([])
    return prognostic_runs

def plot_diff(axnum, config_name, config_diffs, offline_error, var, logy = True):
    colors = ["black"]
    legend_names = ["internal variability proxy"]

    if var == "NNTBSP":
        offline_lower_lim = 1e-5
        offline_upper_lim = 4e-5
        offline_lower_lim = 1.8e-5
        offline_upper_lim = 2.5e-5
        cmap = plt.get_cmap('plasma')
        sm = ScalarMappable(cmap = cmap)
        sm.set_array(np.linspace(offline_lower_lim, offline_upper_lim, 100))
        for i in range(len(config_diffs)):
            assert offline_error[i,0] > offline_lower_lim
            color_index = (offline_error[i,0] - offline_lower_lim)/(offline_upper_lim - offline_lower_lim)
            if color_index > 1:
                color_index = 1
            axnum.plot(config_diffs[i+1], color = cmap(color_index), linewidth = .25)
        var_label = "Temperature"
        plt.colorbar(sm, ax = axnum, pad = .04)
        axnum.plot(lagged_temp, color = "black", linewidth = .8)
        axnum.set_ylim((.8e0, 2e2))
    if var == "NNQBSP":
        offline_lower_lim = 1e-5
        offline_upper_lim = 6e-5
        offline_lower_lim = 1.8e-5
        offline_upper_lim = 3e-5
        cmap = plt.get_cmap('winter')
        sm = ScalarMappable(cmap = cmap)
        sm.set_array(np.linspace(0, offline_upper_lim, 100))
        for i in range(len(config_diffs)):
            assert 1000*offline_error[i,1] > offline_lower_lim
            color_index = (1000*offline_error[i,1] - offline_lower_lim)/(offline_upper_lim - offline_lower_lim)
            if color_index > 1:
                color_index = 1
            axnum.plot(config_diffs[i+1]*1000, color = cmap(color_index), linewidth = .25)
        var_label = "Humidity"
        plt.colorbar(sm, ax = axnum, pad = .04)
        axnum.plot(lagged_hum*1000, color = "black", linewidth = .8)
        axnum.set_ylim((3e-1, 3e1))
    
    patches = [mpatches.Patch(facecolor = x) for x in colors]

    axnum.legend(handles = patches, \
            labels = legend_names, \
            loc = "upper right", \
            borderaxespad = 0.1, \
            fontsize = 12)
            
    axnum.set_title(var_label + " Root Mean Squared Error (RMSE)", fontsize = 16)
    if logy:
        axnum.set_yscale("log")
    
    axnum.set_xlabel("month", fontsize = 14)
    if var == "NNTBSP":
        axnum.set_ylabel("K", fontsize = 14)
    if var == "NNQBSP":
        axnum.set_ylabel("g/kg", fontsize = 14)

    axnum.grid(True, which='major', linestyle='-', linewidth=0.5)
    axnum.grid(True, which='minor', linestyle=':', linewidth=0.25)

diff_T = load_run(config_subdir, "NNTBSP")
diff_Q = load_run(config_subdir, "NNQBSP")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_diff(ax1, config_name, diff_T, offline_test_error, 'NNTBSP')
plot_diff(ax2, config_name, diff_Q, offline_test_error, 'NNQBSP')

fig.suptitle(config_name + ' configuration monthly prognostic error', fontsize = 16)

# plt.subplots_adjust(wspace=0.01)

plt.tight_layout()
fig.savefig('online_diffs.png', dpi = 300, bbox_inches='tight')

with open('prognostic_T.pkl', 'wb') as f:
    pickle.dump(diff_T, f)

with open('prognostic_Q.pkl', 'wb') as f:
    pickle.dump(diff_Q, f)
