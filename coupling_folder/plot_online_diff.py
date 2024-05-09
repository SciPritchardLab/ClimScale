import xarray as xr
import numpy as np
import pandas as pd
import scipy.stats as stats
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

offline_lower_lim_heating = 1e-5
offline_upper_lim_heating = 4.5e-5
offline_lower_lim_moistening = 1e-5
offline_upper_lim_moistening = 4.5e-5
online_lower_lim_temperature = 3e-1
online_upper_lim_temperature = 2e2
online_lower_lim_moisture = 1e-1
online_upper_lim_moisture = 3e1

suptitle_size = 15
figtext_size = 12
axislabel_size = 12
dotsize = 6

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

sp_temp = np.mean(get_monthly_mean(sp_data_year["NNTBSP"].values), axis = 3)
sp_hum = np.mean(get_monthly_mean(sp_data_year["NNQBSP"].values), axis = 3)
sp_temp_lag = np.mean(get_monthly_mean(sp_data_lag["NNTBSP"].values), axis = 3)
sp_hum_lag = np.mean(get_monthly_mean(sp_data_lag["NNQBSP"].values), axis = 3)

dp = sp_data_year['gw'] * (sp_data_year["P0"] * sp_data_year["hyai"] + sp_data_year['hybi']*sp_data_year['NNPSBSP']).diff(dim = "ilev")
error_weights = dp.groupby('time.month').mean('time').mean('lon')
error_weights = error_weights.values/(error_weights.sum(dim = ['lat', 'ilev']).values[:,None,None])
error_weights = np.transpose(error_weights, (0,2,1))

lagged_temp = np.sqrt(np.sum(((sp_temp_lag - sp_temp)**2)*error_weights, axis = (1,2)))
lagged_hum = np.sqrt(np.sum(((sp_hum_lag - sp_hum)**2)*error_weights, axis = (1,2)))*1000

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
    arr = np.array(arr.groupby("time.month").mean("time").mean("lon"))
    if var == "NNTBSP":
        sp_vals = sp_temp[:end_length,:,:]
    elif var == "NNQBSP":
        sp_vals = sp_hum[:end_length,:,:]
    se = (sp_vals-arr[:end_length,:,:])**2
    wse = error_weights[:end_length,:,:]*se
    if var == "NNTBSP":
        return np.sum(wse, axis = (1,2))**.5
    elif var == "NNQBSP":
        return (np.sum(wse, axis = (1,2))**.5)*1000

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
        offline_lower_lim = offline_lower_lim_heating
        offline_upper_lim = offline_upper_lim_heating
        cmap = plt.get_cmap('plasma')
        sm = ScalarMappable(cmap = cmap)
        sm.set_array(np.linspace(offline_lower_lim, offline_upper_lim, 100))
        for i in range(len(config_diffs)):
            assert offline_error[i,0] > offline_lower_lim
            color_index = (offline_error[i,0] - offline_lower_lim)/(offline_upper_lim - offline_lower_lim)
            if color_index > 1.0:
                color_index = 1.0 # needs to be floating point for accurate cmap
            axnum.plot(config_diffs[i+1], color = cmap(color_index), linewidth = .25)
        var_label = "Temperature"
        cbar = plt.colorbar(sm, ax = axnum, pad = .04)
        cbar.set_label('offline test error (K/s)', size = axislabel_size, rotation = 270, labelpad = 19)
        axnum.plot(lagged_temp, color = "black", linewidth = .8)
        axnum.set_ylim((online_lower_lim_temperature, online_upper_lim_temperature))
    if var == "NNQBSP":
        offline_lower_lim = offline_lower_lim_moistening
        offline_upper_lim = offline_upper_lim_moistening
        cmap = plt.get_cmap('winter')
        sm = ScalarMappable(cmap = cmap)
        sm.set_array(np.linspace(offline_lower_lim, offline_upper_lim, 100))
        for i in range(len(config_diffs)):
            assert offline_error[i,1] > offline_lower_lim
            color_index = (offline_error[i,1] - offline_lower_lim)/(offline_upper_lim - offline_lower_lim)
            if color_index > 1.0:
                color_index = 1.0 # needs to be floating point for accurate cmap
            axnum.plot(config_diffs[i+1], color = cmap(color_index), linewidth = .25)
        var_label = "Humidity"
        cbar = plt.colorbar(sm, ax = axnum, pad = .04)
        cbar.set_label('offline test error (g/kg/s)', size = axislabel_size, rotation = 270, labelpad = 19)
        axnum.plot(lagged_hum, color = "black", linewidth = .8)
        axnum.set_ylim((online_lower_lim_moisture, online_upper_lim_moisture))
    
    patches = [mpatches.Patch(facecolor = x) for x in colors]

    axnum.legend(handles = patches, \
            labels = legend_names, \
            loc = "upper right", \
            borderaxespad = 0.1, \
            fontsize = 12)
            
    axnum.set_title(var_label + " Root Mean Squared Error (RMSE)", fontsize = 16)
    if logy:
        axnum.set_yscale("log")
    
    axnum.set_xlabel("month", fontsize = axislabel_size)
    if var == "NNTBSP":
        axnum.set_ylabel("online monthly zonal RMSE (K)", fontsize = axislabel_size)
    if var == "NNQBSP":
        axnum.set_ylabel("online monthly zonal RMSE (g/kg)", fontsize = axislabel_size)

    axnum.grid(True, which='major', linestyle='-', linewidth=0.5)
    axnum.grid(True, which='minor', linestyle=':', linewidth=0.25)

prognostic_T = load_run(config_subdir, "NNTBSP")
prognostic_Q = load_run(config_subdir, "NNQBSP")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_diff(ax1, config_name, prognostic_T, offline_test_error, 'NNTBSP')
plot_diff(ax2, config_name, prognostic_Q, offline_test_error, 'NNQBSP')

fig.suptitle(config_name + ' configuration monthly prognostic error', fontsize = suptitle_size)

# plt.subplots_adjust(wspace=0.01)

plt.tight_layout()
fig.savefig('online_diffs.png', dpi = 300, bbox_inches='tight')

with open('prognostic_T.pkl', 'wb') as f:
    pickle.dump(prognostic_T, f)

with open('prognostic_Q.pkl', 'wb') as f:
    pickle.dump(prognostic_Q, f)

model_info = pd.read_pickle('RESULTS_' + config_subdir + '_sorted.pandas.pkl')
model_info['offline_heating'] = pd.Series(offline_test_error[:,0], name = 'offline_heating', index = model_info.index)
model_info['offline_moistening'] = pd.Series(offline_test_error[:,1], name = 'offline_moistening', index = model_info.index)
model_info['num_months'] = pd.Series([len(prognostic_T[x]) for x in model_info.index], name = 'num_months', index = model_info.index)
assert model_info['num_months'].equals(pd.Series([len(prognostic_Q[x]) for x in model_info.index], name = 'num_months', index = model_info.index))
model_info['online_temperature'] = pd.Series([np.mean(prognostic_T[x]) if len(prognostic_T[x])==12 else None for x in model_info.index], name = 'prognostic_T', index = model_info.index)
model_info['online_moisture'] = pd.Series([np.mean(prognostic_Q[x]) if len(prognostic_Q[x])==12 else None for x in model_info.index], name = 'prognostic_Q', index = model_info.index)

model_info.to_pickle(config_subdir + '_df.pandas.pkl')
survival_time_ratio = model_info['num_months'][(model_info['num_months'] > 0) & (model_info['num_months'] < 12)]/12
prognostic_T_incomplete = np.array([np.mean(prognostic_T[x]) for x in prognostic_T if len(prognostic_T[x]) > 0 and len(prognostic_T[x]) < 12])
prognostic_Q_incomplete = np.array([np.mean(prognostic_Q[x]) for x in prognostic_Q if len(prognostic_Q[x]) > 0 and len(prognostic_Q[x]) < 12])
diagnostic_T_incomplete = model_info['offline_heating'][(model_info['num_months'] > 0) & (model_info['num_months'] < 12)]
diagnostic_Q_incomplete = model_info['offline_moistening'][(model_info['num_months'] > 0) & (model_info['num_months'] < 12)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot for offline heating vs online temperature
x = model_info['offline_heating'][model_info['online_temperature'].notna()]
y = model_info['online_temperature'][model_info['online_temperature'].notna()]
slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, np.log(y))
ax1.scatter(x, y, c = np.ones(len(x)), cmap = 'Reds', s = dotsize, vmin = 0, vmax = 1)
sc1 = ax1.scatter(diagnostic_T_incomplete, prognostic_T_incomplete, c = survival_time_ratio, cmap = 'Reds', marker = 'x', s = 19, vmin = 0, vmax = 1)
plt.colorbar(sc1, ax=ax1)
ax1.text(0.05, 0.92, f'R-squared: {rvalue**2:.2f}', transform=ax1.transAxes, verticalalignment='top', fontsize = figtext_size)
ax1.text(0.05, 0.86, f'p-value: {pvalue:.2f}', transform=ax1.transAxes, verticalalignment='top', fontsize = figtext_size)
# Fit a line to the data
line_lims = np.array([offline_lower_lim_heating, offline_upper_lim_heating])
ax1.plot(line_lims, np.exp(slope*line_lims + intercept), color='black', linewidth=.8, linestyle='--')
ax1.set_xlim([offline_lower_lim_heating, offline_upper_lim_heating])
ax1.set_ylim([online_lower_lim_temperature, online_upper_lim_temperature])
ax1.set_xlabel('offline heating RMSE (K/s)', fontsize = axislabel_size)
ax1.set_ylabel('online temperature RMSE (K)', fontsize = axislabel_size)
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--")

# Scatter plot for offline moistening vs online moisture
x = model_info['offline_moistening'][model_info['online_moisture'].notna()]
y = model_info['online_moisture'][model_info['online_moisture'].notna()]
slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, np.log(y))
ax2.scatter(x, y, c = np.ones(len(x)), cmap='Blues', s = dotsize, vmin = 0, vmax = 1)
sc2 = ax2.scatter(diagnostic_Q_incomplete, prognostic_Q_incomplete, c = survival_time_ratio, cmap = 'Blues', marker = 'x', s = 19, vmin = 0, vmax = 1)
cbar = plt.colorbar(sc2, ax = ax2)
cbar.set_label('Survival Time Ratio', size = axislabel_size, rotation = 270, labelpad = 19)
corr_coef = np.corrcoef(x, np.log(y))[0, 1]
ax2.text(0.05, 0.92, f'R-squared: {rvalue**2:.2f}', transform=ax2.transAxes, verticalalignment='top', fontsize = figtext_size)
ax2.text(0.05, 0.86, f'p-value: {pvalue:.2f}', transform=ax2.transAxes, verticalalignment='top', fontsize = figtext_size)

# Fit a line to the data
line_lims = np.array([offline_lower_lim_moistening, offline_upper_lim_moistening])
ax2.plot(line_lims, np.exp(slope*line_lims + intercept), color='black', linewidth=.8, linestyle='--')
ax2.set_xlim([offline_lower_lim_moistening, offline_upper_lim_moistening])
ax2.set_ylim([online_lower_lim_moisture, online_upper_lim_moisture])
ax2.set_xlabel('offline moistening RMSE (g/kg/s)', fontsize = axislabel_size)
ax2.set_ylabel('online moisture RMSE (g/kg)', fontsize = axislabel_size)
ax2.set_yscale('log')
ax2.grid(True, which="both", ls="--")

fig.suptitle('Offline vs. Online RMSE for Heating and Moistening, ' + config_name + ' configuration', fontsize = suptitle_size)

plt.tight_layout()
fig.savefig('offlinevonline.png', dpi = 300, bbox_inches='tight')
