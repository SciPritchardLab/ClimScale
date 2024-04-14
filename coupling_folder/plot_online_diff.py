import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches

import os
import re
import math
from tqdm import tqdm
import sys

import pickle

config_subdir = sys.argv[1]
config_name = sys.argv[2]
config_color = sys.argv[3]

sp_path = "/ocean/projects/atm200007p/jlin96/sp_proxy/spcamTrim/"
num_runs = len(os.listdir("../coupled_results/")) - 1

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

dp = sp_data['gw'] * (sp_data["P0"] * sp_data["hyai"] + sp_data['hybi']*sp_data['NNPS']).diff(dim = "ilev")
error_weights = dp.groupby('time.month').mean('time')
error_weights_tropo = error_weights.sel(ilev = slice(12,30))
error_weights = error_weights.values/(error_weights.sum(dim = ['lat', 'ilev', 'lon']).values[:,None,None,None])
error_weights = np.transpose(error_weights, (0,2,1,3))

sp_data["NNTBSP"] = sp_data["NNTBP"]
sp_data["NNQBSP"] = sp_data["NNQBP"]

sp_temp = np.array(sp_data["NNTBSP"].groupby("time.month").mean("time"))
sp_hum = np.array(sp_data["NNQBSP"].groupby("time.month").mean("time"))

month_length = np.array([len(sp_data.groupby("time.month")[x]["time"]) for x in np.array(range(12))+1])
month_sum = np.cumsum(month_length)
month_sum_start = np.insert(month_sum, 0, 0)[:-1]
month_start_stop = [(month_sum_start[i],month_sum[i]) for i in range(12)]

month_length_days = month_length/48
month_sum_days = np.cumsum(month_length_days)

lagged_temp = np.array([1.22844432, 1.1918091 , 0.94198483, 1.10117341, 1.14462599,
       1.18472305, 1.19370457, 1.07093466, 1.10986774, 1.1666777 ,
       1.10737392, 0.99146933])
lagged_hum = np.array([0.00043787, 0.00051397, 0.00042736, 0.00045834, 0.00046532,
       0.00052819, 0.00044416, 0.0003921 , 0.00042971, 0.00044279,
       0.00044991, 0.00039916])

lat = np.array(sp_data["lat"])
lon = np.array(sp_data["lon"])
lev = np.array(sp_data["lev"])

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
    end_length = np.arange(1,13)[arr.shape[0]>=month_sum_days][-1] #such that only complete months are added
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
    if arr.shape[0] < 31:
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

def plot_diff(axnum, config_name, config_diffs, var, color, logy = True):
    colors = [color, "black"]
    legend_names = [config_name, "internal variability proxy"]

    if var == "NNTBSP":
        for i in range(len(config_diffs)):
            axnum.plot(config_diffs[i+1], color = colors[0], linewidth = .25)
        var_label = "Temperature"
        axnum.plot(lagged_temp, color = "black", linewidth = .8)
        axnum.set_ylim((.8e0, 2e2))
    if var == "NNQBSP":
        for i in range(len(config_diffs)):
            axnum.plot(config_diffs[i+1]*1000, color = colors[0], linewidth = .25)
        var_label = "Humidity"
        axnum.plot(lagged_hum*1000, color = "black", linewidth = .8)
        axnum.set_ylim((3e-1, 3e1))
    
    patches = [mpatches.Patch(facecolor = x) for x in colors]

    axnum.legend(handles = patches, \
            labels = legend_names, \
            loc = "upper right", \
            borderaxespad = 0.1, \
            fontsize = 15)
            
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

plot_diff(ax1, config_name, diff_T, 'NNTBSP', config_color)
plot_diff(ax2, config_name, diff_Q, 'NNQBSP', config_color)

plt.subplots_adjust(0,0,2,1)

plt.tight_layout()
fig.savefig('online_diffs.png', dpi = 300, bbox_inches='tight')

with open('prognostic_T.pkl', 'wb') as f:
    pickle.dump(diff_T, f)

with open('prognostic_Q.pkl', 'wb') as f:
    pickle.dump(diff_Q, f)
