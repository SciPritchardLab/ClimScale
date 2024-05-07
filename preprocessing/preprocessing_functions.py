# Importing packages

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm

# Relative Humidity Conversion Functions

def eliq(T):
    a_liq = np.array([-0.976195544e-15,-0.952447341e-13,\
                                 0.640689451e-10,\
                      0.206739458e-7,0.302950461e-5,0.264847430e-3,\
                      0.142986287e-1,0.443987641,6.11239921]);
    c_liq = -80.0
    T0 = 273.16
    return 100.0*np.polyval(a_liq,np.maximum(c_liq,T-T0))

def eice(T):
    a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,\
                      0.602588177e-7,0.615021634e-5,0.420895665e-3,\
                      0.188439774e-1,0.503160820,6.11147274]);
    c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
    T0 = 273.16
    return np.where(T>c_ice[0],eliq(T),\
                   np.where(T<=c_ice[1],100.0*(c_ice[3]+np.maximum(c_ice[2],T-T0)*\
                   (c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5])),100.0*np.polyval(a_ice,T-T0)))

def esat(T):
    T0 = 273.16
    T00 = 253.16
    omtmp = (T-T00)/(T0-T00)
    omega = np.maximum(0.0,np.minimum(1.0,omtmp))
    return np.where(T>T0,eliq(T),np.where(T<T00,eice(T),(omega*eliq(T)+(1-omega)*eice(T))))

def RH(T,qv,P0,PS,hyam,hybm):
    R = 287.0
    Rv = 461.0
    p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)
    return Rv*p*qv/(R*esat(T))

# Data Processing Functions

def ls(data_path = ""):
    return os.popen(" ".join(["ls", data_path])).read().splitlines()

def load_data(month, year, data_path):
    datasets = ls(data_path)
    month = str(month).zfill(2)
    year = str(year).zfill(4)
    datasets = [data_path + x for x in datasets if "h1." + year + "-" + month in x]
    return xr.open_mfdataset(datasets)

def make_nn_input(sp_data, subsample = True, spacing = 8, contiguous = True):
    nntbsp = sp_data["NNTBSP"].values
    nnqbsp = sp_data["NNQBSP"].values
    heating = (sp_data["NNTASP"] - sp_data["NNTBSP"])/1800
    heating = heating.values
    moistening = (sp_data["NNQASP"] - sp_data["NNQBSP"])/1800
    moistening = moistening.values
    nnpsbsp = sp_data["NNPSBSP"].values[:,None,:,:]
    solin = sp_data["SOLIN"].values[:,None,:,:] 
    nnshfbsp = sp_data["NNSHFBSP"].values[:,None,:,:]
    nnlhfbsp = sp_data["NNLHFBSP"].values[:,None,:,:]
    nnvbsp = sp_data["NNVBSP"].values
    o3vmr = sp_data["O3VMR"].values
    coszrs = sp_data["COSZRS"].values[:,None,:,:]
    # relative humidity conversion
    hyam = sp_data['hyam'].values[:,:,None,None]
    hybm = sp_data['hybm'].values[:,:,None,None]
    p0 = sp_data["P0"].values[:,None,None,None]
    p = p0*hyam + nnpsbsp*hybm
    r = 287.0
    rv = 461.0
    relhum = rv*p*nnqbsp/(r*esat(nntbsp))
    nn_input = np.concatenate((nntbsp[1:,:,:,:], \
                               relhum[1:,5:,:,:], \
                               heating[:-1,:,:,:], \
                               moistening[:-1,5:,:,:], \
                               nnpsbsp[1:,:,:,:], \
                               solin[1:,:,:,:], \
                               nnshfbsp[1:,:,:,:], \
                               nnlhfbsp[1:,:,:,:], \
                               nnvbsp[1:,:,:,:], \
                               o3vmr[1:,:,:,:], \
                               coszrs[1:,:,:,:]), axis = 1)            
    if not contiguous:
        nn_input = nn_input[:-1,:,:,:] #the last timestep of a run can have funky values
    if subsample:
        nn_input = nn_input[:,:,:,::spacing] #subsample longitudes to avoid spatial autocorrelation
    print(nn_input.shape)
    return nn_input

def make_nn_target(sp_data, subsample = True, spacing = 8, contiguous = True):
    heating = (sp_data["NNTASP"] - sp_data["NNTBSP"])/1800
    moistening = (sp_data["NNQASP"] - sp_data["NNQBSP"])/1800
    nn_target = np.concatenate((heating.values[1:,:,:,:], \
                                moistening.values[1:,5:,:,:]), axis = 1)
    if not contiguous:
        nn_target = nn_target[:-1,:,:,:] #the last timestep of a run can have funky values
    if subsample:
        nn_target = nn_target[:,:,:,::spacing]
    print(nn_target.shape)
    return nn_target

def make_nn_test_target(sp_data, subsample = True, spacing = 8, contiguous = True):
    heating = (sp_data["NNTASP"] - sp_data["NNTBSP"])/1800
    moistening = (sp_data["NNQASP"] - sp_data["NNQBSP"])/1800
    nn_target = np.concatenate((heating.values[1:,:,:,:], \
                                moistening.values[1:,:,:,:]), axis = 1)
    if not contiguous:
        nn_target = nn_target[:-1,:,:,:] #the last timestep of a run can have funky values
    if subsample:
        nn_target = nn_target[:,:,:,::spacing]
    print(nn_target.shape)
    return nn_target

def combine_arrays(*args, contiguous = True):
    if contiguous: # meaning each spData was part of the same run
        ans = np.concatenate((args), axis = 0)[:-1,:,:,:]
        print(ans.shape)
        return ans
    ans = np.concatenate((args), axis = 0)
    print(ans.shape)
    return ans

def reshape_input(nn_input):
    nn_input = nn_input.transpose(1,0,2,3)
    ans = nn_input.ravel(order = 'F').reshape(175,-1,order = 'F')
    print(ans.shape)
    return ans

def reshape_target(nn_target):
    nn_target = nn_target.transpose(1,0,2,3)
    ans = nn_target.ravel(order = 'F').reshape(55,-1,order = 'F')
    print(ans.shape)
    return ans

def reverse_reshape(reshaped_arr, original_shape):
    '''
    reshaped_arr should be num_samples x features for this function to work properly
    '''
    arr = reshaped_arr.transpose().reshape(60, original_shape[0], original_shape[2], original_shape[3], order='F')
    ans = arr.transpose(1,0,2,3)
    print(ans.shape)
    return ans

def normalize_input_train(X_train, norm = "standard", save_files = False, norm_path = "../coupling_folder/norm_files/", save_path = "../training/training_data/"):
    train_mu = np.mean(X_train, axis = 1)[:, None]
    train_std = np.std(X_train, axis = 1)[:, None]
    train_min = X_train.min(axis = 1)[:, None]
    train_max = X_train.max(axis = 1)[:, None]
    if norm == "standard":
        inp_sub = train_mu
        inp_div = train_std
        inp_div[inp_div==0] = 1
    elif norm == "range":
        inp_sub = train_min
        inp_div = train_max - train_min
        inp_div[inp_div==0] = 1
    #normalizing
    X_train = ((X_train - inp_sub)/inp_div).transpose()
    #normalized
    print("X_train shape: ")
    print(X_train.shape)
    print("INP_SUB shape: ")
    print(inp_sub.shape)
    print("INP_DIV shape: ")
    print(inp_div.shape)
    if save_files:
        np.save(save_path + "train_input.npy", np.float32(X_train))
        np.savetxt(norm_path + "inp_sub.txt", inp_sub, delimiter=',')
        np.savetxt(norm_path + "inp_div.txt", inp_div, delimiter=',')
    else:
        return X_train, inp_sub, inp_div

def normalize_input_val(X_val, save_files = False, norm_path = "../coupling_folder/norm_files/", save_path = "../training/training_data/"):
    inp_sub = np.loadtxt(norm_path + "inp_sub.txt")[:, np.newaxis]
    inp_div = np.loadtxt(norm_path + "inp_div.txt")[:, np.newaxis]
    print(inp_sub.shape)
    print(inp_div.shape)
    print("loaded in inp_sub and inp_div")
    X_val = ((X_val - inp_sub)/inp_div).transpose()
    print("X_val shape: ")
    print(X_val.shape)
    print("INP_SUB shape: ")
    print(inp_sub.shape)
    print("INP_DIV shape: ")
    print(inp_div.shape)
    if save_files:
        np.save(save_path + "val_input.npy", np.float32(X_val))
    else:
        return X_val

def normalize_input_test(X_test, save_files = False, norm_path = "../coupling_folder/norm_files/", save_path = "../training/training_data/"):
    inp_sub = np.loadtxt(norm_path + "inp_sub.txt")[:, np.newaxis]
    inp_div = np.loadtxt(norm_path + "inp_div.txt")[:, np.newaxis]
    print(inp_sub.shape)
    print(inp_div.shape)
    print("loaded in inp_sub and inp_div")
    X_test = ((X_test - inp_sub)/inp_div).transpose()
    print("X_test shape: ")
    print(X_test.shape)
    print("INP_SUB shape: ")
    print(inp_sub.shape)
    print("INP_DIV shape: ")
    print(inp_div.shape)
    if save_files:
        np.save(save_path + "test_input.npy", np.float32(X_test))
    else:
        return X_test

def normalize_target_train(y_train_original, save_files = False, norm_path = "../coupling_folder/norm_files/", save_path = "../training/training_data/"):
    y_train = y_train_original.copy()
    out_scale = 1/(np.maximum(np.std(y_train, axis = 1), 1e-12))
    y_train[0:30,:] = y_train[0:30,:]*out_scale[0:30,None]
    y_train[30:55,:] = y_train[30:55,:]*out_scale[30:55,None]      
    y_train = y_train.transpose()
    print("y shape: ")
    print(y_train.shape)
    out_scale = out_scale[:, None]
    print("out_scale shape: ")
    print(out_scale.shape)
    if save_files:
        np.save(save_path + "train_target.npy", np.float32(y_train))
        np.savetxt(norm_path + "out_scale.txt", out_scale, delimiter=',')
    else:
        return y_train

def normalize_target_val(y_val_original, save_files = False, norm_path = "../coupling_folder/norm_files/", save_path = "../training/training_data/"):
    y_val = y_val_original.copy()
    out_scale = np.loadtxt(norm_path + "out_scale.txt")
    y_val[0:30,:] = y_val[0:30,:]*out_scale[0:30,None]
    y_val[30:55,:] = y_val[30:55,:]*out_scale[30:55,None]    
    y_val = y_val.transpose()
    print("y_val shape: ")
    print(y_val.shape)
    print("out_scale shape: ")
    out_scale = out_scale[:, None]
    print(out_scale.shape)
    if save_files:
        np.save(save_path + "val_target.npy", np.float32(y_val))
    else:
        return y_val

def normalize_target_test(y_test_original, save_files = False, norm_path = "../coupling_folder/norm_files/", save_path = "../training/training_data/"):
    y_test = y_test_original.copy()
    out_scale = np.loadtxt(norm_path + "out_scale.txt")
    y_test[0:30,:] = y_test[0:30,:]*out_scale[0:30,None]
    y_test[30:55,:] = y_test[30:55,:]*out_scale[30:55,None]    
    y_test = y_test.transpose()
    print("y_test shape: ")
    print(y_test.shape)
    print("out_scale shape: ")
    out_scale = out_scale[:, None]
    print(out_scale.shape)
    if save_files:
        np.save(save_path + "test_target.npy", np.float32(y_test))
    else:
        return y_test
