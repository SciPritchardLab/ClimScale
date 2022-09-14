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

def ls(keyword = ""):
    os.popen(" ".join(["ls" + keyword])).read().splitlines()

def do_month(month):
    datasets = ls()
    n = str(month)
    datasets = [x for x in datasets if "h1.0000-" + n.zfill(2) in x]
    return xr.open_mfdataset(datasets)

def make_nninput(spData, family, save_diagnostics = False, full_run = False):
    nntbp = spData["NNTBP"].values
    nnqbp = spData["NNQBP"].values
    p0 = spData["P0"].values
    ps = spData["NNPS"].values
    hyam = spData["hyam"].values
    hybm = spData["hybm"].values
    relhum = spData["RELHUM"].values
    tphystnd = spData["TPHYSTND"].values
    phq = spData["PHQ"].values

    p0 = np.array(list(set(p0)))
    print("loaded in data")
    newhum = np.zeros((spData["time"].shape[0],\
                                  spData["lev"].shape[0], \
                                  spData["lat"].shape[0], \
                                  spData["lon"].shape[0]))
    lats = spData["lat"]
    lons = spData["lon"]
    print("starting for loop")
    for i in tqdm(range(len(lats))):
        for j in range(len(lons)):
            latIndex = i
            lonIndex = j
            R = 287.0
            Rv = 461.0
            p = p0 * hyam + ps[:, None, latIndex, lonIndex] * hybm # Total pressure (Pa)
            T = nntbp[:, :, latIndex, lonIndex]
            qv = nnqbp[:, :, latIndex, lonIndex]
            newhum[:,:, latIndex, lonIndex] = Rv*p*qv/(R*esat(T))
    
    nntbp = np.moveaxis(nntbp[1:,:,:,:],0,1)
    print("nntbp")
    print(nntbp.shape)
    
    nnqbp = np.moveaxis(nnqbp[1:,:,:,:],0,1)
    print("nnqbp")
    print(nnqbp.shape)
    
    lhflx = spData["LHFLX"].values[np.newaxis,:-1,:,:]
    print("lhflx")
    print(lhflx.shape)
    
    shflx = spData["SHFLX"].values[np.newaxis,:-1,:,:]
    print("shflx")
    print(shflx.shape)
    
    ps = spData["NNPS"].values[np.newaxis,1:,:,:]
    print("ps")
    print(ps.shape)
    
    solin = spData["SOLIN"].values[np.newaxis,1:,:,:]
    print("solin")
    print(solin.shape)
    
    newhum = np.moveaxis(newhum[1:,:,:,:],0,1)
    print("newhum")
    print(newhum.shape)
    
    oldhum = np.moveaxis(relhum[1:,:,:,:],0,1)
    print("oldhum")
    print(oldhum.shape)
    
    tphystnd = np.moveaxis(tphystnd[1:,:,:,:],0,1)
    print("tphystnd")
    print(tphystnd.shape)
    
    phq = np.moveaxis(phq[1:,:,:,:],0,1)
    print("phq")
    print(phq.shape)
        
    if family == "specific":
        nnInput = np.concatenate((nntbp, \
                                  nnqbp, \
                                  ps, \
                                  solin, \
                                  shflx, \
                                  lhflx))
        
        nnTarget = np.concatenate((tphystnd, phq))
    
    elif family == "relative":
        nnInput = np.concatenate((nntbp, \
                                  newhum, \
                                  ps, \
                                  solin, \
                                  shflx, \
                                  lhflx))
                             
        nnTarget = np.concatenate((tphystnd, phq))
    
    if full_run:
        nnInput = nnInput[:,:-1,:,:] #the last timestep of a run can have funky values
        nnTarget = nnTarget[:,:-1,:,:] #the last timestep of a run can have funky values
    
    print("nnInput")
    nnInput.shape
    
    print("nnTarget")
    nnTarget.shape
    
    errors = (newhum-oldhum/100).flatten()
    result = "Mean relative humidity conversion error: " + str(np.mean(errors)) + "\n"
    result = result + "Variance for relative humidity conversion error: " + str(np.var(errors)) + "\n"
    result = result + "nntbp.shape: " + str(nntbp.shape) + "\n"
    result = result + "nnqbp.shape: " + str(nnqbp.shape) + "\n"
    result = result + "lhflx.shape: " + str(lhflx.shape) + "\n"
    result = result + "shflx.shape: " + str(shflx.shape) + "\n"
    result = result + "ps.shape: " + str(ps.shape) + "\n"
    result = result + "solin.shape: " + str(solin.shape) + "\n"
    result = result + "newhum.shape: " + str(newhum.shape) + "\n"
    result = result + "oldhum.shape: " + str(oldhum.shape) + "\n"
    result = result + "tphystnd.shape: " + str(tphystnd.shape) + "\n"
    result = result + "phq.shape: " + str(phq.shape) + "\n"
    result = result + "nnInput.shape: " + str(nnInput.shape) + "\n"
    print(result)

    if save_diagnostics:
        diagnostics = 'diagnostics_' + str(month) + '.txt'
        with open(diagnostics, 'a') as fp:
            fp.write(result)
    
    return nnInput, nnTarget

def combine_arrays(*args, contiguous = True):
    if contiguous: # meaning each spData was part of the same run
        return np.concatenate((args), axis = 1)[:,:-1,:,:]
    return(np.concatenate((args), axis = 1))
                         
def sample_indices(size, spacing, fixed = True):
    numIndices = np.round(size/spacing)
    if fixed:
        indices = np.array([int(x) for x in np.round(np.linspace(1,size,int(numIndices)))])-1
    else:
        indices = list(range(size))
        np.random.shuffle(indices)
        indices = indices[0:int(numIndices)]
    return indices

def reshape_input(nnData, subsample = False, spacing = 5):
    if subsample:
        nnData = nnData[:,:,:,sampleIndices(nnData.shape[3], spacing, True)]
    nnData = nnData.ravel(order = 'F').reshape(64,-1,order = 'F')
    return nnData

def reshape_target(nnData, subsample = False, spacing = 5):
    if subsample:
        nnData = nnData[:,:,:,sampleIndices(nnData.shape[3], spacing, True)]
    nnData = nnData.ravel(order = 'F').reshape(60,-1,order = 'F')
    return nnData

def normalize_input_train(X_train, reshaped = True, normalization = "standard", family_name = "", save_path = "", save_files = False):
    
    if reshaped:
        train_mu = np.mean(X_train, axis = 1)[:, np.newaxis]
        train_std = np.std(X_train, axis = 1)[:, np.newaxis]
        train_min = X_train.min(axis = 1)[:, np.newaxis]
        train_max = X_train.max(axis = 1)[:, np.newaxis]
    
    else:
        train_mu = np.mean(X_train, axis = (1,2,3))[:, np.newaxis]
        train_std = np.std(X_train, axis = (1,2,3))[:, np.newaxis]
        train_min = X_train.min(axis = (1,2,3))[:, np.newaxis]
        train_max = X_train.max(axis = (1,2,3))[:, np.newaxis]
        
    if normalization == "standard":
        inpsub = train_mu
        inpdiv = train_std
        
    elif normalization == "range":
        inpsub = train_min
        inpdiv = train_max - train_min
        
    #normalizing
    X_train = ((X_train - inpsub)/inpdiv)
    #normalized
    
    print("X_train shape: ")
    print(X_train.shape)
    print("INP_SUB shape: ")
    print(inpsub.shape)
    print("INP_DIV shape: ")
    print(inpdiv.shape)
    
    if save_files:
        with open(save_path + "trainInput.npy", 'wb') as f:
            np.save(f, np.float32(X_train))
        np.savetxt(save_path + "inp_sub.txt", inpsub, delimiter=',')
        np.savetxt(save_path + "inp_div.txt", inpdiv, delimiter=',')
        return
    
    return X_train, inpsub, inpdiv


def normalize_input_val(X_val, inpsub, inpdiv,  family_name = "", save_path = "", save_files = False):
    #normalizing
    X_val = (X_val - inpsub)/inpdiv
    
    print("X_val shape: ")
    print(X_val.shape)
    print("INP_SUB shape: ")
    print(inpsub.shape)
    print("INP_DIV shape: ")
    print(inpdiv.shape)
    
    if save_files:
        with open(save_path + "valInput.npy", 'wb') as f:
            np.save(f, np.float32(X_val))
        return
    
    return X_val


def normalize_target_train(y, reshaped = True, family_name = "", save_path = "", save_files = False):
    
    # specific heat of air = 1004 J/ K / kg
    # latent heat of vaporization 2.5*10^6

    heatScale = 1004
    moistScale = 2.5e6
    outscale = np.concatenate((np.repeat(heatScale, 30), np.repeat(moistScale, 30)))
    
    if reshaped:
        y[0:30,:] = y[0:30,:]*outscale[0:30, np.newaxis]
        y[30:60,:] = y[30:60,:]*outscale[30:60, np.newaxis]
    else:
        y[0:30,:] = y[0:30,:]*outscale[0:30, np.newaxis, np.newaxis, np.newaxis]
        y[30:60,:] = y[30:60,:]*outscale[30:60, np.newaxis, np.newaxis, np.newaxis]        
    
    print("y shape: ")
    print(y.shape)
    print("outscale shape: ")
    print(outscale.shape)
    
    if save_files:
        with open(save_path + "trainOutput.npy", 'wb') as f:
            np.save(f, np.float32(y))
        return

    return y


def normalize_target_val(y, reshaped = True, family_name = "", save_path = "", save_files = False):
    
    # specific heat of air = 1004 J/ K / kg
    # latent heat of vaporization 2.5*10^6

    heatScale = 1004
    moistScale = 2.5e6
    outscale = np.concatenate((np.repeat(heatScale, 30), np.repeat(moistScale, 30)))
    
    if reshaped:
        y[0:30,:] = y[0:30,:]*outscale[0:30, np.newaxis]
        y[30:60,:] = y[30:60,:]*outscale[30:60, np.newaxis]
    else:
        y[0:30,:] = y[0:30,:]*outscale[0:30, np.newaxis, np.newaxis, np.newaxis]
        y[30:60,:] = y[30:60,:]*outscale[30:60, np.newaxis, np.newaxis, np.newaxis]        
    
    print("y shape: ")
    print(y.shape)
    print("outscale shape: ")
    print(outscale.shape)
    
    if save_files:
        with open(save_path + "valOutput.npy", 'wb') as f:
            np.save(f, np.float32(y))
        return
    return y


