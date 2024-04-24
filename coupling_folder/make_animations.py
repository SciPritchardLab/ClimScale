import os, glob
import shutil
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy
import cartopy.feature as cfeature

import cartopy.crs as ccrs
import matplotlib.pylab as plb
import matplotlib.image as imag

from moviepy.editor import *

sp_path = '/ocean/projects/atm200007p/jlin96/longSPrun_clean/'
h1_path = './'

start_day = 0
final_day = len(glob.glob('*cam2.h1.0000*'))
fps = 12

def load_run(run_path, start_day, final_day):
    run_list = os.popen(" ".join(["ls", run_path + "*.h1.000*"])).read().splitlines()
    run_data = xr.open_mfdataset(run_list[start_day:final_day], compat = 'override', join = 'override', coords = "minimal")
    return run_data

sp_data = load_run(sp_path, start_day, final_day)
nn_data = load_run(h1_path, start_day, final_day)

lats = sp_data['lat'].values
lons = sp_data['lon'].values

pressure_grid_p1 = (sp_data['P0'].values[:, np.newaxis] * sp_data['hyai'].values)[:,:,np.newaxis, np.newaxis]
pressure_grid_p2 = sp_data['hybi'].values[:,:,np.newaxis,np.newaxis]*sp_data['NNPSBSP'].values[:,np.newaxis,:,:]
pressure_grid = pressure_grid_p1 + pressure_grid_p2
dp = pressure_grid[:,1:31,:,:] - pressure_grid[:,0:30,:,:]

dp_sum = np.sum(dp, axis = 1)[:,None,:,:]
pressure_weighting = dp/dp_sum

sp_data["heating"] = (sp_data["NNTASP"] - sp_data["NNTBSP"])/1800
sp_data["moistening"] = (sp_data["NNQASP"] - sp_data["NNQBSP"])/1800

nn_data["heating"] = (nn_data["NNTASP"] - nn_data["NNTBSP"])/1800
nn_data["moistening"] = (nn_data["NNQASP"] - nn_data["NNQBSP"])/1800

def plot_diff(xr_data, var, save_files = True):
    num_timesteps = len(xr_data['time'])
    if var == 'NNTBSP':
        map_cmap = plb.cm.afmhot
        cmap = 'bwr'
        vmin = -5
        vmax = 5
    elif var == 'NNQBSP':
        map_cmap = plb.cm.bone
        cmap = 'BrBG'
        vmin = -.0015
        vmax = .0015
    elif var == 'heating':
        map_cmap = plb.cm.afmhot
        cmap = 'bwr'
        vmin = -4e-4
        vmax = 4e-4
    elif var == 'moistening':
        map_cmap = plb.cm.bone
        cmap = 'BrBG'
        vmin = -5e-7
        vmax = 5e-7
    elif var == 'TTEND':
        map_cmap = plb.cm.bone
        cmap = 'BrBG'
        vmin = -1
        vmax = 1
    map_cmaplist = [map_cmap(i) for i in range(map_cmap.N)]
    map_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', map_cmaplist, map_cmap.N)
    bounds = np.linspace(vmin, vmax, 11)
    norm = mpl.colors.BoundaryNorm(bounds, map_cmap.N)

    latlev_image_files = []
    worldmap_image_files = []
    if os.path.exists(var):
        shutil.rmtree(var)
    os.makedirs(var, exist_ok=True)
    os.makedirs(var + '/latlev/', exist_ok = True)
    os.makedirs(var + '/worldmap/', exist_ok = True)
    var_diff = (xr_data[var] - sp_data[var])
    map_var_diff = np.sum(var_diff*pressure_weighting[:num_timesteps,...], axis = 1)
    for t in range(0, num_timesteps, 48):
        day = int(t/48)
        var_diff.isel(time = slice(t, t+48)).mean(dim = ['time','lon']).plot(x='lat', y='lev', yincrease=False, cmap = cmap, vmin = vmin, vmax = vmax)
        latlev_image_file = var + "/latlev/" + var + str(day).zfill(3) + ".png"
        title_str = var + ' difference on day ' + str(day)
        plt.title(title_str)
        plt.savefig(latlev_image_file)
        latlev_image_files.append(latlev_image_file)
        plt.clf()
        plt.close('all')
        fig, ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.Robinson(central_longitude=180)},
                        figsize=(15,6))
        ax.gridlines()
        ax.set_title(title_str)
        imag = ax.pcolormesh(lons, lats, map_var_diff[t,:,:], transform=ccrs.PlateCarree(), cmap=cmap,norm=norm)
        cbar_ax = fig.add_axes([0.84, 0.12, 0.02, 0.76])
        fig.colorbar(imag, cax = cbar_ax) 
        worldmap_image_file = var + "/worldmap/" + var + str(day).zfill(3) + ".png"
        plt.savefig(worldmap_image_file)
        worldmap_image_files.append(worldmap_image_file)
        plt.clf()
        plt.close('all')
    latlev_clip = ImageSequenceClip(sorted(latlev_image_files), fps = fps)
    worldmap_clip = ImageSequenceClip(sorted(worldmap_image_files), fps = fps)
    if save_files:
        latlev_clip.write_videofile(var + '/latlev/' + var + '_latlev.mp4')
        worldmap_clip.write_videofile(var + '/worldmap/' + var + '_worldmap.mp4')
    else:
        return latlev_clip, worldmap_clip

def plot_diff_sum(xr_data, var, save_files = True):
    num_timesteps = len(xr_data['time'])
    if var == 'NNTBSP':
        map_cmap = plb.cm.afmhot
        cmap = 'bwr'
        vmin = -5
        vmax = 5
    elif var == 'NNQBSP':
        map_cmap = plb.cm.bone
        cmap = 'BrBG'
        vmin = -.0015
        vmax = .0015
    elif var == 'heating':
        map_cmap = plb.cm.afmhot
        cmap = 'bwr'
        vmin = -4e-4
        vmax = 4e-4
    elif var == 'moistening':
        map_cmap = plb.cm.bone
        cmap = 'BrBG'
        vmin = -5e-7
        vmax = 5e-7
    elif var == 'TTEND':
        map_cmap = plb.cm.bone
        cmap = 'BrBG'
        vmin = -1
        vmax = 1
    map_cmaplist = [map_cmap(i) for i in range(map_cmap.N)]
    map_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', map_cmaplist, map_cmap.N)
    bounds = np.linspace(vmin, vmax, 11)
    norm = mpl.colors.BoundaryNorm(bounds, map_cmap.N)

    latlev_image_files = []
    worldmap_image_files = []
    if os.path.exists(var):
        shutil.rmtree(var)
    os.makedirs(var, exist_ok=True)
    os.makedirs(var + '_cum/latlev/', exist_ok = True)
    os.makedirs(var + '_cum/worldmap/', exist_ok = True)
    var_diff = (xr_data[var] - sp_data[var])
    map_var_diff = np.sum(var_diff*pressure_weighting[:num_timesteps,...], axis = 1)
    for t in range(0, num_timesteps):
        day = int(t/48)
        if t%48 == 0:
            var_diff.isel(time = slice(0, t)).sum(dim = 'time').mean(dim = 'lon').plot(x='lat', y='lev', yincrease=False, cmap = cmap, vmin = vmin, vmax = vmax)
            latlev_image_file = var + "_cum/latlev/" + var + str(t).zfill(3) + ".png"
            title_str = var + ' cumulative difference on day ' + str(day)
            plt.title(title_str)
            plt.savefig(latlev_image_file)
            latlev_image_files.append(latlev_image_file)
            plt.clf()
            plt.close('all')
            fig, ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.Robinson(central_longitude=180)},
                            figsize=(15,6))
            ax.gridlines()
            ax.set_title(title_str)
            imag = ax.pcolormesh(lons, lats, map_var_diff[t,:,:], transform=ccrs.PlateCarree(), cmap=cmap,norm=norm)
            cbar_ax = fig.add_axes([0.84, 0.12, 0.02, 0.76])
            fig.colorbar(imag, cax = cbar_ax) 
            worldmap_image_file = var + "_cum/worldmap/" + var + str(t).zfill(3) + ".png"
            plt.savefig(worldmap_image_file)
            worldmap_image_files.append(worldmap_image_file)
            plt.clf()
            plt.close('all')
    latlev_clip = ImageSequenceClip(sorted(latlev_image_files), fps = fps)
    worldmap_clip = ImageSequenceClip(sorted(worldmap_image_files), fps = fps)
    if save_files:
        latlev_clip.write_videofile(var + '_cum/latlev/' + var + '_latlev.mp4')
        worldmap_clip.write_videofile(var + '_cum/worldmap/' + var + '_worldmap.mp4')
    else:
        return latlev_clip, worldmap_clip

plot_diff_sum(nn_data, 'heating')
plot_diff_sum(nn_data, 'moistening')
plot_diff(nn_data, 'NNTBSP')
plot_diff(nn_data, 'NNQBSP')
