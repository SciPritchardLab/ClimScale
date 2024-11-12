import os, glob
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import json
import csv
#import sys
#import yaml

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from tensorflow.keras import callbacks
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.activations import relu
from tensorflow.keras.optimizers import SGD
import tensorflow_addons as tfa
from qhoptim.tf import QHAdamOptimizer

from tqdm import tqdm

import matplotlib.pyplot as plt

proj_name = 'noozone'
tuning_dir = '../training/tuning_directory/' + proj_name + "/"
trial_root = os.path.join(tuning_dir, 'trial_*')
trial_paths = sorted(glob.glob(trial_root))

tuning_dict = {}
for trial_path in trial_paths:
    trial_id = trial_path.split('/')[-1]
    json_path = trial_path + '/trial.json'
    with open(json_path) as f:
        trial_data = json.load(f)
    tuning_dict[trial_id] = {}
    hp_values = trial_data['hyperparameters']['values']
    for hp_name, hp_value in hp_values.items():
        tuning_dict[trial_id][hp_name] = hp_value
    tuning_dict[trial_id]['min_val_loss'] = trial_data['metrics']['metrics']['val_mse']['observations'][0]['value'][0]

tuning_df = pd.DataFrame.from_dict(tuning_dict, orient = 'index')
tuning_df.rename_axis('trial_id', inplace=True)
pandas_path = 'RESULTS_' + proj_name + '.pandas.pkl'
# tuning_df.to_pickle(pandas_path)


tuning_df_sorted = tuning_df.sort_values(by=['min_val_loss'])
tuning_df_sorted.reset_index(inplace = True)
tuning_df_sorted.index = pd.RangeIndex(tuning_df_sorted.index.start+1, tuning_df_sorted.index.stop+1)
tuning_df_sorted.rename_axis('rank', inplace=True)
pandas_sorted_path = 'RESULTS_' + proj_name + '_sorted.pandas.pkl'
# tuning_df_sorted.to_pickle(pandas_sorted_path)

fig, ax =  plt.subplots(ncols=1)

tuning_df_sorted['min_val_loss'].plot(ax = ax, label = proj_name)
ax.set_yscale('log')
ax.set_ylabel('validation mean squared error')
ax.set_xlabel('model rank (by mean squared error)')
ax.set_title('Ranked Validation Error for ' + proj_name + ' configuration')
ax.legend()
ax.grid(True, which="both", ls="--")

fig.set_size_inches(8,5)
fig.savefig('ranked_val_error.png', dpi = 300, bbox_inches='tight')

def build_model(hp:dict):
    alpha = hp["leak"]
    dp_rate = hp["dropout"]
    batch_norm = hp["batch_normalization"]
    model = Sequential()
    hiddenUnits = hp['hidden_units']
    model.add(Dense(units = hiddenUnits, input_dim=175, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha = alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dp_rate))
    for i in range(hp["num_layers"]):
        model.add(Dense(units = hiddenUnits, kernel_initializer='normal'))
        model.add(LeakyReLU(alpha = alpha))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dp_rate))
    model.add(Dense(55, kernel_initializer='normal', activation='linear'))
    initial_learning_rate = hp["lr"]
    optimizer = hp["optimizer"]
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate)
    elif optimizer == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate = initial_learning_rate)
    elif optimizer == "RAdam":
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate = initial_learning_rate)
    elif optimizer == "QHAdam":
        optimizer = QHAdamOptimizer(learning_rate = initial_learning_rate, nu2=1.0, beta1=0.995, beta2=0.999)
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ["mse"])
    return model

txtpath = "txt_models/"
parameter_count = []
for i in tqdm(range(len(tuning_df_sorted))):
    convert = ["python", "convert_weights.sungduk.py", "--weights_file"]
    model_rank = i+1
    trial_info = tuning_df_sorted.loc[model_rank].to_dict()
    tuning_dir_prefix = '../training/tuning_directory/%s'%proj_name
    checkpoint = tuning_dir_prefix + '/%s/checkpoint'%trial_info['trial_id']
    model = build_model(trial_info)
    model.load_weights(checkpoint)
    parameter_count.append(model.count_params())
    f_save = "h5_models/" + '%s_model_%03d.h5'%(proj_name, model_rank)
    model.save(f_save)
    txtfile = "_model_%03d.txt"%(model_rank)
    convert = convert + [f_save] + ["--output_file"] + [txtpath + proj_name + txtfile]
    os.system(" ".join(convert))

parameter_count_col = pd.Series(parameter_count, name = 'parameter_count', index = tuning_df_sorted.index)
tuning_df_sorted['parameter_count'] = parameter_count_col
tuning_df_sorted.to_pickle(pandas_sorted_path)

