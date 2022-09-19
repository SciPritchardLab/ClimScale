import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import csv
import yaml
import sys
import re
import fileinput

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

#This python script excepts the family name as input
family = sys.argv[1]

def build_model(hp:dict):
    alpha = hp["leak"]
    dp_rate = hp["dropout"]
    model = Sequential()
    hiddenUnits = hp['hidden_units']
    model.add(Dense(units = hiddenUnits, input_dim=64, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha = alpha))
    if hp["batch_normalization"]:
        model.add(BatchNormalization())
    model.add(Dropout(dp_rate))
    for i in range(hp["num_layers"]):
        model.add(Dense(units = hiddenUnits, kernel_initializer='normal'))
        model.add(LeakyReLU(alpha = alpha))
        if hp["batch_normalization"]:
            model.add(BatchNormalization())
        model.add(Dropout(dp_rate))
    model.add(Dense(60, kernel_initializer='normal', activation='linear'))
    initial_learning_rate = hp["lr"]
    optimizer = hp["optimizer"]
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate)
    elif optimizer == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate = initial_learning_rate)
    elif optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate = initial_learning_rate)
    elif optimizer == "SGD_momentum":
        optimizer = keras.optimizers.SGD(learning_rate = initial_learning_rate, momentum = .9)
    elif optimizer == "SGD_nesterov":
        optimizer = keras.optimizers.SGD(learning_rate = initial_learning_rate, momentum = .9, nesterov = True)
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ["mse"])
    return model

def maketxt(family):
    pklPath = "RESULTS_big_" + family + "Humidity.pandas.pkl"
    pd_ranked = pd.read_pickle(pklPath)
    txtpath = "/ocean/projects/atm200007p/jlin96/prognosticTesting/onlineCoupling/" + family + "/"
    for i in range(len(pd_ranked)):
        convert = ["python", "convert_weights.py", "--weights_file"]
        model_rank = i+1
        trial_info = pd_ranked.loc[model_rank].to_dict()
        tuning_dir_prefix = '/ocean/projects/atm200007p/jlin96/tuning/tools/tuningDirectory/%sHumidity'%family
        dir_weights = tuning_dir_prefix + '/%s/checkpoints/epoch_0/checkpoint'%trial_info['trial_id']
        model = build_model(trial_info)
        model.load_weights(dir_weights)
        f_save = '%s_model_rank-%04d.h5'%(family, model_rank)
        model.save(f_save)
        txtfile = "Model%04d.txt"%(model_rank)
        convert = convert + [f_save] + ["--output_file"] + [txtpath + family + txtfile]
        os.system(" ".join(convert))
        os.system(" ".join(["rm", f_save]))


maketxt(family)

