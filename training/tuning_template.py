import numpy as np
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
import keras_tuner as kt
import os
import h5py
from training_functions import *

with open('training_data/train_input.npy', 'rb') as f:
    train_input = np.load(f)

with open('training_data/train_target.npy', 'rb') as f:
    train_target = np.load(f)

with open('training_data/val_input.npy', 'rb') as f:
    val_input = np.load(f)

with open('training_data/val_target.npy', 'rb') as f:
    val_target = np.load(f)

set_environment(NUM_GPUS_PER_NODE_HERE)

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_mse",
    max_trials=MAX_TRIALS_HERE,
    executions_per_trial=1,
    overwrite=False,
    directory="tuning_directory/",
    project_name="PROJECT_NAME_HERE",
)

kwargs = {'batch_size': 5000,
          'epochs': 100,
          'verbose': 2,
          'shuffle': True
         }

tuner.search(train_input, train_target, validation_data=(val_input, val_target), **kwargs, \
             callbacks=[callbacks.EarlyStopping('val_loss', patience=5)])


