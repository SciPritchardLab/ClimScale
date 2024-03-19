import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
import tensorflow_addons as tfa
from qhoptim.tf import QHAdamOptimizer
from tensorflow.keras import callbacks
import os
import random
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SLURM_LOCALID']
print('CUDA_VISIBLE_DEVICES: ' + str(os.environ["CUDA_VISIBLE_DEVICES"]))
print('imported packages')

# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# random.seed(hash('setting random seeds') % 2**32 - 1)
# np.random.seed(hash('improves reproducibility') % 2**32 - 1)
# tf.random.set_seed(hash('by removing stochasticity') % 2**32 - 1)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

data_path = '/dev/shm/'
train_input = np.load(data_path + 'train_input.npy', mmap_mode='r')
train_target = np.load(data_path + 'train_target.npy', mmap_mode='r')
val_input = np.load(data_path + 'val_input.npy')
val_target = np.load(data_path + 'val_target.npy')
test_input = np.load(data_path + 'test_input.npy')
test_target = np.load(data_path + 'test_target.npy')

project_name = 'PROJECT_NAME_HERE'

# Define sweep config
wandb.login()

print('logged into wandb')

def build_model(hp:dict):
    alpha = hp["leak"]
    dp_rate = hp["dropout"]
    batch_norm = hp["batch_normalization"]
    model = Sequential()
    hidden_units = hp['hidden_units']
    model.add(Dense(units = hidden_units, input_dim=175, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha = alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dp_rate))
    for i in range(hp["num_layers"]):
        model.add(Dense(units = hidden_units, kernel_initializer='normal'))
        model.add(LeakyReLU(alpha = alpha))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dp_rate))
    model.add(Dense(55, kernel_initializer='normal', activation='linear'))
    initial_learning_rate = hp["learning_rate"]
    optimizer = hp["optimizer"]
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate)
    elif optimizer == "RAdam":
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate = initial_learning_rate)
    elif optimizer == "QHAdam":
        optimizer = QHAdamOptimizer(learning_rate = initial_learning_rate, nu2=1.0, beta1=0.995, beta2=0.999)
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ["mse"])
    return model

assert(train_input.shape[0] == train_target.shape[0])
num_samples = train_input.shape[0]

def data_generator(inputs, targets, batch_size, drop_remainder = True):
    indices = np.arange(len(inputs))
    if drop_remainder:
        end_index = len(inputs)//batch_size*batch_size
    else:
        end_index = len(inputs)
    np.random.shuffle(indices)
    for i in range(0, end_index, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield inputs[batch_indices,:], targets[batch_indices,:]

def main():
    logging.basicConfig(level=logging.INFO)
    run = wandb.init(project=project_name)
    batch_size = wandb.config['batch_size']
    num_epochs = wandb.config['num_epochs']
    with tf.device('/CPU:0'):
        train_ds = tf.data.Dataset.from_generator(
            lambda: data_generator(train_input, train_target, batch_size),
                                output_signature=(tf.TensorSpec(shape=(None, 175), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None, 55), dtype=tf.float32)))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    logging.debug("Data loaded")
    model = build_model(wandb.config)
    model.fit(train_ds, validation_data = (val_input, val_target), epochs = num_epochs,  callbacks = [WandbMetricsLogger(), \
                                                                                                      callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True)])
    offline_test_loss, _ = model.evaluate(test_input, test_target, batch_size = batch_size)
    wandb.log({'offline_test_loss': offline_test_loss})
    model.save('model_directory/' + run.name + '.h5')
    run.finish()

# 3: Start the sweep
wandb.agent("SWEEP_ID_HERE", function=main, project = project_name, count=RUNS_PER_GPU_HERE)