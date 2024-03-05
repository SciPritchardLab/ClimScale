import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.activations import swish
from tensorflow.keras import optimizers
import keras_tuner
import tensorflow_addons as tfa
from qhoptim.tf import QHAdamOptimizer
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras import layers, callbacks
import os

train_input = np.load('training_data/train_input.npy')
train_target = np.load('training_data/train_target.npy')
val_input = np.load('training_data/val_input.npy')
val_target = np.load('training_data/val_target.npy')

class wandb_tuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        project_name = 'kitchensinkmse'
        hp = trial.hyperparameters
        model = self.hypermodel.build(trial.hyperparameters)
        run = wandb.init(entity='cbrain', project = project_name, config = hp.values, name = project_name + "_" + str(trial.trial_id).zfill(3))
        self.hypermodel.fit(hp, model, callbacks=[WandbMetricsLogger(), WandbModelCheckpoint("tuning_directory/wandb")], *args, **kwargs)
        run.finish()

def build_model(hp):
    dp_rate = hp.Float("dropout", min_value = 0, max_value = .15)
    batch_norm = hp.Boolean("batch_normalization")
    model = Sequential()
    hidden_units = hp.Int("hidden_units", min_value = 200, max_value = 480)
    model.add(Dense(units = hidden_units, input_dim=175, kernel_initializer='normal', activation = 'swish'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dp_rate))
    for i in range(hp.Int("num_layers", min_value = 4, max_value = 11)):
        model.add(Dense(units = hidden_units, kernel_initializer='normal', activation = 'swish'))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dp_rate))
    model.add(Dense(55, kernel_initializer='normal', activation='linear'))
    initial_learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", ["adam", "RAdam", "QHAdam"])
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate)
    elif optimizer == "RAdam":
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate = initial_learning_rate)
    elif optimizer == "QHAdam":
        optimizer = QHAdamOptimizer(learning_rate = initial_learning_rate, nu2=1.0, beta1=0.995, beta2=0.999)
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ["mse"])
    return model

def set_environment(num_gpus_per_node=4, oracle_port = '8000'):
    num_gpus_per_node = str(num_gpus_per_node)
    nodename = os.environ['SLURMD_NODENAME']
    procid = os.environ['SLURM_LOCALID']
    print(nodename)
    print(procid)
    stream = os.popen('scontrol show hostname $SLURM_NODELIST')
    output = stream.read()
    oracle = output.split("\n")[0]
    print(oracle)
    if procid==num_gpus_per_node:
        os.environ["KERASTUNER_TUNER_ID"] = "chief"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["KERASTUNER_TUNER_ID"] = "tuner-" + str(nodename) + "-" + str(procid)
        os.environ["CUDA_VISIBLE_DEVICES"] = procid
    os.environ["KERASTUNER_ORACLE_IP"] = oracle + ".ib.bridges2.psc.edu" # Use full hostname
    os.environ["KERASTUNER_ORACLE_PORT"] = oracle_port
    print("KERASTUNER_TUNER_ID:    %s"%os.environ["KERASTUNER_TUNER_ID"])
    print("KERASTUNER_ORACLE_IP:   %s"%os.environ["KERASTUNER_ORACLE_IP"])
    print("KERASTUNER_ORACLE_PORT: %s"%os.environ["KERASTUNER_ORACLE_PORT"])

set_environment(num_gpus_per_node = NUM_GPUS_PER_NODE_HERE, oracle_port = '8000')

tuner = wandb_tuner(
    hypermodel = build_model,
    objective = 'val_mse',
    max_trials = MAX_TRIALS_HERE,
    executions_per_trial = 1,
    overwrite = False,
    directory = "tuning_directory/",
    project_name = "PROJECT_NAME_HERE",
)

kwargs = {'batch_size': 2500,
          'epochs': 200,
          'verbose': 2,
          'shuffle': True
         }

tuner.search(train_input, train_target, validation_data=(val_input, val_target), **kwargs)