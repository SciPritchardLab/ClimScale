from training_functions import *

set_environment(NUM_GPUS_PER_NODE_HERE)

memory_map = True
num_epochs = 200
batch_size = 5000
shuffle_buffer = 20000
patience = 20

if memory_map:
    train_input = np.load('/dev/shm/train_input.npy', mmap_mode='r')
    train_target = np.load('/dev/shm/train_target.npy', mmap_mode='r')
    val_input = np.load('/dev/shm/val_input.npy', mmap_mode='r')
    val_target = np.load('/dev/shm/val_target.npy', mmap_mode='r')
    with tf.device('/CPU:0'):
        train_ds = tf.data.Dataset.from_tensor_slices((train_input, train_target))
        val_ds = tf.data.Dataset.from_tensor_slices((val_input, val_target))

        # Applying transformations to the dataset:
        # Shuffle, batch, and prefetch for the training dataset
        train_ds = train_ds.shuffle(buffer_size=shuffle_buffer) # Shuffle the data
        train_ds = train_ds.batch(batch_size, drop_remainder=True)  # Batch the data
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)  # Prefetch for efficiency

        # Batch and prefetch for the validation dataset
        val_ds = val_ds.batch(batch_size)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
else:
    train_input = np.load('/dev/shm/train_input.npy')
    train_target = np.load('/dev/shm/train_target.npy')
    val_input = np.load('/dev/shm/val_input.npy')
    val_target = np.load('/dev/shm/val_target.npy')

lr_scheduler = LearningRateScheduler(lr_schedule)

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_mse",
    max_trials=MAX_TRIALS_HERE,
    executions_per_trial=1,
    overwrite=False,
    directory="tuning_directory/",
    project_name="PROJECT_NAME_HERE",
)

kwargs = {'epochs': num_epochs,
          'verbose': 2,
          'shuffle': True
         }

if memory_map:
    tuner.search(train_ds, validation_data=val_ds, **kwargs, \
                callbacks=[lr_scheduler, callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)])
else:
    tuner.search(train_input, train_target, validation_data=(val_input, val_target), **kwargs, \
                callbacks=[lr_scheduler, callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)])