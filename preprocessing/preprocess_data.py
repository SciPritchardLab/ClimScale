from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_clean/"
data_path_cold = "/ocean/projects/atm200007p/jlin96/longSPrun_clean_cold/"
data_path_warm = "/ocean/projects/atm200007p/jlin96/longSPrun_clean_warm/"
norm_path = "../coupling_folder/norm_files/"

# TRAINING 

print('creating training input')

sp_data_train_input_cold = combine_arrays(make_nn_input(load_data(month = 2, year = 0, data_path = data_path_cold), spacing = 8), \
                                          make_nn_input(load_data(month = 3, year = 0, data_path = data_path_cold), spacing = 8), \
                                          make_nn_input(load_data(month = 4, year = 0, data_path = data_path_cold), spacing = 8), \
                                          make_nn_input(load_data(month = 5, year = 0, data_path = data_path_cold), spacing = 8), \
                                          make_nn_input(load_data(month = 6, year = 0, data_path = data_path_cold), spacing = 8))
reshaped_input_cold = reshape_input(sp_data_train_input_cold)
reshaped_input_cold = reshaped_input_cold[:, :5975723]

sp_data_train_input_standard = combine_arrays(make_nn_input(load_data(month = 2, year = 0, data_path = data_path_standard), spacing = 8), \
                                              make_nn_input(load_data(month = 3, year = 0, data_path = data_path_standard), spacing = 8), \
                                              make_nn_input(load_data(month = 4, year = 0, data_path = data_path_standard), spacing = 8), \
                                              make_nn_input(load_data(month = 5, year = 0, data_path = data_path_standard), spacing = 8), \
                                              make_nn_input(load_data(month = 6, year = 0, data_path = data_path_standard), spacing = 8))
reshaped_input_standard = reshape_input(sp_data_train_input_standard)
reshaped_input_standard = reshaped_input_standard[:, :5975723]

sp_data_train_input_warm = combine_arrays(make_nn_input(load_data(month = 2, year = 0, data_path = data_path_warm), spacing = 8), \
                                          make_nn_input(load_data(month = 3, year = 0, data_path = data_path_warm), spacing = 8), \
                                          make_nn_input(load_data(month = 4, year = 0, data_path = data_path_warm), spacing = 8), \
                                          make_nn_input(load_data(month = 5, year = 0, data_path = data_path_warm), spacing = 8), \
                                          make_nn_input(load_data(month = 6, year = 0, data_path = data_path_warm), spacing = 8))
reshaped_input_warm = reshape_input(sp_data_train_input_warm)
reshaped_input_warm = reshaped_input_warm[:, :5975723]

reshaped_input = np.concatenate((reshaped_input_cold, reshaped_input_standard, reshaped_input_warm), axis = 1)
print(reshaped_input.shape)

del reshaped_input_cold
del reshaped_input_standard
del reshaped_input_warm

normalize_input_train(X_train = reshaped_input, save_files = True)
del reshaped_input

print("finished creating training input")

print("creating training target")

sp_data_train_target_cold = combine_arrays(make_nn_target(load_data(month = 2, year = 0, data_path = data_path_cold), spacing = 8), \
                                           make_nn_target(load_data(month = 3, year = 0, data_path = data_path_cold), spacing = 8), \
                                           make_nn_target(load_data(month = 4, year = 0, data_path = data_path_cold), spacing = 8), \
                                           make_nn_target(load_data(month = 5, year = 0, data_path = data_path_cold), spacing = 8), \
                                           make_nn_target(load_data(month = 6, year = 0, data_path = data_path_cold), spacing = 8))
reshaped_target_cold = reshape_target(sp_data_train_target_cold)
reshaped_target_cold = reshaped_target_cold[:, :5975723]

sp_data_train_target_standard = combine_arrays(make_nn_target(load_data(month = 2, year = 0, data_path = data_path_standard), spacing = 8), \
                                               make_nn_target(load_data(month = 3, year = 0, data_path = data_path_standard), spacing = 8), \
                                               make_nn_target(load_data(month = 4, year = 0, data_path = data_path_standard), spacing = 8), \
                                               make_nn_target(load_data(month = 5, year = 0, data_path = data_path_standard), spacing = 8), \
                                               make_nn_target(load_data(month = 6, year = 0, data_path = data_path_standard), spacing = 8))
reshaped_target_standard = reshape_target(sp_data_train_target_standard)
reshaped_target_standard = reshaped_target_standard[:, :5975723]

sp_data_train_target_warm = combine_arrays(make_nn_target(load_data(month = 2, year = 0, data_path = data_path_warm), spacing = 8), \
                                           make_nn_target(load_data(month = 3, year = 0, data_path = data_path_warm), spacing = 8), \
                                           make_nn_target(load_data(month = 4, year = 0, data_path = data_path_warm), spacing = 8), \
                                           make_nn_target(load_data(month = 5, year = 0, data_path = data_path_warm), spacing = 8), \
                                           make_nn_target(load_data(month = 6, year = 0, data_path = data_path_warm), spacing = 8))
reshaped_target_warm = reshape_target(sp_data_train_target_warm)
reshaped_target_warm = reshaped_target_warm[:, :5975723]

reshaped_target = np.concatenate((reshaped_target_cold, reshaped_target_standard, reshaped_target_warm), axis = 1)
print(reshaped_target.shape)

normalize_target_train(y_train_original = reshaped_target, save_files = True)
del reshaped_target

print("finished creating training target")

# VALIDATION

print("creating validation input")

sp_data_val_input = combine_arrays(make_nn_input(load_data(month = 3, year = 1, data_path = data_path), spacing = 7), \
                                   make_nn_input(load_data(month = 4, year = 1, data_path = data_path), spacing = 7))
normalize_input_val(X_val = reshape_input(sp_data_val_input), save_files = True)
del sp_data_val_input

print("finished creating validation input")

print("creating validation target")

sp_data_val_target = combine_arrays(make_nn_target(load_data(month = 3, year = 1, data_path = data_path), spacing = 7), \
                                    make_nn_target(load_data(month = 4, year = 1, data_path = data_path), spacing = 7))
normalize_target_val(y_val_original = reshape_target(sp_data_val_target), save_files = True)
del sp_data_val_target

print("finished creating validation target")

# TEST

offline_save_path = "../offline_evaluation/testing_data/"
offline_timesteps = 336

print("creating test input")

sp_data_test_input = make_nn_input(load_data(month = 9, year = 1, data_path = data_path), subsample = False)[0:offline_timesteps,:,:,:]
np.save(offline_save_path + "test_input.npy", sp_data_test_input)
del sp_data_test_input

print("finished creating test input")

print("creating test target")

sp_data_test_target = make_nn_test_target(load_data(month = 9, year = 1, data_path = data_path), subsample = False)[0:offline_timesteps,:,:,:]
np.save(offline_save_path + "test_target.npy", sp_data_test_target)
del sp_data_test_target

print("finished creating test target")

# TEST EXTENSION

offline_save_path_2x = "../offline_evaluation/testing_data_2x/"
offline_save_path_4x = "../offline_evaluation/testing_data_4x/"
offline_save_path_multi = "../offline_evaluation/testing_data_multi/"
offline_timesteps_2x = 672
offline_timesteps_4x = 1344
offline_timesteps_multi = 112

print("creating test input extended")

sp_data_test_input = make_nn_input(load_data(month = 9, year = 1, data_path = data_path), subsample = False)

sp_data_test_input_2x = sp_data_test_input[0:offline_timesteps_2x,:,:,:]
print(sp_data_test_input_2x.shape)
np.save(offline_save_path_2x + "test_input_2x.npy", sp_data_test_input_2x)
del sp_data_test_input_2x

sp_data_test_input_4x = sp_data_test_input[0:offline_timesteps_4x,:,:,:]
print(sp_data_test_input_4x.shape)
np.save(offline_save_path_4x + "test_input_4x.npy", sp_data_test_input_4x)
del sp_data_test_input_4x

sp_data_test_input_multi = combine_arrays(make_nn_input(load_data(month = 7, year = 0, data_path = data_path_cold), subsample = False)[0:offline_timesteps_multi,:,:,:], \
                                          make_nn_input(load_data(month = 9, year = 1, data_path = data_path), subsample = False)[0:offline_timesteps_multi,:,:,:], \
                                          make_nn_input(load_data(month = 7, year = 0, data_path = data_path_warm), subsample = False)[0:offline_timesteps_multi,:,:,:], contiguous = False)
print(sp_data_test_input_multi.shape)
np.save(offline_save_path_multi + "test_input_multi.npy", sp_data_test_input_multi)
del sp_data_test_input_multi

print("finished creating test input extended")

print("creating test target extended")

sp_data_test_target = make_nn_test_target(load_data(month = 9, year = 1, data_path = data_path), subsample = False)

sp_data_test_target_2x = sp_data_test_target[0:offline_timesteps_2x,:,:,:]
print(sp_data_test_target_2x.shape)
np.save(offline_save_path_2x + "test_target_2x.npy", sp_data_test_target_2x)
del sp_data_test_target_2x

sp_data_test_target_4x = sp_data_test_target[0:offline_timesteps_4x,:,:,:]
print(sp_data_test_target_4x.shape)
np.save(offline_save_path_4x + "test_target_4x.npy", sp_data_test_target_4x)
del sp_data_test_target_4x

sp_data_test_target_multi = combine_arrays(make_nn_test_target(load_data(month = 7, year = 0, data_path = data_path_cold), subsample = False)[0:offline_timesteps_multi,:,:,:], \
                                           make_nn_test_target(load_data(month = 9, year = 1, data_path = data_path), subsample = False)[0:offline_timesteps_multi,:,:,:], \
                                           make_nn_test_target(load_data(month = 7, year = 0, data_path = data_path_warm), subsample = False)[0:offline_timesteps_multi,:,:,:], contiguous = False)
print(sp_data_test_target_multi.shape)
np.save(offline_save_path_multi + "test_target_multi.npy", sp_data_test_target_multi)
del sp_data_test_target_multi

print("finished creating test target extended")