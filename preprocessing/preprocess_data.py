from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_clean/"
norm_path = "../coupling_folder/norm_files/"

# TRAINING 

print('creating training input')

sp_data_train_input = combine_arrays(make_nn_input(load_data(month = 2, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 3, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 4, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 5, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 6, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 7, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 8, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 9, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 10, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 11, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 12, year = 0, data_path = data_path), spacing = 2), \
                                     make_nn_input(load_data(month = 1, year = 1, data_path = data_path), spacing = 2))
normalize_input_train(X_train = reshape_input(sp_data_train_input), save_files = True)
del sp_data_train_input

print("finished creating training input")

print("creating training target")

sp_data_train_target = combine_arrays(make_nn_target(load_data(month = 2, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 3, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 4, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 5, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 6, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 7, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 8, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 9, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 10, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 11, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 12, year = 0, data_path = data_path), spacing = 2), \
                                      make_nn_target(load_data(month = 1, year = 1, data_path = data_path), spacing = 2))
normalize_target_train(y_train_original = reshape_target(sp_data_train_target), save_files = True)
del sp_data_train_target

print("finished creating training target")

# VALIDATION

print("creating validation input")

sp_data_val_input = combine_arrays(make_nn_input(load_data(month = 7, year = 1, data_path = data_path)), \
                                   make_nn_input(load_data(month = 8, year = 1, data_path = data_path)))
normalize_input_val(X_val = reshape_input(sp_data_val_input), save_files = True)
del sp_data_val_input

print("finished creating validation input")

print("creating validation target")

sp_data_val_target = combine_arrays(make_nn_target(load_data(month = 7, year = 1, data_path = data_path)), \
                                    make_nn_target(load_data(month = 8, year = 1, data_path = data_path)))
normalize_target_val(y_val_original = reshape_target(sp_data_val_target), save_files = True)
del sp_data_val_target

print("finished creating validation target")

# TEST

offline_save_path = "../offlinetesteval/testing_data/"
offline_timesteps = 336

print("creating test input")

sp_data_test_input = make_nn_input(load_data(month = 9, year = 1, data_path = data_path), subsample = False)[0:offline_timesteps,:,:,:]
np.save(offline_save_path + "test_input.npy", sp_data_test_input)
del sp_data_test_input

print("finished creating test input")

print("creating test target")

sp_data_test_target = make_nn_target(load_data(month = 9, year = 1, data_path = data_path), subsample = False)[0:offline_timesteps,:,:,:]
np.save(offline_save_path + "test_target.npy", sp_data_test_target)
del sp_data_test_target

print("finished creating test target")

# create offline error weights

print("creating offline error weights")

sp_data = load_data(month = 9, year = 1, data_path = data_path)

offline_error_weights = sp_data['gw'] * (sp_data["P0"] * sp_data["hyai"] + sp_data['hybi']*sp_data['NNPSBSP']).diff(dim = "ilev")
offline_error_weights = offline_error_weights[0:offline_timesteps,:,:,:]
offline_error_weights_tropo = offline_error_weights[:,12:30,:,:]
offline_error_weights_ablated = offline_error_weights[:,5:30,:,:]

offline_error_weights = offline_error_weights/np.sum(offline_error_weights)
offline_error_weight_tropo = offline_error_weights_tropo/np.sum(offline_error_weights_tropo)
offline_error_weights_ablated = offline_error_weights_ablated/np.sum(offline_error_weights_ablated)
print(offline_error_weights.shape)
print(offline_error_weights_tropo.shape)
print(offline_error_weights_ablated.shape)

np.save(offline_save_path + "offline_error_weights.npy", np.float32(offline_error_weights))
np.save(offline_save_path + "offline_error_weights_tropo.npy", np.float32(offline_error_weights_tropo))
np.save(offline_save_path + "offline_error_weights_ablated.npy", np.float32(offline_error_weights_ablated))

print("finished creating offline error weights")