from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_clean/"
norm_path = "../coupling_folder/norm_files/"

# TRAINING 

print('creating training input')

sp_data_train_input = combine_arrays(make_nn_input(load_data(month = 2, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 3, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 4, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 5, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 6, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 7, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 8, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 9, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 10, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 11, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 12, year = 0, data_path = data_path)), \
                                     make_nn_input(load_data(month = 1, year = 1, data_path = data_path)))
normalize_input_train(X_train = reshape_input(sp_data_train_input), save_files = True)
del sp_data_train_input

print("finished creating training input")

print("creating training target")

sp_data_train_target = combine_arrays(make_nn_target(load_data(month = 2, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 3, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 4, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 5, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 6, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 7, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 8, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 9, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 10, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 11, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 12, year = 0, data_path = data_path)), \
                                      make_nn_target(load_data(month = 1, year = 1, data_path = data_path)))
normalize_target_train(y_train_original = reshape_target(sp_data_train_target), save_files = True)
del sp_data_train_target

print("finished creating training target")

inp_sub = np.loadtxt(norm_path + "inp_sub.txt")[:, np.newaxis]
inp_div = np.loadtxt(norm_path + "inp_div.txt")[:, np.newaxis]
print(inp_sub.shape)
print(inp_div.shape)
print("loaded in inp_sub and inp_div")

# VALIDATION

print("creating validation input")

sp_data_val_input = combine_arrays(make_nn_input(load_data(month = 9, year = 1, data_path = data_path)), \
                                   make_nn_input(load_data(month = 10, year = 1, data_path = data_path)))
normalize_input_val(X_val = reshape_input(sp_data_val_input), inp_sub = inp_sub, inp_div = inp_div, save_files = True)
del sp_data_val_input

print("finished creating validation input")

print("creating validation target")

sp_data_val_target = combine_arrays(make_nn_target(load_data(month = 9, year = 1, data_path = data_path)), \
                                    make_nn_target(load_data(month = 10, year = 1, data_path = data_path)))
normalize_target_val(y_val_original = reshape_target(sp_data_val_target), save_files = True)
del sp_data_val_target

print("finished creating validation target")

# TEST

offline_save_path = "../offlinetesteval/testing_data/"
offline_timesteps = 336

print("creating test input")

sp_data_test_input = make_nn_input(load_data(month = 11, year = 1, data_path = data_path), subsample = False)[0:offline_timesteps,:,:,:]
np.save(offline_save_path + "test_input.npy", sp_data_test_input)
del sp_data_test_input

print("finished creating test input")

print("creating test target")

sp_data_test_target = make_nn_target(load_data(month = 11, year = 1, data_path = data_path), subsample = False)[0:offline_timesteps,:,:,:]
np.save(offline_save_path + "test_target.npy", sp_data_test_target)
del sp_data_test_target

print("finished creating test target")

# create offline error weights

print("creating offline error weights")

sp_data = load_data(month = 9, year = 1, data_path = data_path)

# AREA WEIGHTING

lats = np.array(sp_data["lat"])
assert(90+lats[0]==90-lats[63])
last_lat_mdiff = 90+lats[0]
lat_mdiff = np.diff(lats)/2
lat_buff = np.append(lat_mdiff, last_lat_mdiff)
lat_edges = lat_buff + lats
lat_edges = np.append(-90, lat_edges)
area_weights = np.diff(np.sin(lat_edges*np.pi/180))[np.newaxis, np.newaxis,:,np.newaxis]

# PRESSURE WEIGHTING

pressure_grid_p1 = (sp_data['P0'].values[:, np.newaxis] * sp_data['hyai'].values)[:,:,np.newaxis, np.newaxis]
pressure_grid_p2 = sp_data['hybi'].values[:,:,np.newaxis,np.newaxis]*sp_data['NNPS'].values[:,np.newaxis,:,:]
pressure_grid = pressure_grid_p1 + pressure_grid_p2
dp = pressure_grid[:,1:31,:,:] - pressure_grid[:,0:30,:,:]
dp = dp[0:offline_timesteps,:,:,:] # save only the first week

# ERROR WEIGHTS

offline_error_weights = dp*area_weights

offline_error_weights_strato = offline_error_weights
offline_error_weights = offline_error_weights_strato[:,12:30,:,:]

offline_error_weights_strato = offline_error_weights_strato/np.sum(offline_error_weights_strato)
offline_error_weights = offline_error_weights/np.sum(offline_error_weights)
print(offline_error_weights_strato.shape)
print(offline_error_weights.shape)

np.save(offline_save_path + "offline_error_weights_strato.npy", np.float32(offline_error_weights_strato))
np.save(offline_save_path + "offline_error_weights.npy", np.float32(offline_error_weights))

