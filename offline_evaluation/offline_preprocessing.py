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
