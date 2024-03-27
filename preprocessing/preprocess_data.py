from preprocessing_functions import *

data_path = "/ocean/projects/atm200007p/jlin96/longSPrun_clean/"
norm_path = "../coupling_folder/norm_files/"

# TRAINING 

print('creating training input')

sp_data_train_input = combine_arrays(make_nn_input(load_data(month = 2, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 3, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 4, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 5, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 6, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 7, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 8, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 9, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 10, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 11, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 12, year = 0, data_path = data_path), spacing = 8), \
                                     make_nn_input(load_data(month = 1, year = 1, data_path = data_path), spacing = 8))
normalize_input_train(X_train = reshape_input(sp_data_train_input), save_files = True)
del sp_data_train_input

print("finished creating training input")

print("creating training target")

sp_data_train_target = combine_arrays(make_nn_target(load_data(month = 2, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 3, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 4, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 5, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 6, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 7, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 8, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 9, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 10, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 11, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 12, year = 0, data_path = data_path), spacing = 8), \
                                      make_nn_target(load_data(month = 1, year = 1, data_path = data_path), spacing = 8))
normalize_target_train(y_train_original = reshape_target(sp_data_train_target), save_files = True)
del sp_data_train_target

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

print("creating test input")

sp_data_test_input = combine_arrays(make_nn_input(load_data(month = 6, year = 1, data_path = data_path), spacing = 11), \
                                   make_nn_input(load_data(month = 7, year = 1, data_path = data_path), spacing = 11))
normalize_input_test(X_test = reshape_input(sp_data_test_input), save_files = True)
del sp_data_test_input

print("finished creating test input")

print("creating test target")

sp_data_test_target = combine_arrays(make_nn_target(load_data(month = 6, year = 1, data_path = data_path), spacing = 11), \
                                    make_nn_target(load_data(month = 7, year = 1, data_path = data_path), spacing = 11))
normalize_target_test(y_test_original = reshape_target(sp_data_test_target), save_files = True)
del sp_data_test_target

print("finished creating test target")

