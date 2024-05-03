import numpy as np
import pandas as pd
import pickle
import sys

config_subdir = sys.argv[1]
config_name = config_subdir

offline_test_error = np.load('../offline_evaluation/offline_test_error/rmse.npy')

with open('prognostic_T.pkl', 'rb') as f:
    prognostic_T = pickle.load(f)

with open('prognostic_Q.pkl', 'rb') as f:
    prognostic_Q = pickle.load(f)

model_info = pd.read_pickle('RESULTS_' + config_subdir + '_sorted.pandas.pkl')
model_info['offline_heating'] = pd.Series(offline_test_error[:,0], name = 'offline_heating', index = model_info.index)
model_info['offline_moistening'] = pd.Series(offline_test_error[:,1], name = 'offline_moistening', index = model_info.index)
model_info['num_months'] = pd.Series([len(prognostic_T[x]) for x in model_info.index], name = 'num_months', index = model_info.index)
assert model_info['num_months'].equals(pd.Series([len(prognostic_Q[x]) for x in model_info.index], name = 'num_months', index = model_info.index))
model_info['online_temperature'] = pd.Series([np.mean(prognostic_T[x]) if len(prognostic_T[x])==12 else None for x in model_info.index], name = 'prognostic_T', index = model_info.index)
model_info['online_moisture'] = pd.Series([np.mean(prognostic_Q[x])*1000 if len(prognostic_Q[x])==12 else None for x in model_info.index], name = 'prognostic_Q', index = model_info.index)

model_info.to_pickle(config_subdir + '_df.pandas.pkl')