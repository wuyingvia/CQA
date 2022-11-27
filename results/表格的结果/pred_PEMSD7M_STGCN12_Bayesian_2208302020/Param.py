DATANAME = 'PEMSD7M'
TIMESTEP_IN = 12
TIMESTEP_OUT = 12 # It can be overrided by param_stgcn3
N_NODE = 228
CHANNEL = 1
BATCHSIZE = 64
LEARN = 0.001
EPOCH = 200
PATIENCE = 10
OPTIMIZER = 'Adam'
# OPTIMIZER = 'RMSprop'
# LOSS = 'MSE'
LOSS = 'MAE'
TRAINRATIO = 0.8 # TRAIN + VAL
TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
FLOWPATH = '../PEMSD7M/V_228.csv'
ADJPATH = '../PEMSD7M/W_228.csv'

#Quantile factor
#quantiles_list = [0.2, 0.8]

import numpy as np
quantiles_list = np.arange(0,1,0.1)

# for bayesian
prior_mu = 0.0
prior_sigma = 0.1
kl_weight = 0.05

# for final evalute uncertainty qualification: coverage and interval widths
# for bayesian for final sample
sample = 50