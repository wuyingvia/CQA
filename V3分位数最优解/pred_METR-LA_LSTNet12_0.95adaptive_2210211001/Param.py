# DATANAME = 'PEMSD7M'
# TIMESTEP_IN = 12
# TIMESTEP_OUT = 12 # It can be overrided by param_stgcn3
# N_NODE = 228
# CHANNEL = 1
# BATCHSIZE = 64
# LEARN = 0.001
# EPOCH = 200
# PATIENCE = 10
# OPTIMIZER = 'Adam'
# # OPTIMIZER = 'RMSprop'
# # LOSS = 'MSE'
# LOSS = 'MAE'
# TRAINRATIO = 0.5 # TRAIN + VAL
# CALRATIO = 0.8 # Calibration
# TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
# FLOWPATH = '../PEMSD7M/V_228.csv'
# ADJPATH = '../PEMSD7M/W_228.csv'
#
# import numpy as np
# quantiles_list = [0.1, 0.9]
#
#
# # calibration error quantile rate
# cal_q = 0.005
#
# # for adaptive
# lamda = 0.21

# DATANAME = 'METR-LA'

TIMESTEP_IN = 12
TIMESTEP_OUT = 12
CHANNEL = 1
BATCHSIZE = 64
LEARN = 0.001
EPOCH = 200
PATIENCE = 10
OPTIMIZER = 'Adam'
# OPTIMIZER = 'RMSprop'
# LOSS = 'MSE'
LOSS = 'MAE'
TRAINRATIO = 0.5 # TRAIN + VAL
CALRATIO = 0.8 # Calibration
TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
# FLOWPATH = '../METRLA/metr-la.h5'
# ADJPATH = '../METRLA/W_metrla.csv'

# # # PEMS03
# DATANAME = 'PEMS03'
# N_NODE = 358
# FLOWPATH = '../PEMS03/PEMS03.npz'
# ADJPATH = '../PEMS03/adj_PEMS03.csv'

# # PEMS04, 注意这个有三个纬度【流量，是？， 速度】
# DATANAME = 'PEMS04'
# N_NODE = 307
# FLOWPATH = '../PEMS04/PEMS04.npz'
# ADJPATH = '../PEMS04/adj_PEMS04.csv'

# # PEMS07
# DATANAME = 'PEMS07'
# N_NODE = 883
# FLOWPATH = '../PEMS07/PEMS07.npz'
# ADJPATH = '../PEMS07/adj_PEMS07.csv'

# # PEMS08 注意这个有三个纬度【流量，是？， 速度】
# DATANAME = 'PEMS08'
# N_NODE = 170
# FLOWPATH = '../PEMS08/PEMS08.npz'
# ADJPATH = '../PEMS08/adj_PEMS08.csv'

# METR-LA
DATANAME = 'METR-LA'
N_NODE = 207
FLOWPATH = '../METRLA/metr-la.h5'
ADJPATH = '../METRLA/W_metrla.csv'

quantiles_list = [0.15, 0.95]

# calibration error quantile rate
import numpy as np
cal_list = np.arange(0,21)/20

# for adaptive
lambda_list = np.arange(0,21)/20