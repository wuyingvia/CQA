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
TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
CALRATIO = 0.8
CALSPLIT = 0.5
# calibration error quantile rate
import numpy as np
cal_list = np.arange(0,21)/20