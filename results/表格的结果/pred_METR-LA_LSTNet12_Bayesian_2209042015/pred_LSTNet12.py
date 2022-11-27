import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from torchsummary import summary
import Metrics
import Utils
from LSTNet import *
from Param import *
from Param_LSTNet12 import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def getXSYS_single(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN+TIMESTEP_OUT-1:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN+TIMESTEP_OUT-1:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    YS = np.squeeze(YS)
    return XS, YS

def getModel(name):
    model = LSTNet(prior_mu=prior_mu, prior_sigma=prior_sigma,
                   data_m=N_NODE*CHANNEL,
                 window=TIMESTEP_IN,
                 hidRNN=64,
                 hidCNN=64,
                 CNN_kernel=3,
                 skip=3,
                 highway_window=24).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS, scaler):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    #summary(model, (TIMESTEP_IN, N_NODE * CHANNEL), device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    print('LOSS is :',LOSS)
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    
    min_val_loss = np.inf
    wait = 0
    kl_loss = torchbnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl = kl_loss(model)
    for epoch in range(EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            # loss.backward()
            cost = loss + kl_weight * kl
            cost.backward(retain_graph=True)
            optimizer.step()
            #loss_sum += loss.item() * y.shape[0]
            loss_sum += cost.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = Utils.inverse_transform(np.squeeze(YS),scaler['mean'], scaler['std']), \
                  Utils.inverse_transform(np.squeeze(YS_pred),scaler['mean'], scaler['std'])
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
# def testModel(name, mode, XS, YS):
#     print('Model Testing Started ...', time.ctime())
#     print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
#     XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
#     test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
#     test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
#     model = getModel(name)
#     model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
#     print('LOSS is :',LOSS)
#     if LOSS == 'MSE':
#         criterion = nn.MSELoss()
#     if LOSS == 'MAE':
#         criterion = nn.L1Loss()
#     torch_score = evaluateModel(model, criterion, test_iter)
#     YS_pred = predictModel(model, test_iter)
#     print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
#     YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred)
#     YS = scaler.inverse_transform(YS)
#     YS_pred = scaler.inverse_transform(YS_pred)
#     print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
#     np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
#     np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
#     MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
#     with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
#         f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
#         f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
#     print('*' * 40)
#     print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
#     print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
#     print('Model Testing Ended ...', time.ctime())
#

def testModel(name, mode, XS, YS, sample, scaler):
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    print('LOSS is :',LOSS)
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    YS_pred = []
    for i in range(sample):
        YS_pred_ = np.squeeze(predictModel(model, test_iter))
        YS_pred_ = Utils.inverse_transform(YS_pred_,scaler['mean'], scaler['std'])
        YS_pred.append(np.expand_dims(YS_pred_,axis=0))
    YS_pred = np.vstack(YS_pred)
    y_l_pred = np.min(YS_pred, axis=0)
    y_u_pred = np.max(YS_pred, axis=0)

    YS = np.squeeze(YS)
    YS = Utils.inverse_transform(YS,scaler['mean'], scaler['std'])

    # compute the coverage and interval width
    results = {}
    results["Point predictions"] = np.array(YS_pred)
    results["Upper limit"] = np.array(y_l_pred)
    results["Lower limit"] = np.array(y_u_pred)
    results["Confidence interval widths"] = y_u_pred - y_l_pred
    results["Mean confidence interval widths"] = np.mean(y_u_pred - y_l_pred)
    independent_coverage = np.logical_and(np.logical_and(y_u_pred >= YS, y_l_pred <= YS), np.logical_and(y_l_pred > 0, y_u_pred > 0, YS > 0))

    results["Independent coverage indicators"] = independent_coverage
    results["Mean independent coverage"] = np.sum(independent_coverage) / np.sum(np.logical_and(y_l_pred > 0, y_u_pred > 0, YS > 0))

    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
                %(results["Mean independent coverage"], results["Mean confidence interval widths"]))
    print("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
                %(results["Mean independent coverage"], results["Mean confidence interval widths"]))

################# Parameter Setting #######################
MODELNAME = 'LSTNet12'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + 'Bayesian' + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
# torch.backends.cudnn.deterministic = True
########################################################### 
# GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
# device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")

GPU = '0'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################
# data = pd.read_hdf(FLOWPATH).values
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# print('data.shape', data.shape)
###########################################################
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('LSTNet.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_LSTNet12.py', PATH)

    data = pd.read_hdf(FLOWPATH).values
    print('data.shape', data.shape)

    trainx, trainy = getXSYS_single(data, 'TRAIN')
    # transform
    mean = trainx.mean()
    std = trainy.std()
    scaler = {'mean': mean, 'std': std}
    data = Utils.transform(data, scaler['mean'], scaler['std'])

    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS_single(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS, scaler)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS_single(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS, sample, scaler)

    
if __name__ == '__main__':
    main()

