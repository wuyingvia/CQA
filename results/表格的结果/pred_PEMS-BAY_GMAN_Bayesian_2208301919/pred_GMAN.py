from operator import ge
import sys
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time

import Metrics
from GMAN import *
from Param import *
from Param_GMAN import *
import Utils
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS) # [num_samples, timestep, nodes]
    return XS, YS


# generate TE: temporal embedding 
# np.ndarray: [batch_size, TIMESTEP_IN+TIMESTEPOUT, 2]
def getTE(df, mode):
    # TE_DIM = 2: [dayofweek, timeofday].
    # data: numpy, data_time: numpy from getTimestamp 

    time = pd.DatetimeIndex(df.index)
    # torch.tensor: (34272, 1)     Value: 0-6, Monday=0, Sunday=6
    dayofweek = np.reshape(np.array(time.weekday), (-1, 1))
    timeofday = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    timeofday = np.reshape(timeofday, (-1, 1))
    time = np.concatenate((dayofweek, timeofday), -1)
    TRAIN_NUM = int(time.shape[0] * TRAINRATIO)
    TE = []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            t = time[i:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            TE.append(t)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  time.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            t = time[i:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            TE.append(t)
    TE = np.array(TE)
    return TE


# generate SE: spatial embedding
# torch.tensor: [nodes, SEdims] 
def getSE(SE_file):
    with open(SE_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    return SE


def getModel(name, device):
    SE = getSE(SEPATH).to(device=device)
    model = GMAN(prior_mu=prior_mu, prior_sigma=prior_sigma,SE=SE, timestep_in=TIMESTEP_IN, device=device).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, te, y in data_iter:
            y_pred = model(x, te)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, te, y in data_iter:
            YS_pred_batch = model(x, te)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred


def trainModel(name, mode, XS, TE, YS, device, scaler):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name, device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    TE_torch = torch.Tensor(TE).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('TE_torch.shape:  ', TE_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    
    min_val_loss = np.inf
    wait = 0

    print('LOSS is :',LOSS)
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    kl_loss = torchbnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl = kl_loss(model)
    for epoch in range(EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, te, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x, te)
            #loss = criterion(y_pred, y)
            loss = criterion(y_pred, y)
            cost = loss + kl_weight * kl
            cost.backward(retain_graph=True)

            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
            del x, te, y, y_pred, loss
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
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())


# def testModel(name, mode, XS, TE, YS, device):
#     if LOSS == "MaskMAE":
#         criterion = Utils.masked_mae
#     if LOSS == 'MSE':
#         criterion = nn.MSELoss()
#     if LOSS == 'MAE':
#         criterion = nn.L1Loss()
#     print('Model Testing Started ...', time.ctime())
#     print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
#     XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
#     TE_torch = torch.Tensor(TE).to(device)
#     test_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
#     test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
#     model = getModel(name, device)
#     model.load_state_dict(torch.load(PATH+ '/' + name + '.pt'))
#
#     torch_score = evaluateModel(model, criterion, test_iter)
#     YS_pred = predictModel(model, test_iter)
#     print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
#     YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
#     print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
#     np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
#     np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
#     MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
#     print('*' * 40)
#     print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
#     f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
#     f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
#     print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
#     f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
#     for i in range(TIMESTEP_OUT):
#         MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
#         print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
#         f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
#     f.close()
#     print('Model Testing Ended ...', time.ctime())


def testunmodel(name, mode, XS, TE, YS, device, sample, scaler):
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    TE_torch = torch.Tensor(TE).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name, device)
    model.load_state_dict(torch.load(PATH+ '/' + name + '.pt'))

    YS_pred = []
    for i in range(sample):
        YS_pred_ = np.squeeze(predictModel(model, test_iter))
        YS_pred_ = Utils.inverse_transform(YS_pred_,['mean'], scaler['std'])
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
MODELNAME = 'GMAN'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + 'Bayesian' + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
import os
cpu_num = 12
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
########################################################### 
GPU = '0'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################
# df = pd.read_hdf(FLOWPATH)
# data = df.values
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# print('data.shape', data.shape)     # [timestamp, sensors]
##############################################################
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('GMAN.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_GMAN.py', PATH)

    df = pd.read_hdf(FLOWPATH)
    data = df.values
    print('data.shape', data.shape)

    trainx, trainy = getXSYS(data, 'TRAIN')
    # transform
    mean = trainx.mean()
    std = trainy.std()
    scaler = {'mean': mean, 'std': std}
    data = Utils.transform(data, scaler['mean'], scaler['std'])

    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS.shape', trainXS.shape, trainYS.shape)
    trainTE = getTE(df, "TRAIN")
    print('TRAIN TE.shape', trainTE.shape)
    trainModel(MODELNAME, "train", trainXS, trainTE, trainYS, device, scaler )

    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape YS.shape', testXS.shape, testYS.shape)
    testTE = getTE(df, "TEST")
    print('TEST TE.shape', testTE.shape)
    testunmodel(MODELNAME, "test", testXS, testTE, testYS, device, scaler)

if __name__ == '__main__':
    main()


