import sys
import os
import shutil

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import Metrics
from STGCN import *
from Param import *
from Param_STGCN12 import *
import Utils
torch.set_num_threads(1)
import torchbnn as bnn

def getXSYS_single(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    CAL_NUM = int(data.shape[0] * CALRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN+TIMESTEP_OUT-1:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'CAL':
        for i in range(TRAIN_NUM - TIMESTEP_IN, CAL_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1 ):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN+TIMESTEP_OUT-1:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(CAL_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN+TIMESTEP_OUT-1:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, np.newaxis, :, :], YS[:, np.newaxis, :]
    return XS, YS

def getModel(name):
    ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0

    if DATANAME == 'PEMS04':
        W = pd.read_csv(ADJPATH).values
    elif DATANAME == 'PEMS07':
        W = pd.read_csv(ADJPATH).values
    elif DATANAME == 'PEMS08':
        W = pd.read_csv(ADJPATH).values
    elif DATANAME == 'PEMS03':
        W = pd.read_csv(ADJPATH).values
    elif DATANAME == 'METR-LA':
        A = pd.read_csv(ADJPATH).values
        W = weight_matrix(A)
    elif DATANAME == 'PEMS-BAY':
        A = pd.read_csv(ADJPATH).values
        W = weight_matrix(A)
    elif DATANAME == 'PEMSD7M':
        A = pd.read_csv(ADJPATH).values
        W = weight_matrix(A)

    L = scaled_laplacian(W)
    Lk = cheb_poly(L, ks)
    Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
    model = STGCN(prior_mu = prior_mu, prior_sigma = prior_sigma, ks=ks, kt=kt,
                   bs = bs, T = T, n=n, Lk=Lk, p=p).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
            kl = kl_loss(model)
            cost = l + kl_weight * kl
            l_sum += cost.item() * y.shape[0]
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

def trainModel(name, mode, XS, YS,scaler):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    #summary(model, (CHANNEL, TIMESTEP_IN, N_NODE), device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    # training data trains the underlying model
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl = kl_loss(model)
    min_val_loss = np.inf
    wait = 0
    for epoch in range(EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            cost = loss + kl_weight * kl
            cost.backward(retain_graph=True)
            optimizer.step()
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
    YS, YS_pred = Utils.inverse_transform(np.squeeze(YS), scaler['mean'], scaler['std']), \
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
def testunmodel(name, mode, XS, YS, scaler, sample):
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    YS_pred = []
    for i in range(sample):
        YS_pred_ = np.squeeze(predictModel(model, test_iter))
        YS_pred_ = Utils.inverse_transform(YS_pred_, scaler['mean'], scaler['std'] )
        YS_pred.append(np.expand_dims(YS_pred_,axis=0))
    YS_pred = np.vstack(YS_pred)
    y_l_pred = np.min(YS_pred, axis=0)
    y_u_pred = np.max(YS_pred, axis=0)
    YS = Utils.inverse_transform(np.squeeze(YS), scaler['mean'], scaler['std'] )
    mask =  YS>0
    y_l_pred = y_l_pred * (y_l_pred>0)
    independent_coverage = np.logical_and(np.logical_and(y_u_pred >= YS, y_l_pred <= YS), YS>0)
# compute the coverage and interval width
    results = {}
    results["Point predictions"] = np.array(YS_pred)
    results["Upper limit"] = np.array(y_l_pred)
    results["Lower limit"] = np.array(y_u_pred)
    results["Confidence interval widths"] = np.abs(y_u_pred - y_l_pred)*mask
    results["Mean confidence interval widths"] = np.sum(results["Confidence interval widths"])/\
                                             np.sum(mask)
    results["Independent coverage indicators"] = independent_coverage
    results["Mean independent coverage"] = np.sum(independent_coverage.astype(float))/np.sum(mask)

    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
        % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

    print('*' * 40)
    print("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
        % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

################# Parameter Setting #######################
MODELNAME = 'STGCN' + str(TIMESTEP_OUT)

import argparse
import configparser
parser = argparse.ArgumentParser()
parser.add_argument('--config',default='../configuration/PEMS03.conf',type = str, help = 'configuration file path')
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--uncer_m', type=str, default='bayesian') # quantile/quantile_conformal/adaptive/dropout/bayesian
parser.add_argument('--dropout', type=float, default='0.3')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
config = configparser.ConfigParser()
config.read(args.config)
config_data = config['Data']
DATANAME = config_data['DATANAME']
FLOWPATH = config_data['FLOWPATH']
N_NODE = int(config_data['N_NODE'])
ADJPATH = config_data['ADJPATH']
UNCER_M = args.uncer_m

if UNCER_M == 'quantile':
    quantiles_list = [0.05, 0.95]
elif UNCER_M == 'quantile_conformal':
    quantiles_list = [0.15, 0.85]
elif UNCER_M == 'dropout':
    drop = args.dropout

KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + UNCER_M + '_' + datetime.now().strftime(
    "%y%m%d%H%M")
print(KEYWORD)
PATH = '../save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
##########################################################
# GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
# device = torch.device('cuda:6') if torch.cuda.is_available() else torch.device("cpu")
GPU = '0'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################

# data = pd.read_csv(FLOWPATH,index_col=[0]).values
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# print('data.shape', data.shape)
    
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('STGCN.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_STGCN6.py', PATH)

    if DATANAME == 'PEMS04':
        data = np.squeeze(np.load(FLOWPATH)['data'])[...,0]
    elif DATANAME == 'PEMS08':
        data = np.squeeze(np.load(FLOWPATH)['data'])[..., 0]
    elif DATANAME == 'PEMS03':
        data = np.squeeze(np.load(FLOWPATH)['data'])
    elif DATANAME == 'PEMS07':
        data = np.squeeze(np.load(FLOWPATH)['data'])
    elif DATANAME == 'METR-LA':
        data = pd.read_hdf(FLOWPATH).values
    elif DATANAME == 'PEMS-BAY':
        data = pd.read_hdf(FLOWPATH).values
    elif DATANAME == 'PEMSD7M':
        data = pd.read_csv(FLOWPATH,index_col=[0]).values

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
    testunmodel(MODELNAME, 'test', testXS, testYS, scaler, sample)
    
if __name__ == '__main__':
    main()

