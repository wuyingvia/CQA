
import shutil
import pandas as pd
from datetime import datetime
import time

import Metrics
from GMAN import *
from Param import *
from Param_GMAN import *
import Utils
import os
torch.set_num_threads(1)

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    CAL_NUM = int(data.shape[0] * CALRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)

    elif mode == 'CAL':
        for i in range(TRAIN_NUM - TIMESTEP_IN, CAL_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1 ):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(CAL_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
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
    df.index = pd.to_datetime(df.index).astype('datetime64[ns]')
    # torch.tensor: (34272, 1)     Value: 0-6, Monday=0, Sunday=6
    dayofweek = np.reshape(np.array(time.weekday), (-1, 1))
    timeofday = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    timeofday = np.reshape(timeofday, (-1, 1))
    time = np.concatenate((dayofweek, timeofday), -1)
    
    TRAIN_NUM = int(time.shape[0] * TRAINRATIO)
    CAL_NUM = int(time.shape[0] * CALRATIO)
    TE = []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            t = time[i:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            TE.append(t)
    elif mode == 'CAL':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  CAL_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            t = time[i:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            TE.append(t)
    elif mode == 'TEST':
        for i in range(CAL_NUM - TIMESTEP_IN,  time.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
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

def getModel(name, device, SEPATH):
    SE = getSE(SEPATH).to(device=device)
    model = GMAN(SE, timestep_in=TIMESTEP_IN, device = device).to(device)
    return model

def evaluateModel(model, data_iter,quantiles_list):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, te, y  in data_iter:
            #y_pred = model(x)
            y_pred = model(x, te)
            l = 0.0
            for i in range(len(quantiles_list)):
                y_pred_ = y_pred[..., i]
                loss_ = quantile_loss_v2(y_pred_, y, quantiles_list[i])
                l += loss_
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
            # (16, 12, 170, 2)
            YS_pred_batch = YS_pred_batch.reshape([-1, x.shape[1],x.shape[2],2])
            YS_pred.append(YS_pred_batch)
        m = np.vstack(YS_pred)
    return m

def quantile_loss_v2(y_pred, y, q):
    '''
    this version is max loss between prediction and true value
    '''
    error = (y_pred-y)
    single_loss = torch.max(q*error, (q-1)*error)
    loss = torch.mean(single_loss)
    return loss

def trainModel(name, mode, XS, TE, YS, device,quantiles_list, scaler, SEPATH):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name, device, SEPATH)
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
    for epoch in range(EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, te, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x, te)
            loss = 0.0
            for i in range(len(quantiles_list)):
                y_pred_ = y_pred[..., i]
                loss_ = quantile_loss_v2(y_pred_, y, quantiles_list[i])
                loss += loss_
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
            del x, te, y, y_pred, loss
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, val_iter, quantiles_list)
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
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", train_loss, "validation loss:",
              val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % (
                "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:",
                val_loss))

    torch_score = evaluateModel(model, train_iter, quantiles_list)
    YS_pred_ = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    YS = Utils.inverse_transform(YS, scaler['mean'], scaler['std'])

    for i in range(len(quantiles_list)):
        YS_pred = Utils.inverse_transform(YS_pred_[..., i], scaler['mean'], scaler['std'])
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)

        with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
            f.write("%s, %s, quantile, MSE, RMSE, MAE, MAPE, %.2f, %.10f, %.10f, %.10f, %.10f\n" % (
                name, mode, quantiles_list[i], MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e\n" % (name, mode, torch_score))
    print('Model Training Ended ...', time.ctime())

def V1_calModel(name, mode, XS, TE, YS, scaler,cal_q, SEPATH):
    print("model fixed calibration state:")
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    print('MODEL CALIBRATION STATE:')
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    TE_torch = torch.Tensor(TE).to(device)
    cal_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    cal_iter = torch.utils.data.DataLoader(cal_data, BATCHSIZE, shuffle=False)
    model = getModel(name, device, SEPATH)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS_pred = predictModel(model, cal_iter)
    YS_pred_0 = YS_pred[..., 0]
    YS_pred_1 = YS_pred[..., 1]
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred_0.shape)
    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(YS_pred_0), np.squeeze(YS_pred_1)
    YS, YS_pred_0, YS_pred_1 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])

    # choose the best expected quantile
    # split some data
    split = int(YS.shape[0]* CALSPLIT)
    YS_train = YS[:split]
    YS_0_train = YS_pred_0[:split]
    YS_1_train = YS_pred_1[:split]
    YS_val = YS[split:]
    YS_0_val = YS_pred_0[split:]
    YS_1_val = YS_pred_1[split:]

    YS_1_train = YS_1_train * (YS_1_train>0)
    error_low =  YS_train - YS_1_train
    error_high =  YS_0_train - YS_train
    error = np.maximum(error_low,error_high)

    cali_error = []
    for i in range(len(cal_q)):
        cali_error_ = np.quantile(error,cal_q[i])
        cali_error.append(cali_error_)

    err = [np.stack(cali_error),np.stack(cali_error)]
    independent_coverage_l = []
    for i in range(len(cal_list)):
        y_u_pred = YS_0_val + err[0][i]
        y_l_pred = YS_1_val - err[1][i]

        y_l_pred = y_l_pred * (y_l_pred>0)
        mask = YS_val>0
        independent_coverage = np.logical_and(np.logical_and(y_u_pred >= YS_val, y_l_pred <= YS_val), mask)
        m_coverage = np.sum(independent_coverage.astype(float))/np.sum(mask)
        independent_coverage_l.append(m_coverage)
    m_coverage = np.stack(independent_coverage_l)
    index = np.argmin(np.abs(m_coverage - 0.90))
    return  [err[0][index], err[1][index]]

def V2_calModel(name, mode, XS,  TE, YS,  scaler, lambda_list, SEPATH):
    '''
    this is to compute conformity scores through eq 7
    the first time to compute conformity scores through residual between prediction and true value
    the second correction of conformity scores through max{\hat{q}_low - y, y - \hat{q}_high}
    return the interval
    '''
    print('MODEL CALIBRATION STATE:')
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    TE_torch = torch.Tensor(TE).to(device)
    cal_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    cal_iter = torch.utils.data.DataLoader(cal_data, BATCHSIZE, shuffle=False)
    model = getModel(name, device, SEPATH)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS_pred = predictModel(model, cal_iter)
    YS_pred_0 = YS_pred[..., 0]
    YS_pred_1 = YS_pred[..., 1]
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred_0.shape)
    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(YS_pred_0), np.squeeze(YS_pred_1)
    YS, YS_pred_0, YS_pred_1 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])

    YS_pred_0 = torch.Tensor(YS_pred_0).to(device)
    YS_pred_1 = torch.Tensor(YS_pred_1).to(device)
    YS = torch.Tensor(YS).to(device)

     # split some data
    split = int(YS.shape[0]* CALSPLIT)

    YS_train = YS[:split]
    YS_0_train = YS_pred_0[:split]
    YS_1_train = YS_pred_1[:split]

    YS_val = YS[split:]
    YS_0_val = YS_pred_0[split:]
    YS_1_val = YS_pred_1[split:]

    YS_1_train = YS_1_train * (YS_1_train>0)
    error_low =  YS - YS_pred_1
    error_high =  YS_pred_0 - YS

    err_dis = torch.cat([error_low,error_high])
    error_quantile = 100

    node = YS.size(2)
    seq = YS.size(1)

    # return [error_q, seq, node]
    corr_err_list = []
    for q in range(0, error_quantile+1):
        q_n = []
        for n in range(node):
            q_t = []
            for t in range(seq):
                corr_err = torch.quantile(err_dis[:, t, n], q/error_quantile)
                q_t.append(corr_err)
            q_t = torch.stack(q_t)
            q_n.append(q_t)
        q_n = torch.stack(q_n).T
        corr_err_list.append(q_n)
    corr_err_list = torch.stack(corr_err_list)
    # corr_err_list = []
    # for q in range(0, error_quantile+1):
    #     corr_err = torch.quantile(err_dis, q/error_quantile, axis=0)
    #     corr_err_list.append(corr_err)
    # corr_err_list = torch.stack(corr_err_list)

    coverage_list = []
    interval_list = []
    for m in range(0, error_quantile+1):
        y_u_pred = YS_pred_0 + corr_err_list[m]
        y_l_pred = YS_pred_1 - corr_err_list[m]
        y_l_pred = y_l_pred * (y_l_pred >=0)
        mask = YS > 0

        coverage = torch.logical_and(y_u_pred >= YS, y_l_pred <= YS)
        coverage_a = torch.mean(coverage.float(),axis=0)
        coverage_list.append(coverage_a)
        interval_a = torch.mean(torch.abs(y_u_pred - y_l_pred),axis=0)       
        interval_list.append(interval_a)

    # torch.Size([sampe, seq, node])
    coverage_list = torch.stack(coverage_list)
    interval_list = torch.stack(interval_list)


    # 归一化interval_list
    interval_nor = []
    for n in range(node):
        interval_n = []
        for t in range(seq):
            interval_ = (interval_list[:,t, n] - torch.min(interval_list[:,t, n])) /\
                        (torch.max(interval_list[:,t, n]) - torch.min(interval_list[:,t, n]))
            interval_n.append(interval_)
        interval_n = torch.stack(interval_n)
        interval_nor.append(interval_n)
    interval_nor = torch.stack(interval_nor).T

    coverage_nor = []
    for n in range(node):
        coverage_n = []
        for t in range(seq):
            coverage_ = (coverage_list[:,t, n] - torch.min(coverage_list[:,t, n])) /(
                    torch.max(coverage_list[:,t, n]) - torch.min(coverage_list[:,t, n]) + 1e-5)
            coverage_n.append(coverage_)
        coverage_n = torch.stack(coverage_n)
        coverage_nor.append(coverage_n)
    coverage_nor = torch.stack(coverage_nor).T 

    cor_err = []
    lambda_list = torch.Tensor(lambda_list).to(device)
    for i in lambda_list:
        loss = - i * coverage_nor + (1 - i) * interval_nor
        index = torch.argmin(loss,axis=0).cpu().numpy()
        err_t = []
        for t in range(seq):
            err_n = []
            for n in range(node):
                corr_err = corr_err_list[index[t,n], t, n]
                err_n.append(corr_err)
            err_n = torch.stack(err_n)
            err_t.append(err_n)
        err_i = torch.stack(err_t)
        cor_err.append(err_i)
    cor_err = torch.stack(cor_err)

    # 选择期望的cor_err
    independent_coverage_l =[]
    for i in range(len(lambda_list)):
        y_u_pred = YS_0_val + cor_err[i]
        y_l_pred = YS_1_val - cor_err[i]

        y_l_pred = y_l_pred * (y_l_pred>0)  
        mask = YS_val>0
        independent_coverage = torch.logical_and(torch.logical_and(y_u_pred >= YS_val, 
        y_l_pred <= YS_val), mask)
        m_coverage = torch.sum(independent_coverage)/torch.sum(mask)
        independent_coverage_l.append(m_coverage)
    m_coverage = torch.stack(independent_coverage_l)
    index = torch.argmin(torch.abs(m_coverage - 0.91))

    return [cor_err[index].cpu().numpy(), cor_err[index].cpu().numpy()]

def ftestunModel_1(name, mode, XS, TE, YS, scaler, SEPATH):
    print('model uncertainty test')
    print('timestep_in, timestep_out', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, TE_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(TE).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name, device, SEPATH)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(predictModel(model, test_iter)[...,0]), \
                               np.squeeze(predictModel(model, test_iter)[...,1])
    YS, YS_pred_0, YS_pred_1 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])


    y_u_pred = YS_pred_0
    y_l_pred = YS_pred_1

    y_l_pred = y_l_pred * (y_l_pred>0)
    mask = YS>0
    independent_coverage = np.logical_and(np.logical_and(y_u_pred >= YS, y_l_pred <= YS), YS>0)
# compute the coverage and interval width
    results = {}
    results["Point predictions"] = np.array(YS)
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

def ftestunModel_2(name, mode, XS, TE, YS, err, scaler, cal_list, SEPATH):

    print('model uncertainty test')
    print('timestep_in, timestep_out', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, TE_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(TE).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name, device, SEPATH)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(predictModel(model, test_iter)[...,0]), \
                               np.squeeze(predictModel(model, test_iter)[...,1])
    YS, YS_pred_0, YS_pred_1 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])

    y_u_pred = YS_pred_0 + err[0]
    y_l_pred = YS_pred_1 - err[1]

    y_l_pred = y_l_pred * (y_l_pred > 0)
    mask = YS > 0
    independent_coverage = np.logical_and(np.logical_and(y_u_pred >= YS, y_l_pred <= YS), YS > 0)
    # compute the coverage and interval width
    results = {}
    results["Upper limit"] = np.array(y_l_pred)
    results["Lower limit"] = np.array(y_u_pred)
    results["Confidence interval widths"] = np.abs(y_u_pred - y_l_pred) * mask
    results["Mean confidence interval widths"] = np.sum(results["Confidence interval widths"]) / \
                                                 np.sum(mask)
    results["Independent coverage indicators"] = independent_coverage
    results["Mean independent coverage"] = np.sum(independent_coverage.astype(float)) / np.sum(mask)

    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
                % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

    print('*' * 40)
    print("Calibration error, %.4f\n" % np.mean(err[0] + err[1]))
    print("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
          % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

################# Parameter Setting #######################
MODELNAME = 'GMAN'

import argparse
import configparser
parser = argparse.ArgumentParser()
parser.add_argument('--config',default='../configuration/PEMS08.conf',type = str, help = 'configuration file path')
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--uncer_m', type=str, default='quantile_conformal') # quantile/quantile_conformal/adaptive
parser.add_argument('--q', type=float, default=0.85) # quantile/quantile_conformal/adaptive

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
config = configparser.ConfigParser()
config.read(args.config)
config_data = config['Data']
DATANAME = config_data['DATANAME']
FLOWPATH = config_data['FLOWPATH_GMAN']
N_NODE = int(config_data['N_NODE'])
ADJPATH = config_data['ADJPATH']
FLOWPATH_GMAN = config_data['FLOWPATH_GMAN']
SEPATH = config_data['SEPATH']
UNCER_M = args.uncer_m
q = args.q
if UNCER_M == 'quantile':
    quantiles_list = [1-q, q]
elif UNCER_M == 'quantile_conformal':
    quantiles_list = [1-q, q]
elif UNCER_M == 'adaptive':
    quantiles_list = [1-q, q]

KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + UNCER_M + '_' + datetime.now().strftime(
    "%y%m%d%H%M")
print(KEYWORD)

PATH = '../save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
# torch.backends.cudnn.deterministic = True
###########################################################
# GPU = sys.argv[-1] if len(sys.argv) == 2 else '5'
# device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")

GPU = '0'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################

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

    if DATANAME == 'PEMS04':
        df = pd.read_csv(FLOWPATH_GMAN,index_col=[0])
        data = df.values
        df.index = pd.to_datetime(df.index)
    elif DATANAME == 'PEMS08':
        df = pd.read_csv(FLOWPATH_GMAN,index_col=[0])
        data = df.values
        df.index = pd.to_datetime(df.index)
    elif DATANAME == 'PEMS03':
        df = pd.read_csv(FLOWPATH_GMAN,index_col=[0])
        data = df.values
        df.index = pd.to_datetime(df.index)
    elif DATANAME == 'PEMS07':
        df = pd.read_csv(FLOWPATH_GMAN, index_col=[0])
        data = df.values
        df.index = pd.to_datetime(df.index)
    elif DATANAME == 'METR-LA':
        df = pd.read_hdf(FLOWPATH)
        data = df.values
    elif DATANAME == 'PEMS-BAY':
        df = pd.read_hdf(FLOWPATH)
        data = df.values
    elif DATANAME == 'PEMSD7M':
        df = pd.read_csv(FLOWPATH,index_col=[0])
        data = df.values
        df.index = pd.to_datetime(df.index)

    print('data.shape', data.shape)

    trainx, trainy = getXSYS(data, 'TRAIN')
    # transform
    mean = trainx.mean()
    std = trainy.std()
    scaler = {'mean': mean, 'std': std}
    data = Utils.transform(data, scaler['mean'], scaler['std'])

# # for training
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS.shape', trainXS.shape, trainYS.shape)
    trainTE = getTE(df, "TRAIN")
    print('TRAIN TE.shape', trainTE.shape)
    trainModel(MODELNAME, "train", trainXS, trainTE, trainYS, device, quantiles_list, scaler, SEPATH)

    if UNCER_M == 'quantile':
        # for quantile test
        testXS, testYS = getXSYS(data, 'TEST')
        print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
        testTE = getTE(df, "TEST")
        ftestunModel_1(MODELNAME, 'test', testXS, testTE, testYS, scaler, SEPATH)
        print('testing ended')
    elif UNCER_M == 'quantile_conformal':
    # for quantile conformal calibration
        print('cal started', time.ctime())
        calXS, calYS = getXSYS(data, 'CAL')
        calTE = getTE(df, "CAL")
        print('CAl XS.shape YS,shape', calXS.shape, calXS.shape)
        err = V1_calModel(MODELNAME, 'cal', calXS, calTE, calYS, scaler, cal_list, SEPATH)
        print('cal ended', time.ctime())
    # for quantile conformal test
        print('testing started', time.ctime())
        testXS, testYS = getXSYS(data, 'TEST')
        testTE = getTE(df, "TEST")
        print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
        ftestunModel_2(MODELNAME, 'test', testXS, testTE, testYS, err, scaler, cal_list, SEPATH)
        print('testing ended', time.ctime())
    elif UNCER_M == 'adaptive':
     # # for our method calibration
        print(KEYWORD, 'cal started', time.ctime())
        calXS, calYS = getXSYS(data, 'CAL')
        calTE = getTE(df, "CAL")
        print('CAl XS.shape YS,shape', calXS.shape, calXS.shape)
        err = V2_calModel(MODELNAME, 'cal', calXS, calTE, calYS, scaler, lambda_list, SEPATH)
        print(KEYWORD, 'cal ended', time.ctime())
        # for quantile conformal test
        print(KEYWORD, 'testing started', time.ctime())
        testXS, testYS = getXSYS(data, 'TEST')
        testTE = getTE(df, "TEST")
        print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
        ftestunModel_2(MODELNAME, 'test', testXS, testTE, testYS, err, scaler, lambda_list, SEPATH)
        print(KEYWORD, 'testing ended', time.ctime())
if __name__ == '__main__':
    main()


