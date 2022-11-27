import sys
import os
import shutil

import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import Metrics
import Utils
from GraphWaveNet import *
from Param import *
from Param_GraphWaveNet import *
torch.set_num_threads(1)
def getTimestamp(data):
    # data is a pandas dataframe with timestamp ID.
    data_feature = data.values.reshape(data.shape[0],data.shape[1],1)
    feature_list = [data_feature]
    num_samples, num_nodes = data.shape
    time_ind = (data.index.values - data.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_in_day

def getXSYSTIME(data, data_time, mode):
    # When CHANNENL = 2, use this function to get data plus time as two channels.
    # data: numpy, data_time: numpy from getTimestamp 
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    CAL_NUM = int(data.shape[0] * CALRATIO)
    XS, YS, XS_TIME = [], [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            t = data_time[i:i + TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    elif mode == 'CAL':
        for i in range(TRAIN_NUM - TIMESTEP_IN, CAL_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            t = data_time[i:i + TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            t = data_time[i:i + TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    XS, YS, XS_TIME = np.array(XS), np.array(YS), np.array(XS_TIME)
    XS = np.concatenate([np.expand_dims(XS, axis=-1), np.expand_dims(XS_TIME, axis=-1)], axis=-1)
    XS, YS = XS.transpose(0, 3, 2, 1), YS[:, :, :, np.newaxis]
    return XS, YS

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
        for i in range(CAL_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def getModel(name):
    # way1: only adaptive graph.
    # model = gwnet(device, num_nodes = N_NODE, in_dim=CHANNEL).to(device)
    # return model

    # way2: adjacent graph + adaptive graph

    adj_mx = load_adj(ADJPATH, ADJTYPE, DATANAME)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    model = gwnet(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=supports, dropout=0.0).to(device)
    return model

def evaluateModel(model, data_iter,quantiles_list):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            #y_pred = model(x)
            y_pred = model(x)
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
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def quantile_loss_v2( y_pred, y, q):
    '''
    this version is max loss between prediction and true value
    '''

    error = (y_pred-y)
    single_loss = torch.max(q*error, (q-1)*error)
    loss = torch.mean(single_loss)
    return loss

def trainModel(name, mode, XS, YS, quantiles_list, scaler):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)

    min_val_loss = np.inf
    wait = 0

    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)

    for epoch in range(EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = 0.0
            for i in range(len(quantiles_list)):
                y_pred_ = y_pred[..., i]
                loss_ = quantile_loss_v2(y_pred_, y, quantiles_list[i])
                loss += loss_
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
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

def V1_calModel(name, mode, XS, YS, scaler,cal_q):
    print("model fixed calibration state:")
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    print('MODEL CALIBRATION STATE:')
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    cal_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    cal_iter = torch.utils.data.DataLoader(cal_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    #model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    model.load_state_dict(torch.load('/home/wuying/CODE/mymodel/2022/DL-Traff-Graph/results/表格的结果/pred_PEMS04_GraphWaveNet_quantile_conformal_2211091844/GraphWaveNet.pt'))

    YS_pred_0 = predictModel(model, cal_iter)[..., 0]
    YS_pred_1 = predictModel(model, cal_iter)[..., 1]
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

def V2_calModel(name, mode, XS, YS,  scaler, lambda_list):
    '''
    this version is to compute our methods
    '''
    print('MODEL CALIBRATION STATE:')
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    cal_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    cal_iter = torch.utils.data.DataLoader(cal_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    #model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    model.load_state_dict(torch.load('/home/wuying/CODE/mymodel/2022/DL-Traff-Graph/results/表格的结果/pred_PEMS04_GraphWaveNet_quantile_conformal_2211091844/GraphWaveNet.pt'))

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
    index = torch.argmin(torch.abs(m_coverage - 0.90))

    return [cor_err[index].cpu().numpy(), cor_err[index].cpu().numpy()]

def ftestunModel_1(name, mode, XS, YS,scaler):
    print('model uncertainty test')
    print('timestep_in, timestep_out', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data)
    model = getModel(name)
    #model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    model.load_state_dict(torch.load('/home/wuying/CODE/mymodel/2022/DL-Traff-Graph/results/表格的结果/pred_PEMS04_GraphWaveNet_quantile_conformal_2211091844/GraphWaveNet.pt'))

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

    # save the prediction data
    np.save(PATH + '/' + MODELNAME + '_pred_lower.npy', y_l_pred)
    np.save(PATH + '/' + MODELNAME + '_pred_upper.npy', y_u_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)

    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
        % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

    print('*' * 40)
    print("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
        % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

    # save the prediction data
    # np.save(PATH + '/' + MODELNAME + '_pred_lower.npy', y_l_pred)
    # np.save(PATH + '/' + MODELNAME + '_pred_upper.npy', y_u_pred)
    # np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)

def ftestunModel_2(name, mode, XS, YS, quantiles_list, err, scaler, cal_list):
    print('model uncertainty test')
    print('timestep_in, timestep_out', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data)
    model = getModel(name)
    #model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    model.load_state_dict(torch.load('/home/wuying/CODE/mymodel/2022/DL-Traff-Graph/results/表格的结果/pred_PEMS04_GraphWaveNet_quantile_conformal_2211091844/GraphWaveNet.pt'))

    YS_pred_0 = predictModel(model, test_iter)[...,0]
    YS_pred_1 = predictModel(model, test_iter)[...,1]

    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(YS_pred_0), \
                               np.squeeze(YS_pred_1)
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
    results["Calbration error"] = np.mean(err)


    # save the prediction data
    np.save(PATH + '/' + MODELNAME + '_pred_lower.npy', y_l_pred)
    np.save(PATH + '/' + MODELNAME + '_pred_upper.npy', y_u_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)


    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("calibration error,  %.4f\n "
                % results["Calbration error"])
        f.write("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
                % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

    print('*' * 40)
    print("Calibration error, %.4f\n" % np.mean(err[0] + err[1]))
    print("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
          % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

################# Parameter Setting #######################
MODELNAME = 'GraphWaveNet'

import argparse
import configparser
parser = argparse.ArgumentParser()
parser.add_argument('--config',default='../configuration/PEMS04.conf',type = str, help = 'configuration file path')
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--uncer_m', type=str, default='adaptive') # quantile/quantile_conformal/adaptive
parser.add_argument('--q', type=float, default=0.85) # quantile/quantile_conformal/adaptive

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
q = args.q
if UNCER_M == 'quantile':
    quantiles_list = [0.05, 0.95]
elif UNCER_M == 'quantile_conformal':
    quantiles_list = [0.15, 0.85]
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
    shutil.copy2('GraphWaveNet.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_GraphWaveNet.py', PATH)

    if DATANAME == 'PEMS04':
        data = np.squeeze(np.load(FLOWPATH)['data'])[...,0]
    elif DATANAME == 'PEMS08':
        data = np.squeeze(np.load(FLOWPATH)['data'])[...,0]
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

    trainx, trainy = getXSYS(data, 'TRAIN')
    # transform
    mean = trainx.mean()
    std = trainy.std()
    scaler = {'mean': mean, 'std': std}
    data = Utils.transform(data, scaler['mean'], scaler['std'])

    # trainXS, trainYS = getXSYS(data, 'TRAIN')
    # print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    # trainModel(MODELNAME, 'train', trainXS, trainYS, quantiles_list, scaler)
    # print('training ended')

    if UNCER_M == 'quantile':
    # for quantile test
        print(KEYWORD, 'testing started', time.ctime())
        testXS, testYS = getXSYS(data, 'TEST')
        print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
        ftestunModel_1(MODELNAME, 'test', testXS, testYS, scaler)
        print(KEYWORD, 'testing ended')
    elif UNCER_M == 'quantile_conformal':
        # for quantile conformal calibration
        print('cal started', time.ctime())
        calXS, calYS = getXSYS(data, 'CAL')
        print('CAl XS.shape YS,shape', calXS.shape, calYS.shape)
        err = V1_calModel(MODELNAME, 'cal', calXS, calYS, scaler, cal_list)
        print(KEYWORD, 'cal ended', time.ctime())
        # for quantile conformal test
        print(KEYWORD, 'testing started', time.ctime())
        testXS, testYS = getXSYS(data, 'TEST')
        print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
        ftestunModel_2(MODELNAME, 'test', testXS, testYS, quantiles_list, err, scaler,cal_list)
        print(KEYWORD, 'testing ended', time.ctime())

    elif UNCER_M == 'adaptive':
        # for our method calibration
        print(KEYWORD, 'cal started', time.ctime())
        calXS, calYS = getXSYS(data, 'CAL')
        print('CAl XS.shape YS,shape', calXS.shape, calYS.shape)
        err = V2_calModel(MODELNAME, 'cal', calXS, calYS, scaler, lambda_list)
        print(KEYWORD, 'cal ended', time.ctime())
        # for our method test
        print(KEYWORD, 'testing started', time.ctime())
        testXS, testYS = getXSYS(data, 'TEST')
        print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
        ftestunModel_2(MODELNAME, 'test', testXS, testYS, quantiles_list, err, scaler, lambda_list)

        
        print(KEYWORD, 'testing ended', time.ctime())


if __name__ == '__main__':
    main()

