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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
    ks, kt, bs, T, n, p, num_quantiles = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0, len(
        quantiles_list)
    A = pd.read_csv(ADJPATH).values
    W = weight_matrix(A)
    L = scaled_laplacian(W)
    Lk = cheb_poly(L, ks)
    Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
    model = STGCN(ks, kt, bs, T, n, Lk, p, num_quantiles).to(device)
    return model

def evaluateModel(model, data_iter, quantiles_list):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            # y_pred = model(x)
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
    '''
    return y_pred[:,:,:,:,i]

    '''
    model.eval()
    YS_pred = []

    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_ = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_)
    YS_pred = np.vstack(YS_pred)
    return YS_pred

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
    # take the 0.5 quantiles
    for i in range(len(quantiles_list)):
        YS_pred = Utils.inverse_transform(YS_pred_[..., i], scaler['mean'], scaler['std'])
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)

        with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
            f.write("%s, %s, quantile, MSE, RMSE, MAE, MAPE, %.2f, %.10f, %.10f, %.10f, %.10f\n" % (
                name, mode, quantiles_list[i], MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e\n" % (name, mode, torch_score))
    print('Model Training Ended ...', time.ctime())

def quantile_loss_v2(y_pred, y, q):
    '''
    this version is max loss between prediction and true value
    '''

    error = (y_pred - y)
    single_loss = torch.max(q * error, (q - 1) * error)
    loss = torch.mean(single_loss)
    return loss

def V1_calModel(name, mode, XS, YS, scaler,cal_q):
    '''
    this version is quantile calibration with referenced with maximum error
    '''
    print("model fixed calibration state:")
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    print('MODEL CALIBRATION STATE:')
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    cal_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    cal_iter = torch.utils.data.DataLoader(cal_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS_pred_0 = predictModel(model, cal_iter)[..., 0]
    YS_pred_1 = predictModel(model, cal_iter)[..., 1]
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred_0.shape)
    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(YS_pred_0), np.squeeze(YS_pred_1)
    YS, YS_pred_0, YS_pred_1 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])
    # need to over 0
    YS_pred_1 = YS_pred_1 * (YS_pred_1>0)
    error_low = YS_pred_1 - YS
    error_high = np.absolute(YS - YS_pred_0)
    error = np.maximum(error_low,error_high)

    cali_error = []
    for i in range(len(cal_q)):
        cali_error_ = np.quantile(error,cal_q[i])
        cali_error.append(cali_error_)
    return [np.stack(cali_error),np.stack(cali_error)]

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
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS_pred_0 = predictModel(model, cal_iter)[..., 0]
    YS_pred_1 = predictModel(model, cal_iter)[..., 1]
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred_0.shape)
    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(YS_pred_0), np.squeeze(YS_pred_1)
    YS, YS_pred_0, YS_pred_1 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])

    error_low = (YS - YS_pred_1)* (YS_pred_1 > 0)* (YS>0)
    error_high = (YS_pred_0 - YS)* (YS>0)

    err_dis = [error_low,error_high]
    error_quantile = 101
    corr_u_err_list = []
    corr_l_err_list = []

    # return 上边界残差[100，207] 下边界残差[100，207]
    # 从中选择上下边界残差 return [2,207] 循环 100* 100
    # 计算上下边界
    # 计算coverage，返回coverage，interval, return [207, 100]
    # 画图，计算拐点

    # return 上边界残差[100，207] 下边界残差[100，207]
    # 放到gpu上计算
    # y_pred_l 如果小于0，则等于0
    # ys 如果小于0，则等于0

    YS_pred_0 = torch.Tensor(YS_pred_0).to(device)
    YS_pred_1 = torch.Tensor(YS_pred_1).to(device)
    YS = torch.Tensor(YS).to(device)

    for j in range(1, error_quantile):
        corr_u_err = np.quantile(err_dis[1], 1 / (error_quantile - 1) * j, axis=0)
        corr_u_err = torch.Tensor(corr_u_err).to(device)
        corr_u_err_list.append(corr_u_err)
    for z in range(1, error_quantile):
        corr_l_err = np.quantile(err_dis[0], 1 / (error_quantile - 1) * z, axis=0)
        corr_l_err = torch.Tensor(corr_l_err).to(device)
        corr_l_err_list.append(corr_l_err)
    coverage_list = []
    interval_list = []
    for m in range(0, error_quantile - 5):
        for n in range(0, error_quantile - 5):
            y_u_pred = YS_pred_0 + corr_u_err_list[m]
            y_l_pred = YS_pred_1 - corr_l_err_list[n]
            y_l_pred = y_l_pred * (y_l_pred >=0)
            mask = YS > 0
            coverage = torch.logical_and(torch.logical_and(y_u_pred >= YS, y_l_pred <= YS), mask)
            coverage_a = torch.sum(coverage.float(), axis=0)/torch.sum(mask, axis=0)
            coverage_list.append(coverage_a)
            interval = torch.abs(y_u_pred - y_l_pred)*mask
            interval_a = torch.sum(interval, axis=0)/torch.sum(mask, axis=0)
            interval_list.append(interval_a)

    coverage_list = torch.stack(coverage_list)
    interval_list = torch.stack(interval_list)
    group = int(np.sqrt(len(coverage_list)))
    node = interval_list.size(1)

    # 归一化interval_list
    interval_nor = [(interval_list[:, t] - torch.min(interval_list[:, t])) /
                    (torch.max(interval_list[:, t]) - torch.min(interval_list[:, t])) for t in range(node)]
    interval_nor = torch.stack(interval_nor).T

    coverage_nor = [(coverage_list[:, t] - torch.min(coverage_list[:, t])) /
                    (torch.max(coverage_list[:, t]) - torch.min(coverage_list[:, t])) for t in range(node)]
    coverage_nor = torch.stack(coverage_nor).T

    corr_u_err_l, corr_l_err_ = [], []
    for i in range(len(lambda_list)):
        loss = [- lambda_list[i] * coverage_nor[:, t] + (1 - lambda_list[i]) * interval_nor[:, t] for t in range(node)]
        loss = torch.stack(loss).T
        index = [torch.argmin(loss[:, t]) for t in range(node)]
        index = torch.stack(index).cpu().numpy()
        index_u = (np.trunc(index / group)).astype(int)
        index_l = (np.mod(index, group)).astype(int)
        corr_u_err = [torch.stack(corr_u_err_list)[index_u[t], t] for t in range(node)]
        corr_l_err = [torch.stack(corr_l_err_list)[index_l[t], t] for t in range(node)]
        corr_u_err = torch.stack(corr_u_err).cpu().numpy()
        corr_l_err = torch.stack(corr_l_err).cpu().numpy()
        corr_u_err_l.append(corr_u_err)
        corr_l_err_.append(corr_l_err)

    return [corr_u_err_l, corr_l_err_]

def ftestunModel_1(name, mode, XS, YS,scaler):
    print('model uncertainty test')
    print('timestep_in, timestep_out', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data)
    model = getModel(name)
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

def ftestunModel_2(name, mode, XS, YS, quantiles_list, err, scaler, cal_list):
    print('model uncertainty test')
    print('timestep_in, timestep_out', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(predictModel(model, test_iter)[...,0]), \
                               np.squeeze(predictModel(model, test_iter)[...,1])
    YS, YS_pred_0, YS_pred_1 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])

    for i in range(len(cal_list)):
        y_u_pred = YS_pred_0 + err[0][i]
        y_l_pred = YS_pred_1 - err[1][i]

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

        results["Calbration error"] = np.mean(err)

        with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
            f.write("calibration error,  %.4f\n "
                % results["Calbration error"])
            f.write("calibration quantile rate,  %.2f\n "
                % cal_list[i])
            f.write("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
            % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

        print('*' * 40)
        print("Calibration error, %.4f\n" % np.mean(err[0][i] + err[1][i]))
        print("calibration quantile rate, %.2f\n" % cal_list[i])
        print("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
            % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

################# Parameter Setting #######################
MODELNAME = 'STGCN' + str(TIMESTEP_OUT)
#KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + 'quantile' + '_' + datetime.now().strftime("%y%m%d%H%M")
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + '0.70adaptive' + '_' + datetime.now().strftime(
    "%y%m%d%H%M")
PATH = '../save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
###########################################################

GPU = '0'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('STGCN.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_STGCN12.py', PATH)

    data = pd.read_hdf(FLOWPATH).values
    print('data.shape', data.shape)
    # data = np.squeeze(np.load(FLOWPATH)['data'])
    # print('data.shape', data.shape)
    trainx, trainy = getXSYS_single(data, 'TRAIN')
    # transform
    mean = trainx.mean()
    std = trainy.std()
    scaler = {'mean': mean, 'std': std}
    data = Utils.transform(data, scaler['mean'], scaler['std'])

    trainXS, trainYS = getXSYS_single(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS, quantiles_list, scaler)
    print('training ended')

# # for quantile test
#     KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + 'quantile' + '_' + datetime.now().strftime(
#         "%y%m%d%H%M")
#     print(KEYWORD, 'testing started', time.ctime())
#     testXS, testYS = getXSYS_single(data, 'TEST')
#     print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
#     ftestunModel_1(MODELNAME, 'test', testXS, testYS, scaler)
#     print('testing ended')

# # for quantile conformal calibration
#     KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + 'quantile_conformal' + '_' + datetime.now().strftime(
#         "%y%m%d%H%M")
#     print(KEYWORD, 'cal started', time.ctime())
#     calXS, calYS = getXSYS_single(data, 'CAL')
#     print('CAl XS.shape YS,shape', calXS.shape, calXS.shape)
#     err = V1_calModel(MODELNAME, 'cal', calXS, calYS, scaler, cal_list)
#     print(KEYWORD, 'cal ended', time.ctime())
# # for quantile conformal test
#     print(KEYWORD, 'testing started', time.ctime())
#     testXS, testYS = getXSYS_single(data, 'TEST')
#     print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
#     ftestunModel_2(MODELNAME, 'test', testXS, testYS, quantiles_list, err, scaler,cal_list)
#     print(KEYWORD, 'testing ended', time.ctime())

# for our method calibration
    KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + '0.85adaptive' + '_' + datetime.now().strftime(
       "%y%m%d%H%M")
    print(KEYWORD, 'cal started', time.ctime())
    calXS, calYS = getXSYS_single(data, 'CAL')
    print('CAl XS.shape YS,shape', calXS.shape, calXS.shape)
    err = V2_calModel(MODELNAME, 'cal', calXS, calYS, scaler, lambda_list)
    print(KEYWORD, 'cal ended', time.ctime())
# for our method test
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS_single(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    ftestunModel_2(MODELNAME, 'test', testXS, testYS, quantiles_list, err, scaler, lambda_list)
    print(KEYWORD, 'testing ended', time.ctime())


if __name__ == '__main__':
    main()

