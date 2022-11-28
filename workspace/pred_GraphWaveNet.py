
import os
import shutil
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
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            t = data_time[i:i+TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    elif mode == 'CAL':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  CAL_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            t = data_time[i:i+TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    elif mode == 'TEST':
        for i in range(CAL_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            t = data_time[i:i+TIMESTEP_IN, :]
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
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'CAL':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  CAL_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(CAL_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def getModel(name, ADJPATH):
    # way1: only adaptive graph.
    # model = gwnet(device, num_nodes = N_NODE, in_dim=CHANNEL).to(device)
    # return model
    
    # way2: adjacent graph + adaptive graph
    adj_mx = load_adj(ADJPATH, ADJTYPE, DATANAME)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    model = gwnet(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=supports).to(device)
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

def trainModel(name, mode, XS, YS, scaler, ADJPATH):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name, ADJPATH)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
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
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
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

def calModel(name, mode, XS, YS, scaler, cal_list, ADJPATH):
    '''
    this version is quantile calibration with referenced with maximum error
    '''
    print("model fixed calibration state:")
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    print('MODEL CALIBRATION STATE:')
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    # split half data to choose the the expected calibration quantile


    cal_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    cal_iter = torch.utils.data.DataLoader(cal_data, BATCHSIZE, shuffle=False)
    model = getModel(name, ADJPATH)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS_pred_0 = predictModel(model, cal_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred_0.shape)
    YS, YS_pred_0 = np.squeeze(YS), np.squeeze(YS_pred_0)
    YS, YS_pred_0 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std'])

    # choose the best expected quantile
    # split some data
    split = int(YS.shape[0]* CALSPLIT)
    YS_train = YS[:split]
    YS_0_train = YS_pred_0[:split]
    YS_val = YS[split:]
    YS_0_val = YS_pred_0[split:]

    YS_1_train = YS_0_train * (YS_0_train>0)
    error = YS_train - YS_1_train

    cali_error = []
    for i in range(len(cal_list)):
        cali_error_ = np.quantile(error,cal_list[i])
        cali_error.append(cali_error_)

    err = [np.stack(cali_error),np.stack(cali_error)]
    independent_coverage_l = []
    for i in range(len(cal_list)):
        y_u_pred = YS_0_val + err[0][i]
        y_l_pred = YS_0_val - err[1][i]

        y_l_pred = y_l_pred * (y_l_pred>0)
        mask = YS_val>0
        independent_coverage = np.logical_and(np.logical_and(y_u_pred >= YS_val, y_l_pred <= YS_val), mask)
        m_coverage = np.sum(independent_coverage.astype(float))/np.sum(mask)
        independent_coverage_l.append(m_coverage)
    m_coverage = np.stack(independent_coverage_l)
    index = np.argmin(np.abs(m_coverage - 0.9))
    return  [err[0][index], err[1][index]]

def testModel(name, mode, XS, YS, scaler, err, ADJPATH):
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name, ADJPATH)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()

    YS, YS_pred_0= np.squeeze(YS), np.squeeze(predictModel(model, test_iter))
    YS, YS_pred_0 = Utils.inverse_transform(YS, scaler['mean'], scaler['std']), \
                               Utils.inverse_transform(YS_pred_0, scaler['mean'], scaler['std'])

    y_u_pred = YS_pred_0 + err[0]
    y_l_pred = YS_pred_0 - err[1]

    y_l_pred = y_l_pred * (y_l_pred > 0)
    mask = YS > 0
    independent_coverage = np.logical_and(np.logical_and(y_u_pred >= YS, y_l_pred <= YS), YS > 0)
    # compute the coverage and interval width
    results = {}
    results["Point predictions"] = np.array(YS)
    results["Upper limit"] = np.array(y_l_pred)
    results["Lower limit"] = np.array(y_u_pred)
    results["Confidence interval widths"] = np.abs(y_u_pred - y_l_pred) * mask
    results["Mean confidence interval widths"] = np.sum(results["Confidence interval widths"]) / \
                                                 np.sum(mask)
    results["Independent coverage indicators"] = independent_coverage
    results["Mean independent coverage"] = np.sum(independent_coverage.astype(float)) / np.sum(mask)

    results["Calbration error"] = np.mean(err)

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
parser.add_argument('--config',default='../configuration/PEMS03.conf',type = str, help = 'configuration file path')
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--uncer_m', type=str, default='conformal') # quantile/quantile_conformal/adaptive/dropout/bayesian
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
    trainx, trainy = getXSYS(data, 'TRAIN')
    # transform
    mean = trainx.mean()
    std = trainy.std()
    scaler = {'mean': mean, 'std': std}
    data = Utils.transform(data, scaler['mean'], scaler['std'])

    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS, scaler, ADJPATH)

    print(KEYWORD, 'calibration started', time.ctime())
    calXS, calYS = getXSYS(data, 'CAL')
    print('CAL XS.shape YS,shape', calXS.shape, calYS.shape)
    err = calModel(MODELNAME, 'cal', calXS, calYS, scaler, cal_list, ADJPATH)

    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS, scaler, err, ADJPATH)

    
if __name__ == '__main__':
    main()

