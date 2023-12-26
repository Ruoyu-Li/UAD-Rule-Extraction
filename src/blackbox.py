import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.ensemble import IsolationForest as IForest
from sklearn.metrics import roc_auc_score, f1_score
import pickle
from global_var import *
from normalize import *
from utils import *
from data_load import load_data
from AE import AutoEncoder
from VAE import VAE

torch.manual_seed(SEED)


def train_model(model_name, X_train, X_eval, y_eval, dataset, subset, save_model=False, **kwargs):
    if model_name == 'AE':
        model = train_AE(X_train, X_eval, y_eval, dataset, subset, save_model=save_model, **kwargs)
    elif model_name == 'VAE':
        model = train_VAE(X_train, X_eval, y_eval, dataset, subset, save_model=save_model, **kwargs)
    elif model_name == 'OCSVM':
        model = train_ocsvm(X_train, X_eval, y_eval, dataset, subset, save_model=save_model, **kwargs)
    elif model_name == 'IForest':
        model = train_iforest(X_train, X_eval, y_eval, dataset, subset, save_model=save_model, **kwargs)
    else:
        print('no such blackbox model provided')
        model = None
    return model


def train_AE(X_train, X_eval, y_eval, dataset, subset, save_model=False, **kwargs):
    try:
        out_ratio = kwargs['out_ratio']
    except:
        out_ratio = np.arange(0.001, 0.051, 0.001)
    EPOCH = 50
    BATCH_SIZE = 64
    LR = 0.001
    n_feat = X_train.shape[1]
    
    train_set = TensorDataset(torch.from_numpy(X_train).float())
    eval_set = TensorDataset(torch.from_numpy(X_eval).float(), torch.from_numpy(y_eval).float())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, drop_last=True)

    ae = AutoEncoder(n_feat=n_feat).cuda(DEVICE)
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    loss_func = nn.MSELoss().cuda(DEVICE)

    for epoch in range(EPOCH):
        for i, (x, ) in enumerate(train_loader):
            x = x.cuda(DEVICE)
            _, x_rec = ae(x)
            loss_train = loss_func(x, x_rec)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            if i % 100 == 0 and 'verbose' in kwargs:
                print('Epoch :', epoch, '|', f'train_loss:{loss_train.data}')

    ae.eval()
    mse_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_loader):
            x = x.cuda(DEVICE)
            _, x_rec = ae(x)
            mse = mse_each(x, x_rec)
            y[y != 0] = 1
            mse_list.append(mse)
            y_list.append(y)
    mse_all = torch.concat(mse_list).view(-1, )
    y_true = torch.concat(y_list).view(-1, )
    mse_neg = mse_all[y_true == 0]

    best_thres, best_score = None, 0
    for FPR in out_ratio:
        thres = mse_neg.quantile(1 - FPR).item()
        y_pred = (mse_all > thres).view(-1, ).int().cpu()
        tp = TP(y_true, y_pred)
        fp = FP(y_true, y_pred)
        fn = FN(y_true, y_pred)
        recall = tp / (tp + fn)
        prec = tp / (tp + fp)
        score = 2 * recall * prec / (recall + prec)
        if score > best_score:
            best_thres = thres
            best_score = score
    print('best score:', best_score)
    ae.thres = best_thres

    if save_model:
        torch.save(ae, os.path.join(TARGET_MODEL_DIR, f'AE_{dataset}_{subset}.model'))
    return ae


def train_VAE(X_train, X_eval, y_eval, dataset, subset, save_model=False, **kwargs):
    try:
        out_ratio = kwargs['out_ratio']
    except:
        out_ratio = np.arange(0.001, 0.051, 0.001)
    EPOCH = 50
    BATCH_SIZE = 32
    LR = 0.001
    n_feat = X_train.shape[1]
    
    train_set = TensorDataset(torch.from_numpy(X_train).float())
    eval_set = TensorDataset(torch.from_numpy(X_eval).float(), torch.from_numpy(y_eval).float())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, drop_last=True)

    vae = VAE(n_feat=n_feat).cuda(DEVICE)
    optimizer = torch.optim.Adam(vae.parameters(), lr=LR)

    for epoch in range(EPOCH):
        for i, (x, ) in enumerate(train_loader):
            x = x.cuda(DEVICE)
            result = vae(x)
            loss_train = vae.loss_func(*result)['loss']
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            if i % 100 == 0 and 'verbose' in kwargs:
                print('Epoch :', epoch, '|', f'train_loss:{loss_train.data}')

    vae.eval()
    loss_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_loader):
            x = x.cuda(DEVICE)
            result = vae(x)
            loss = vae.loss_func_each(*result)
            loss_list.append(loss)
            y[y != 0] = 1
            y_list.append(y)
    loss_all = torch.concat(loss_list).view(-1, )
    y_true = torch.concat(y_list).view(-1, )
    loss_neg = loss_all[y_true == 0]

    best_thres, best_score = None, 0
    for FPR in out_ratio:
        thres = loss_neg.quantile(1 - FPR).item()
        y_pred = (loss_all > thres).view(-1, ).int().cpu()
        tp = TP(y_true, y_pred)
        fp = FP(y_true, y_pred)
        fn = FN(y_true, y_pred)
        recall = tp / (tp + fn)
        prec = tp / (tp + fp)
        score = 2 * recall * prec / (recall + prec)
        if score > best_score:
            best_thres = thres
            best_score = score
    print('best score:', best_score)
    vae.thres = best_thres

    if save_model:
        torch.save(vae, os.path.join(TARGET_MODEL_DIR, f'VAE_{dataset}_{subset}.model'))
    return vae


def train_ocsvm(X_train, X_eval, y_eval, dataset, subset, save_model=False, **kwargs):
    try:
        out_ratio = kwargs['out_ratio']
        nu_list = kwargs['out_ratio']
    except:
        out_ratio = np.arange(0.001, 0.03, 0.002)
        nu_list = np.arange(0.001, 0.03, 0.002)

    best_param, best_auc = None, 0
    for nu_value in nu_list:
        model = OCSVM(nu=nu_value)
        model.fit(X_train)
        score = -model.score_samples(X_eval)
        auc = roc_auc_score(y_eval, score)
        if auc > best_auc:
            best_param = nu_value
            best_auc = auc

    model = OCSVM(nu=best_param)
    model.fit(X_train)
    score_all = -model.score_samples(X_eval)
    score_neg = score_all[y_eval == 0]    

    best_offset, best_score = None, 0
    for FPR in out_ratio:
        offset = np.quantile(score_neg, 1-FPR)
        y_pred = (score_all > offset).astype('int')
        score = f1_score(y_eval, y_pred)
        if score > best_score:
            best_offset = offset
            best_score = score
    print('best score:', best_score)
    model.offset_ = best_offset

    if save_model:
        with open(os.path.join(TARGET_MODEL_DIR, f'OCSVM_{dataset}_{subset}.model'), 'wb') as f:
            pickle.dump(model, f)
    return model


def train_iforest(X_train, X_eval, y_eval, dataset, subset, save_model=False, **kwargs):
    try:
        out_ratio = kwargs['out_ratio']
    except:
        out_ratio = np.arange(0.02, 0.051, 0.002)

    best_model, best_score = None, 0
    for FPR in out_ratio:
        model = IForest(n_estimators=500, contamination=FPR, random_state=SEED)
        model.fit(X_train)
        y_pred = model.predict(X_eval)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        score = f1_score(y_eval, y_pred)
        if score > best_score:
            best_model = model
            best_score = score
    print('best score:', best_score)

    if save_model:
        with open(os.path.join(TARGET_MODEL_DIR, f'IForest_{dataset}_{subset}.model'), 'wb') as f:
            pickle.dump(best_model, f)
    return model
