import sys
sys.path.append('../')
import numpy as np
from sklearn.ensemble import IsolationForest as IForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import torch
import torch.nn as nn
from global_var import *
from normalize import *
from utils import *
from data_load import load_data
from AE import AutoEncoder
from VAE import VAE
import ExtBound
import KITree
import importlib


def print_metrics(y_true, y_pred, y_model):
    prec, rec, f1 = precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
    fid = (y_pred == y_model).sum() / y_pred.shape[0]
    fid_neg = ((y_pred == 0) & (y_model == 0)).sum() / (y_model == 0).sum()
    fid_pos = ((y_pred == 1) & (y_model == 1)).sum() / (y_model == 1).sum()
    tpr = TP(y_true, y_pred) / (y_true == 1).sum()
    tnr = TN(y_true, y_pred) / (y_true == 0).sum()
    print('prec:', prec, 'rec:', rec, 'f1:', f1, 'tpr', tpr, 'tnr', tnr, 'fid:', fid, 'fid_pos:', fid_pos, 'fid_neg:', fid_neg)
    return prec, rec, f1, tpr, tnr, fid


# null + EBE
def null_EBE(func, thres, X_train, score, X_test, y_test, y_model):
    ext = ExtBound.ExtBound(func, thres)
    ext.fit(X_train, score)
    ext.set_bound()
    y_pred = np.zeros(y_test.shape)
    for i, x in enumerate(X_test):
        y_pred[i] = ext.predict_sample(x)
    print('null + EBE:')
    print_metrics(y_test, y_pred, y_model)


# KMeans + EBE
def KMeans_EBE(func, thres, X_train, score, X_test, y_test, y_model, k=5):
    cluster = KMeans(n_clusters=k)
    y_cluster = cluster.fit_predict(X_train)
    ext_list = []
    for i in set(y_cluster):
        ext = ExtBound.ExtBound(func, thres)
        ext.fit(X_train, score)
        ext.set_bound()
        ext_list.append(ext)
    y_pred = np.zeros(y_test.shape)
    for i, x in enumerate(X_test):
        y_pred[i] = ext.predict_sample(x)
    print(f'KMeans(k={k}) + EBE:')
    print_metrics(y_test, y_pred, y_model)

# KMeans (to rules) + EBE
def KMeans_toRule_EBE(func, thres, X_train, score, X_test, y_test, y_model, k=5):
    cluster = KMeans(n_clusters=k)
    y_cluster = cluster.fit_predict(X_train)
    ext_list = []
    for i in set(y_cluster):
        ext = ExtBound.ExtBound(func, thres)
        idx = (y_cluster == i)
        X_cluster = X_train[idx]
        int_bound = np.stack([np.min(X_cluster, axis=0), np.max(X_cluster, axis=0)]).transpose()
        ext.fit(X_train[idx], score[idx], int_bound=int_bound)
        ext.set_bound()
        ext_list.append(ext)
    y_pred = np.zeros(y_test.shape)
    for i, x in enumerate(X_test):
        y_cluster = cluster.predict(x.reshape(1, -1))[0]
        y_pred[i] = ext_list[y_cluster].predict_sample(x)
    print(f'KMeans(k={k}) (to rules) + EBE:')
    print_metrics(y_test, y_pred, y_model)    

class ICTree(KITree.KITree):
    def fit(self, X, s, bound=None):
        self.data = X
        self.score = s
        if type(bound) != np.ndarray:
            self.bound = np.zeros((X.shape[1], 2))
            self.bound[:, 0] = -np.inf
            self.bound[:, 1] = np.inf
        else:
            self.bound = bound
        if self.require_split():
            self.score_norm = self.normalize_score(s)
            # self.cal_label()
            # matrix size: feature size * (sample size - 1)
            criterion_matrix = np.zeros((self.data.shape[1], self.data.shape[0] - 1))
            thres_matrix = np.zeros((self.data.shape[1], self.data.shape[0] - 1))
            # iterate and find feature_id and thres that give maximum criterion
            for i in range(self.data.shape[1]):
                sort_idx = np.argsort(self.data[:, i])
                for j in range(sort_idx.shape[0] - 1):
                    if self.data[sort_idx[j], i] == self.data[sort_idx[j+1], i]:
                        thres_matrix[i, j] = self.data[sort_idx[j], i]
                        criterion_matrix[i, j] = 0
                    else:
                        thres = int((self.data[sort_idx[j+1], i] + self.data[sort_idx[j], i]) / 2.)
                        thres_matrix[i, j] = thres
                        criterion_matrix[i, j] = self.cal_criterion(sort_idx, j)
            max_idx = np.where(criterion_matrix == np.max(criterion_matrix))
            self.feature_id = max_idx[0][0]
            self.feat_thres = thres_matrix[max_idx[0][0], max_idx[1][0]]

            # print('****** Now, the depth of kitree is = ', self.level, ' ****** ')

            # split data and start recursion
            left_idx = self.data[:, self.feature_id] <= self.feat_thres
            right_idx = self.data[:, self.feature_id] > self.feat_thres
            left_bound = self.update_bound(0)
            right_bound = self.update_bound(1)

            self.left = ICTree(self.func, self.func_thres, self.level + 1, n_beam=self.n_beam, rho=self.rho, eta=self.eta)
            self.right = ICTree(self.func, self.func_thres, self.level + 1, n_beam=self.n_beam, rho=self.rho, eta=self.eta)
            self.left.fit(self.data[left_idx], self.score[left_idx], left_bound)
            self.right.fit(self.data[right_idx], self.score[right_idx], right_bound)

        else:
            if self.data.shape[0] == 0:
                self.bound[:, 0] = np.inf
                self.bound[:, 1] = -np.inf
            else:
                self.bound[:, 0] = np.min(self.data, axis=0)
                self.bound[:, 1] = np.max(self.data, axis=0)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            curr = self
            while curr.feature_id != None:
                curr, _ = curr.next_node(x)
            flag = 0
            for k, x_dim in enumerate(x):
                if self.bound[k, 0] <= x_dim <= self.bound[k, 1]:
                    flag += 1
            y_pred[i] = int(flag == x.shape[0])
        return y_pred


def ICTree_hypercube(func, thres, X_train, score, X_test, y_test, y_model):
    KITree.KITree(func, thres)
    ictree = ICTree(func, thres)
    ictree.fit(X_train, score)
    y_pred = ictree.predict(X_test)
    print('ICTree + hypercube:')
    print_metrics(y_test, y_pred, y_model)


datasets = {
    'cicids_custom': [
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday'
    ],
    'toniot_custom': [
        'backdoor',
        'ddos',
        'dos',
        'injection',
        'mitm',
        'password',
        'runsomware',
        'scanning',
        'xss'
    ]
}

for d in datasets:
    for sub in datasets[d]:
        print(f'using {d}-{sub}')
        X, _, y, _ = load_data(d, sub, mode='train')
        X = X[y == 0]
        y = y[y == 0]
        X_test, y_test = load_data(d, sub, mode='test')
        print(X.shape[0], X_test.shape[0])

        # ae = torch.load(os.path.join(TARGET_MODEL_DIR, f'VAE_{d}_{sub}.model')).cuda(DEVICE)
        # ae.eval()
        # with open(os.path.join(NORMALIZER_DIR, f'{d}_{sub}.norm'), 'rb') as f:
        #     normalizer = pickle.load(f)
        # thres = ae.thres
        # X_train = normalizer.transform(X)
        # X_test = normalizer.transform(X_test)

        # func = lambda x: ae.score_samples(x)
        # score = func(X_train)
        # y_model = ae.predict(X_test)


        with open(os.path.join(TARGET_MODEL_DIR, f'OCSVM_{d}_{sub}.model'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(NORMALIZER_DIR, f'{d}_{sub}.norm'), 'rb') as f:
            normalizer = pickle.load(f)
        X_train = normalizer.transform(X)
        X_test = normalizer.transform(X_test)

        score = -model.score_samples(X_train)
        thres = -model.offset_
        func = lambda x: -model.score_samples(x)

        y_model = model.predict(X_test)
        y_model[y_model == 1] = 0
        y_model[y_model == -1] = 1


        kdt = KITree.KITree(func, thres)
        kdt.fit(X_train, score)

        y_pred = kdt.predict(X_test)
        print_metrics(y_test, y_pred, y_model)

        # null_EBE(func, thres, X_train, score, X_test, y_test, y_model)
        # KMeans_EBE(func, thres, X_train, score, X_test, y_test, y_model, 2)
        # KMeans_EBE(func, thres, X_train, score, X_test, y_test, y_model, 5)
        # KMeans_EBE(func, thres, X_train, score, X_test, y_test, y_model, 10)
        # ICTree_hypercube(func, thres, X_train, score, X_test, y_test, y_model)
        KMeans_toRule_EBE(func, thres, X_train, score, X_test, y_test, y_model, 2)
        KMeans_toRule_EBE(func, thres, X_train, score, X_test, y_test, y_model, 5)
        KMeans_toRule_EBE(func, thres, X_train, score, X_test, y_test, y_model, 10)
        KMeans_toRule_EBE(func, thres, X_train, score, X_test, y_test, y_model, 15)
        KMeans_toRule_EBE(func, thres, X_train, score, X_test, y_test, y_model, 20)
        KMeans_toRule_EBE(func, thres, X_train, score, X_test, y_test, y_model, 25)
        KMeans_toRule_EBE(func, thres, X_train, score, X_test, y_test, y_model, 30)
