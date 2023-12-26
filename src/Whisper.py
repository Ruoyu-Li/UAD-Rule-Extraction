import sys
import numpy as np
from scipy.fftpack import fft
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import pickle
from global_var import *
from normalize import *
from utils import *
from data_load import load_data
import torch
from sklearn.metrics import roc_auc_score, roc_curve

torch.manual_seed(SEED)

EPOCH = 50
BATCH_SIZE = 32
LR = 0.001
FPR = 0.005

class Whisper():
    def __init__(self, n_cluster):
        self.n_cluster = n_cluster
        self.estimator = KMeans(n_clusters=n_cluster)
        self.threshold = None

    def frequency_analysis(self, x_train):
        freq_list = []
        for data in x_train:
            freq_list.append(np.abs(fft(np.asarray(data))))  # 不取对数
            # log_Y.append(np.log(np.abs(fft(np.asarray(data)))))  # todo 论文里说取对数有用，但实测下来，不取对数的AUC能高4、5个点
        return freq_list


    # 计算欧拉距离
    def calcDis(self, dataSet, centroids):
        clalist = []
        for data in dataSet:
            diff = np.array(data) - np.array(centroids)  # 相减
            squaredDiff = diff ** 2  # 平方
            squaredDist = np.sum((squaredDiff))  # 和  (axis=1表示行)
            distance = squaredDist ** 0.5  # 开根号
            clalist.append(distance)
        clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
        return clalist

    # 计算质心
    def classify(self, dataSet, centroids, k):
        # 计算样本到质心的距离
        clalist = self.calcDis(dataSet, centroids, k)

        # 分组并计算新的质心
        # minDistIndices = np.argmin(clalist)  # axis=1 表示求出每行的最小值的下标

        newCentroids = (dataSet.mean())

        changed = newCentroids - centroids
        print('changed = ',changed)

        return changed, newCentroids

    # 创建数据集
    def pred_isBenign(self, x_test,centroids):
        dis = self.calcDis(x_test, centroids)
        return dis
        # return True if dis <= mean else False # x_train 中良性的

    def get_score(self, y_test,mses,threshold):
        B_T_num = 0
        M_T_num = 0
        B_F_num = 0
        M_F_num = 0
        B_num = 0
        M_num = 0
        length = len(y_test)
        for i in range(len(y_test)):
            if mses[i] < threshold and y_test[i] == 0:
                B_T_num = B_T_num+1
                B_num = B_num+1
            if mses[i] >= threshold and y_test[i] == 0:
                B_F_num = B_F_num+1
                B_num = B_num+1
            if mses[i] >= threshold and y_test[i] == 1:
                M_T_num = M_T_num + 1
                M_num = M_num+1
            if mses[i] < threshold and y_test[i] == 1:
                M_F_num = M_F_num + 1
                M_num = M_num+1
        print('B_T_num = ',B_T_num,'B_F_num = ',B_F_num,'M_T_num = ',M_T_num,'M_F_num = ',M_F_num,'B_num = ',B_num,'M_num = ',M_num)

        print('B_T_num = ',B_T_num/B_num,'B_F_num = ',B_F_num/B_num,'M_T_num = ',M_T_num/M_num,'M_F_num = ',M_F_num/M_num)
        # return np.sum(y_predict == y_test) / len(y_predict)

    # 计算 mse
    def MSE(self, x_decode, x):
        eps = x - x_decode
        out = torch.mean(eps ** 2)
        return out

    def predict(self, x):
        dis, mean = self.pred_isBenign(x, self.centroids, self.mean_dis)
        return 0 if dis <= mean else 1

    def score_samples(self, x):
        dis = self.pred_isBenign(x, self.centroids)
        return dis
    
    # 计算平均距离
    def calc_mean_dis(self, X, centroids):
        dis = self.calcDis(X, centroids)
        mean_dis = np.mean(dis)
        return mean_dis

    def fit(self, X_train ,y_train):
        X_benign = self.frequency_analysis(X_train)
        self.estimator.fit(X_benign)
        self.centroids = self.estimator.cluster_centers_
        self.mean_dis = self.calc_mean_dis(X_benign, self.centroids)

        if self.threshold is None:
            fprs, tprs, thresholds = roc_curve(y_train, self.score_samples(X_train))
            optimal_idx = np.argmin(np.abs(fprs - FPR))
            self.threshold = thresholds[optimal_idx]

    def predict(self, X_test):
        X_freq = self.frequency_analysis(X_test)
        dis = self.calcDis(X_freq, self.centroids)
        return np.array([0 if d <= self.threshold else 1 for d in dis])


    def score_samples(self, X_test):
        X_freq = self.frequency_analysis(X_test)
        dis = self.calcDis(X_freq, self.centroids)
        return dis

def train_process(dataset, subset):
    # Load and preprocess data
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    print("X_train.shape",X_train.shape, " | y_train.shape",y_train.shape, " | X_eval.shape",X_eval.shape, " | y_eval.shape",y_eval.shape)

    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)

    whisper = Whisper(n_cluster=10)
    whisper.fit(X_train, y_train)

    # Save the trained model and normalizer
    with open(os.path.join(TARGET_MODEL_DIR, f'Whisper_{dataset}_{subset}.model'), 'wb') as f:
        pickle.dump(whisper, f)
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)

def test_process(dataset, subset):
    # Load trained model and normalizer
    with open(os.path.join(TARGET_MODEL_DIR, f'Whisper_{dataset}_{subset}.model'), 'rb') as f:
        whisper = pickle.load(f)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)

    X_test, y_test = load_data(dataset, subset, mode='test')
    X_test = normalizer.transform(X_test)

    y_pred = whisper.predict(X_test)
    scores = whisper.score_samples(X_test)

    tp = TP(y_test, y_pred)
    fp = FP(y_test, y_pred)
    tn = TN(y_test, y_pred)
    fn = FN(y_test, y_pred)
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    auc = roc_auc_score(y_test, scores)
    print('TPR:', tpr)
    print('FPR:', fpr)
    print('AUC:', auc)
    save_result({
        'dataset': [dataset], 'subset': [subset],
        'TPR': [tpr], 'FPR': [fpr], 'AUC': [auc]
        }, 'OCSVM')  


if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = sys.argv[2]
    subset = sys.argv[3]
    if mode == 'train':
        train_process(dataset, subset)
    elif mode == 'test':
        test_process(dataset, subset)