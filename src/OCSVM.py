import sys
import numpy as np
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.metrics import roc_auc_score, f1_score
import pickle
from global_var import *
from normalize import *
from utils import *
from data_load import load_data


# FPR = 0.001
FPR_list = np.arange(0.001, 0.03, 0.002)
nu_list = np.arange(0.001, 0.03, 0.002)

def train_process(dataset, subset):
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    X_train = X_train[y_train == 0]
    print(X_train.shape)

    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)

    model = OCSVM(nu=0.001)
    model.fit(X_train)
    score_all = -model.score_samples(X_eval)
    score_neg = score_all[y_eval == 0]

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
    for FPR in FPR_list:
        offset = np.quantile(score_neg, 1-FPR)
        y_pred = (score_all > offset).astype('int')
        score = f1_score(y_eval, y_pred)
        if score > best_score:
            print('FPR', FPR, 'score', score)
            best_offset = offset
            best_score = score
    model.offset_ = best_offset

    with open(os.path.join(TARGET_MODEL_DIR, f'OCSVM_{dataset}_{subset}.model'), 'wb') as f:
        pickle.dump(model, f)
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)


def test_process(dataset, subset):
    with open(os.path.join(TARGET_MODEL_DIR, f'OCSVM_{dataset}_{subset}.model'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)

    X, y = load_data(dataset, subset, mode='test')
    X = normalizer.transform(X)

    score = -model.score_samples(X)
    y_pred = (score > -model.offset_).astype('int')
    tp = TP(y, y_pred)
    fp = FP(y, y_pred)
    tn = TN(y, y_pred)
    fn = FN(y, y_pred)
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)# score_samples means the lower, the more abnormal

    auc = roc_auc_score(y, score)
    print('TPR:', tpr)
    print('FPR:', fpr)
    print('AUC:', auc)
    save_result({
        'dataset': dataset, 'subset': subset,
        'TPR': tpr, 'FPR': fpr, 'AUC': auc
        }, 'OCSVM')    

if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = sys.argv[2]
    subset = sys.argv[3]
    if mode == 'train':
        train_process(dataset, subset)
    elif mode == 'test':
        test_process(dataset, subset)
