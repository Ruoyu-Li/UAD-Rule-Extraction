import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append('../')
from normalize import *
from utils import *
from data_load import load_data
from blackbox import train_model
import ExtBound
import KITree


datasets = {
    # 'cicids_custom': [
    #     'Tuesday',
    #     'Wednesday',
    #     'Thursday',
    #     'Friday'
    # ],
    # 'toniot_custom': [
    #     'backdoor',
    #     'ddos',
    #     'dos',
    #     'injection',
    #     'mitm',
    #     'password',
    #     'runsomware',
    #     'scanning',
    #     'xss'
    # ],
    'cicids_improved': [
        'tuesday',
        'wednesday',
        'thursday',
        'friday'     
    ]
}

model_names = [
    'AE', 
    'VAE', 
    'OCSVM', 
    'IForest'
]

noise_props = [
    0,
    0.01, 
    0.03, 
    0.05, 
    0.1
]

def evaluate_blackbox(y_true, y_model):
    conf_matrix = confusion_matrix(y_true, y_model)
    tn, fp, fn, tp = conf_matrix.ravel()
    tnr, fpr, fnr, tpr = tn/(tn+fp), fp/(tn+fp), fn/(fn+tp), tp/(fn+tp)
    prec= precision_score(y_true, y_model)
    rec = recall_score(y_true, y_model)
    f1 = f1_score(y_true, y_model)
    return {
        'blackbox_tnr': tnr,
        'blackbox_fpr': fpr,
        'blackbox_fnr': fnr,
        'blackbox_tpr': tpr,
        'blackbox_prec': prec,
        'blackbox_rec': rec,
        'blackbox_f1': f1,
    }

def evaluate_surrogate(y_true, y_model, y_pred, y_pred_perturb):
    fid = (y_model == y_pred).sum() / y_model.shape[0]
    rb = (y_pred == y_pred_perturb).sum() / y_pred.shape[0]
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    tnr, fpr, fnr, tpr = tn/(tn+fp), fp/(tn+fp), fn/(fn+tp), tp/(fn+tp)
    prec= precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
        'fidelity': fid,
        'robustness': rb,
        'surrogate_tnr': tnr,
        'surrogate_fpr': fpr,
        'surrogate_fnr': fnr,
        'surrogate_tpr': tpr,
        'surrogate_prec': prec,
        'surrogate_rec': rec,
        'surrogate_f1': f1,
    }    

def random_noise(X, prop):
    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    diff = X_max - X_min
    X_rand = np.random.uniform(size=(int(X.shape[0] * prop), X.shape[1]))
    X_noise = X_rand * diff + X_min
    X_new = np.concatenate([X, X_noise])
    np.random.shuffle(X_new)
    return X_new

def permute_noise(X, prop):
    idx = np.random.choice(X.shape[0], int(X.shape[0] * prop), replace=False)
    for k in range(X.shape[1]):
        np.random.shuffle(X[idx, k])
    return X

def attack_noise(X, X_test, y_test, prop):
    X_att = X_test[y_test != 0]
    try:
        idx = np.random.choice(X_att.shape[0], int(X.shape[0] * prop), replace=False)
    except:
        idx = np.arange(X_att.shape[0])
    X_new = np.concatenate([X, X_att[idx]])
    return X_new

for dataset in datasets:
    for subset in datasets[dataset]:
        X_train, X_eval, _, y_eval = load_data(dataset, subset, 'train', random_select=1000)
        X_test, y_test = load_data(dataset, subset, 'test')
        for prop in noise_props:
            # X_train = random_noise(X_train, prop)
            # X_train = permute_noise(X_train, prop)
            X_train = attack_noise(X_train, X_test, y_test, prop)
            normalizer = StandardScaler()
            normalizer.fit(X_train)
            X_train, X_eval, X_test = normalizer.transform(X_train), normalizer.transform(X_eval), normalizer.transform(X_test)
            for model_name in model_names:
                print('dataset', dataset, 'subset', subset, 'prop', prop, 'model', model_name)

                out_ratio = np.arange(0.001, max(0.05, prop), 0.003)
                model = train_model(model_name, X_train, X_eval, y_eval, dataset, subset, out_ratio=out_ratio)

                if model_name == 'AE' or model_name == 'VAE':
                    model.eval()
                    thres_ = model.thres
                    func_ = lambda x: model.score_samples(x)
                else:
                    thres_ = -model.offset_
                    func_ = lambda x: -model.score_samples(x)
                y_model = (func_(X_test) > thres_).astype(int)
                result_blackbox = evaluate_blackbox(y_test, y_model)
                
                score_ = func_(X_train)
                kdt_ = KITree.KITree(func_, thres_)
                kdt_.fit(X_train, score_)
                y_pred = kdt_.predict(X_test)
                
                X_perturb = np.zeros(X_test.shape)
                idx = np.ones(X_test.shape[0]).astype(bool)
                delta = 0.01  # Define perturbation magnitude
                scale = 1
                while scale >= 0:
                    X_perturb[idx] = X_test[idx] + delta * scale
                    y_model_perturb = (func_(X_perturb) > thres_).astype(int)
                    idx = (y_model != y_model_perturb)
                    if idx.sum() > 0:
                        scale -= 0.1
                    else:
                        break
                y_pred_perturb = kdt_.predict(X_perturb)
                result_surrogate = evaluate_surrogate(y_test, y_model, y_pred, y_pred_perturb)

                save_result(
                    dict({'dataset': dataset, 'subset': subset, 'model': model_name, 'prop': prop},
                         **result_blackbox,
                         **result_surrogate
                    ),
                    # 'noise_rand',
                    # 'noise_perm',
                    'noise_att'
                )
