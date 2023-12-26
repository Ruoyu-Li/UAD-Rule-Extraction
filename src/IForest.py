import sys
import numpy as np
from sklearn.ensemble import IsolationForest as IForest
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score, f1_score
import pickle
from global_var import *
from normalize import *
from utils import *
from data_load import load_data


FPR_list = np.arange(0.02, 0.051, 0.002)

def train_process(dataset, subset):
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    X_train = X_train[y_train == 0]
    print(X_train.shape)

    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)

    best_model, best_score = None, 0
    for FPR in FPR_list:
        model = IForest(n_estimators=500, contamination=FPR, random_state=SEED)
        model.fit(X_train)
        y_pred = model.predict(X_eval)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        score = f1_score(y_eval, y_pred)
        if score > best_score:
            print('contamination', FPR, 'score', score)
            best_model = model
            best_score = score

    with open(os.path.join(TARGET_MODEL_DIR, f'IForest_{dataset}_{subset}.model'), 'wb') as f:
        pickle.dump(best_model, f)
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)


def test_process(dataset, subset):
    with open(os.path.join(TARGET_MODEL_DIR, f'IForest_{dataset}_{subset}.model'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)

    X, y = load_data(dataset, subset, mode='test')
    X = normalizer.transform(X)

    y[y != 0] = 1
    y_pred = model.predict(X)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    tp = TP(y, y_pred)
    fp = FP(y, y_pred)
    tn = TN(y, y_pred)
    fn = FN(y, y_pred)
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    score = - model.score_samples(X)
    auc = roc_auc_score(y, score)
    print('TPR:', tpr)
    print('FPR:', fpr)
    print('AUC:', auc)
    save_result({
        'dataset': dataset, 'subset': subset,
        'TPR': tpr, 'FPR': fpr, 'AUC': auc
        }, 'IForest')    

if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = sys.argv[2]
    subset = sys.argv[3]
    if mode == 'train':
        train_process(dataset, subset)
    elif mode == 'test':
        test_process(dataset, subset)
