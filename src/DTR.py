import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.metrics import roc_auc_score
import pickle
from global_var import *
from normalize import *
from utils import *
from data_load import load_data


FPR = 0.005

def train_process(dataset, subset):
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    X_train = X_train[y_train == 0]
    print(X_train.shape)

    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)
    
    model = DTR(max_depth=10, random_state=SEED)
    model.fit(X_train)

    with open(os.path.join(TARGET_MODEL_DIR, f'DTR_{dataset}_{subset}.model'), 'wb') as f:
        pickle.dump(model, f)
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)


def test_process(dataset, subset):
    with open(os.path.join(TARGET_MODEL_DIR, f'DTR_{dataset}_{subset}.model'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)

    X, y = load_data(dataset, subset, mode='test')
    X = normalizer.normalize(X)

    y_pred = model.predict(X)
    n_label = np.argmax(np.bincount(y_pred[y == 0]))
    print(n_label)
    tp = ((y != n_label) & (y_pred != n_label)).sum()
    fp = ((y == n_label) & (y_pred != n_label)).sum()
    tn = ((y == n_label) & (y_pred == n_label)).sum()
    fn = ((y != n_label) & (y_pred == n_label)).sum()
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)

    c = model.cluster_centers_[n_label]
    score = []
    for x in X:
        score.append(np.linalg.norm(x - c))
    auc = roc_auc_score(y, score)
    print('TPR:', tpr)
    print('FPR:', fpr)
    print('AUC:', auc)
    save_result({'TPR': [tpr], 'FPR': [fpr], 'AUC': [auc]}, f'DTR_{dataset}_{subset}')
    

if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = sys.argv[2]
    subset = sys.argv[3]
    if mode == 'train':
        train_process(dataset, subset)
    elif mode == 'test':
        test_process(dataset, subset)
