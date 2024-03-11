import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import pickle
from global_var import *
from normalize import *
from utils import *
from data_load import load_data

torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 50
BATCH_SIZE = 64
LR = 0.001
# FPR = 0.005
FPR_list = np.arange(0.001, 0.051, 0.001)

class AutoEncoder(nn.Module):
    def __init__(self, n_feat):
        super(AutoEncoder, self).__init__()
        self.n_feat = n_feat
        self.encoder  =  nn.Sequential(
            nn.Linear(self.n_feat, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.n_feat),
            nn.LeakyReLU(0.2),
            # nn.Tanh()
        )
        self.thres = 0

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def loss_func(self, x, recons):
        return F.mse_loss(x, recons)

    def loss_func_each(self, x, x_rec):
        return torch.square(x - x_rec).mean(dim=1).detach()
    
    def score_samples(self, X, cuda=True):
        if cuda:
            X = torch.from_numpy(X).to(DEVICE).float()
        else:
            X = torch.from_numpy(X).float()
        _, recons = self.forward(X)
        return self.loss_func_each(X, recons).cpu().numpy().reshape(-1, )
    
    def predict(self, X):
        return (self.score_samples(X) > self.thres).astype(int)


def train_process(dataset, subset):
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    # X_train, X_eval = X_train[y_train == 0], X_eval[y_eval == 0]
    n_feat = X_train.shape[1]
    print(X_train.shape)

    # normalizer = Normalizer()
    # normalizer = mm_Normalizer()
    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)
    
    train_set = TensorDataset(torch.from_numpy(X_train).float())
    eval_set = TensorDataset(torch.from_numpy(X_eval).float(), torch.from_numpy(y_eval).float())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, drop_last=True)

    ae = AutoEncoder(n_feat=n_feat).to(DEVICE)
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    loss_func = nn.MSELoss().to(DEVICE)

    for epoch in range(EPOCH):
        for i, (x, ) in enumerate(train_loader):
            x = x.to(DEVICE)
            _, x_rec = ae(x)
            loss_train = loss_func(x, x_rec)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # with torch.no_grad():
            #     for j, (x, ) in enumerate(eval_loader):
            #         x = x.to(DEVICE)
            #         _, x_rec = ae(x)
            #         loss_eval = loss_func(x, x_rec)
            if i % 100 == 0:
                # print('Epoch :', epoch, '|', f'train_loss:{loss_train.data}', '|', f'eval_loss:{loss_eval.data}')
                print('Epoch :', epoch, '|', f'train_loss:{loss_train.data}')

    ae.eval()
    mse_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_loader):
            x = x.to(DEVICE)
            _, x_rec = ae(x)
            mse = mse_each(x, x_rec)
            y[y != 0] = 1
            mse_list.append(mse)
            y_list.append(y)
    mse_all = torch.concat(mse_list).view(-1, )
    y_true = torch.concat(y_list).view(-1, )
    mse_neg = mse_all[y_true == 0]

    best_thres, best_score = None, 0
    for FPR in FPR_list:
        thres = mse_neg.quantile(1 - FPR).item()
        y_pred = (mse_all > thres).view(-1, ).int().cpu()
        tp = TP(y_true, y_pred)
        fp = FP(y_true, y_pred)
        fn = FN(y_true, y_pred)
        recall = tp / (tp + fn)
        prec = tp / (tp + fp)
        score = 2 * recall * prec / (recall + prec)
        if score > best_score:
            print('FPR', FPR, 'score', score)
            best_thres = thres
            best_score = score
    ae.thres = best_thres

    torch.save(ae, os.path.join(TARGET_MODEL_DIR, f'AE_{dataset}_{subset}.model'))
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)
    # with open(os.path.join(MODEL_DIR, f'AE_{dataset}_{subset}.thres'), 'w') as f:
    #     f.write(str(thres))


def test_process(dataset, subset):
    ae = torch.load(os.path.join(TARGET_MODEL_DIR, f'AE_{dataset}_{subset}.model')).to(DEVICE)
    ae.eval()
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)
    thres = ae.thres
    # with open(os.path.join(MODEL_DIR, f'AE_{dataset}_{subset}.thres'), 'r') as f:
    #     thres = float(f.read())

    X, y = load_data(dataset, subset, mode='test')
    X = normalizer.transform(X)
    test_set = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, drop_last=True)

    tp, fp, tn, fn = 0, 0, 0, 0
    mse_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            _, x_rec = ae(x)
            mse = mse_each(x, x_rec).cpu()
            y_pred = (mse > thres).view(-1, ).int().cpu()
            y[y != 0] = 1
            tp += TP(y, y_pred)
            fp += FP(y, y_pred)
            tn += TN(y, y_pred)
            fn += FN(y, y_pred)
            mse_list.extend(mse.numpy())
            y_list.extend(y.numpy())
    tpr = (tp / (tp + fn)).item()
    fpr = (fp / (tn + fp)).item()
    auc = roc_auc_score(y_list, mse_list)
    print('TPR:', tpr)
    print('FPR:', fpr)
    print('AUC:', auc)
    save_result({
        'dataset': dataset, 'subset': subset,
        'TPR': tpr, 'FPR': fpr, 'AUC': auc
        }, 'AE')
    

if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = sys.argv[2]
    subset = sys.argv[3]
    if mode == 'train':
        train_process(dataset, subset)
    elif mode == 'test':
        test_process(dataset, subset)
