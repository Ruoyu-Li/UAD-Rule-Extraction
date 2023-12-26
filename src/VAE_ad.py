import sys
import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import pickle
from global_var import *
from normalize import *
from utils import *
from data_load import load_data

torch.manual_seed(SEED)

EPOCH = 50
BATCH_SIZE = 64
LR = 0.001
FPR = 0.005

class VAE(nn.Module):
    def __init__(self, n_feat, z_dim=10, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, z_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 4, n_feat * 2),
        )
        self.L = 10
        self.prior = Normal(0, 1)
        self.n_feat = n_feat
        self.z_dim = z_dim
    
    def predict(self, x) -> dict:
        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1) #both with size [batch_size, latent_size]
        # print(latent_sigma)
        latent_sigma = latent_sigma.clip(-1e+2)
        latent_sigma = softplus(latent_sigma)
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample([self.L])  # shape: [L, batch_size, latent_size]
        z = z.view(self.L * batch_size, self.z_dim)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = recon_sigma.clip(-1e+2)
        recon_sigma = softplus(recon_sigma)
        recon_mu = recon_mu.view(self.L, *x.shape)
        recon_sigma = recon_sigma.view(self.L, *x.shape)
        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def forward(self, x: torch.Tensor) -> dict:
        pred_result = self.predict(x)
        x = x.unsqueeze(0)  # unsqueeze to broadcast input across sample dimension (L)
        log_lik = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x).mean(
            dim=0)  # average over sample dimension
        log_lik = log_lik.mean(dim=0).sum()
        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        loss = kl - log_lik
        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)

    def recon_prob(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = self.predict(x)
        recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])
        x = x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)  # vector of shape [batch_size]
        return p

    def is_anomaly(self, x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        p = self.recon_prob(x)
        return p < alpha

    
def train_process(dataset, subset):
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    X_train, X_eval = X_train[y_train == 0], X_eval[y_eval == 0]
    n_feat = X_train.shape[1]
    print(X_train.shape)

    # normalizer = Normalizer()
    # normalizer = mm_Normalizer()
    normalizer = std_Normalizer()
    normalizer.fit(X_train)
    X_train, X_eval = normalizer.normalize(X_train), normalizer.normalize(X_eval)
    
    train_set = TensorDataset(torch.from_numpy(X_train).float())
    eval_set = TensorDataset(torch.from_numpy(X_eval).float())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, drop_last=True)

    vae = VAE(n_feat=n_feat).cuda(DEVICE)
    optimizer = torch.optim.Adam(vae.parameters(), lr=LR)

    for epoch in range(EPOCH):
        for i, (x, ) in enumerate(train_loader):
            x = x.cuda(DEVICE)
            loss = vae(x)
            loss_train = loss['loss']
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Epoch :', epoch, '|', f'train_loss:{loss_train.data}')

    vae.eval()
    prob_list = []
    with torch.no_grad():
        for i, (x, ) in enumerate(eval_loader):
            x = x.cuda(DEVICE)
            p = vae.recon_prob(x)
            prob_list.append(p)
    thres = torch.concat(prob_list).view(-1, ).quantile(1 - FPR).item()
    print(thres)

    torch.save(vae, os.path.join(MODEL_DIR, f'VAE_{dataset}_{subset}.model'))
    with open(os.path.join(MODEL_DIR, f'VAE_{dataset}_{subset}.norm'), 'wb') as f:
        pickle.dump(normalizer, f)
    with open(os.path.join(MODEL_DIR, f'VAE_{dataset}_{subset}.thres'), 'w') as f:
        f.write(str(thres))


def test_process(dataset, subset):
    vae = torch.load(os.path.join(MODEL_DIR, f'VAE_{dataset}_{subset}.model')).cuda(DEVICE)
    vae.eval()
    with open(os.path.join(MODEL_DIR, f'VAE_{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f'VAE_{dataset}_{subset}.thres'), 'r') as f:
        thres = float(f.read())

    X, y = load_data(dataset, subset, mode='test')
    X = normalizer.normalize(X)
    test_set = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, drop_last=True)

    tp, fp, tn, fn = 0, 0, 0, 0
    prob_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.cuda(DEVICE)
            p = vae.recon_prob(x).cpu()
            y_pred = (p > thres).view(-1, ).int().cpu()
            y[y != 0] = 1
            tp += TP(y, y_pred)
            fp += FP(y, y_pred)
            tn += TN(y, y_pred)
            fn += FN(y, y_pred)
            prob_list.extend(p.numpy())
            y_list.extend(y.numpy())
    tpr = (tp / (tp + fn)).item()
    fpr = (fp / (tn + fp)).item()
    auc = roc_auc_score(y_list, prob_list)
    print('TPR:', tpr)
    print('FPR:', fpr)
    print('AUC:', auc)
    save_result({'TPR': [tpr], 'FPR': [fpr], 'AUC': [auc]}, f'VAE_{dataset}_{subset}')
    

if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = sys.argv[2]
    subset = sys.argv[3]
    if mode == 'train':
        train_process(dataset, subset)
    elif mode == 'test':
        test_process(dataset, subset)
