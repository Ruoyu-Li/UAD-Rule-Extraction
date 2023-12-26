import sys
import numpy as np
import math
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

EPOCH = 50
BATCH_SIZE = 32
LR = 0.001
KL_WEIGHT = 0.005
# FPR = 0.005
FPR_list = np.arange(0.001, 0.051, 0.001)

class VAE(nn.Module):
    def __init__(self,
                 n_feat,
                 z_dim = 10,
                 hidden_dim = 32):
        super(VAE, self).__init__()

        self.z_dim = z_dim

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_var = nn.Linear(hidden_dim, z_dim)      

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 4, n_feat),
        )
        self.thres = 0


    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_func(self, *args):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + KL_WEIGHT * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
    
    def loss_func_each(self, *args):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        recons_loss = torch.square(input - recons).mean(axis=1)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        loss = recons_loss + KL_WEIGHT * kld_loss
        loss = torch.nan_to_num(loss, nan=100.)
        return loss.detach()
    
    def score_samples(self, X, cuda=True):
        if cuda:
            X = torch.from_numpy(X).cuda(DEVICE).float()
        else:
            X = torch.from_numpy(X).float()
        result = self.forward(X)
        return self.loss_func_each(*result).cpu().numpy().reshape(-1, )

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
    
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
            if i % 100 == 0:
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
    for FPR in FPR_list:
        thres = loss_neg.quantile(1 - FPR).item()
        y_pred = (loss_all > thres).view(-1, ).int().cpu()
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
    vae.thres = best_thres

    torch.save(vae, os.path.join(TARGET_MODEL_DIR, f'VAE_{dataset}_{subset}.model'))
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)
    # with open(os.path.join(MODEL_DIR, f'VAE_{dataset}_{subset}.thres'), 'w') as f:
    #     f.write(str(thres))


def test_process(dataset, subset):
    vae = torch.load(os.path.join(TARGET_MODEL_DIR, f'VAE_{dataset}_{subset}.model')).cuda(DEVICE)
    vae.eval()
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)
    thres = vae.thres
    # with open(os.path.join(MODEL_DIR, f'VAE_{dataset}_{subset}.thres'), 'r') as f:
    #     thres = float(f.read())

    X, y = load_data(dataset, subset, mode='test')
    X = normalizer.transform(X)
    test_set = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, drop_last=True)

    tp, fp, tn, fn = 0, 0, 0, 0
    loss_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.cuda(DEVICE)
            result = vae(x)
            loss = vae.loss_func_each(*result).cpu()
            y_pred = (loss > thres).view(-1, ).int().cpu()
            y[y != 0] = 1
            tp += TP(y, y_pred)
            fp += FP(y, y_pred)
            tn += TN(y, y_pred)
            fn += FN(y, y_pred)
            loss_list.extend(loss.numpy())
            y_list.extend(y.numpy())
    tpr = (tp / (tp + fn)).item()
    fpr = (fp / (tn + fp)).item()
    auc = roc_auc_score(y_list, loss_list)
    print('TPR:', tpr)
    print('FPR:', fpr)
    print('AUC:', auc)
    save_result({
        'dataset': dataset, 'subset': subset,
        'TPR': tpr, 'FPR': fpr, 'AUC': auc
        }, 'VAE')    

if __name__ == '__main__':
    mode = sys.argv[1]
    dataset = sys.argv[2]
    subset = sys.argv[3]
    if mode == 'train':
        train_process(dataset, subset)
    elif mode == 'test':
        test_process(dataset, subset)
