import numpy as np


class ExtBound():
    def __init__(self, func, thres, max_iter=100, n_beam=10, n_sampling=50, rho=0.3, eta=0.1, eps=0.01):
        self.func = func
        self.thres = thres
        # bound: [X.shape[1], 2]
        self.int_bound = None
        self.ext_bound = None

        self.n_sampling = n_sampling
        self.n_beam = n_beam
        self.rho = rho
        self.eta = eta
        self.eps = eps
        self.max_iter = max_iter

    def fit(self, X, s, int_bound=None):
        self.init_bound(int_bound, X.shape[1])

        X_inlier, X_outlier = X[s <= self.thres], X[s > self.thres]
        s_inlier, s_outlier = s[s <= self.thres], s[s > self.thres]
        if X_inlier.shape[0] == 0:
            self.ext_bound[:, 1] = -np.inf
            # print('no legitimate samples here')
            return

        radius = self.get_sampling_radius(X_inlier)
        cov = self.get_cov(radius)
        for j in range(X.shape[1]):
            for d in range(2):
                # anchor: [n_beam, X.shape[1]]; s_anchor: [n_beam, ]
                X_anchor = self.get_init_anchor(X_inlier, j, d)
                X_anchor_init = np.copy(X_anchor)
                s_anchor = self.func(X_anchor).reshape(self.n_beam, )
                s_anchor_init = np.copy(s_anchor)
                k = 1
                # flag_break: 0 -> max_iter; 1 -> thres; -1 -> all reach int_bound; -2 -> all go backward
                flag_break = 0
                # an anchor is alive utill it reaches int_bound
                num_alive = self.n_beam

                while k <= self.max_iter:
                    # sample around anchors
                    X_sample = np.zeros((num_alive, self.n_sampling, X.shape[1]))
                    s_sample = np.zeros((num_alive, self.n_sampling))
                    for n in range(num_alive):
                        X_sample[n] = self.explorer_sampling(X_anchor[n], cov, j, d)
                        # s_sample[n] = self.func(X_sample[n]).reshape(self.n_sampling, )
                    s_sample = self.func(X_sample.reshape(-1, X.shape[1])).reshape(num_alive, self.n_sampling)

                    # find samples with largest score and calculate grad to obtain next anchors
                    # idx_flat = np.argsort(s_sample.ravel())[-num_alive:]
                    # pos = np.unravel_index(idx_flat, s_sample.shape)
                    # X_sample_max, s_sample_max = X_sample[pos], s_sample[pos]
                    idx_max = np.argsort(s_sample, axis=1)[:, -1]
                    X_sample_max = X_sample[np.arange(num_alive), idx_max]
                    s_sample_max = s_sample[np.arange(num_alive), idx_max]
                    grad = (s_sample_max - s_anchor) / (X_sample_max - X_anchor)[:, j]
                    grad = self.grad_sign(grad)
                    X_anchor_next = (X_anchor + X_sample_max) / 2.
                    X_anchor_next[:, j] = X_anchor_next[:, j] + grad * self.eta

                    # check if anchors reach int_bound
                    flag_int = self.check_reach_int_bound(X_anchor_next, j, d)
                    if (flag_int == -1).all():
                        flag_break = -1
                        break

                    # check if anchors go backward
                    flag_bwd = self.check_go_backward(X_anchor, X_anchor_next, j, d)
                    if (flag_bwd == -2).all():
                        flag_break = -2
                        break

                    filt_cond = (flag_int == 0)
                    num_alive = filt_cond.sum()
                    X_anchor_next = X_anchor_next[filt_cond]

                    # check if anchors reach thres
                    s_anchor_next = self.func(X_anchor_next).reshape(num_alive, )
                    flag_thres = self.check_reach_thres(s_anchor_next)
                    if (flag_thres == 1).any():
                        flag_break = 1
                        x_inbound = X_anchor[filt_cond][s_anchor_next == np.max(s_anchor_next)]
                        x_outbound = X_anchor_next[s_anchor_next == np.max(s_anchor_next)]
                        self.ext_bound[j, d] = (x_inbound[0, j] + x_outbound[0, j]) / 2.
                        break
                    
                    X_anchor = X_anchor_next
                    s_anchor = s_anchor_next
                    k += 1
                
                # check if anchors alive move enough distance
                if flag_break == 0:
                    if self.check_anchor_move(s_anchor, s_anchor_init):
                        self.ext_bound[j, d] = X_anchor[:,  j].min() if d == 0 else X_anchor[:,  j].max()
                # print('dim:', j, 'direction:', {0: 'lower', 1: 'upper'}[d], 
                #       'flag_break:', {0: 'max_iter', 1: 'reach thres', -1: 'reach int_bound', -2: 'go backward'}[flag_break], 
                #       'value:', self.ext_bound[j, d])
                        
    def init_bound(self, int_bound, n_dim):
        if type(int_bound) != np.ndarray:
            self.int_bound = np.zeros((n_dim, 2))
            self.int_bound[:, 0] = -np.inf
            self.int_bound[:, 1] = np.inf
        else:
            self.int_bound = int_bound
        self.ext_bound = np.zeros((n_dim, 2))
        self.ext_bound[:, 0] = -np.inf
        self.ext_bound[:, 1] = np.inf

    def get_init_anchor(self, X, dim, dr):
        anchor = np.zeros((self.n_beam, X.shape[1]))
        if dr == 0:
            anchor[0] = X[np.argmin(X[:, dim])]
            anchor[1:, dim] = np.min(X[:, dim])
        else:
            anchor[0] = X[np.argmax(X[:, dim])]
            anchor[1:, dim] = np.max(X[:, dim])
        for j in range(X.shape[1]):
            if j == dim:
                continue
            else:
                anchor[1:, j] = np.random.uniform(np.min(X[:, j]), np.max(X[:, j]), self.n_beam - 1)
        return anchor
    
    def get_sampling_radius(self, X):
        radius = (np.max(X, axis=0) - np.min(X, axis=0)) * self.rho
        radius[radius == 0] = self.rho
        radius[:] = self.rho
        return radius
    
    def get_cov(self, radius):
        cov = np.zeros((radius.size, radius.size))
        for i in range(radius.size):
            cov[i, i] = radius[i] / 3
        return cov
        
    def explorer_sampling(self, x, cov, dim, dr):
        X_sample = np.random.multivariate_normal(x, cov, size=self.n_sampling)
        diff = np.abs(X_sample[:, dim] - x[dim])
        if dr == 0:
            X_sample[:, dim] = x[dim] - diff
        else:
            X_sample[:, dim] = x[dim] + diff
        return X_sample
    
    def grad_sign(self, grad):
        grad[grad > 0] = 1
        grad[grad < 0] = -1
        return grad
    
    def check_reach_int_bound(self, X, dim, dr):
        flag = np.zeros(X.shape[0])
        bound = self.int_bound[dim, dr]
        if dr == 0:
            flag[np.where(X[:, dim] <= bound)] = -1
        else:
            flag[np.where(X[:, dim] > bound)] = -1
        return flag
    
    def check_go_backward(self, X, X_next, dim, dr):
        flag = np.zeros(X.shape[0])
        diff = X_next[:, dim] - X[:, dim]
        if dr == 0:
            flag[np.where(diff > 0)] = -1
        else:
            flag[np.where(diff < 0)] = -1
        return flag
    
    def check_reach_thres(self, s):
        flag = np.zeros(s.shape[0])
        flag[np.where(s <= self.thres)] = 0
        flag[np.where(s > self.thres)] = 1
        return flag
    
    def check_anchor_move(self, s, s_init):
        diff = (s.max() - s_init.max()) / s_init.max()
        if diff < self.eps:
            return 0
        else:
            return 1
    
    def set_bound(self):
        if (self.ext_bound[:, 1] == -np.inf).all():
            self.bound = self.ext_bound
        self.bound = np.zeros(self.int_bound.shape)
        n_dim = self.int_bound.shape[0]
        for dim in range(n_dim):
            self.bound[dim, 0] = max(self.int_bound[dim, 0], self.ext_bound[dim, 0])
            self.bound[dim, 1] = min(self.int_bound[dim, 1], self.ext_bound[dim, 1])
        
    def get_bound(self):
        return self.bound
    
    def predict_sample(self, x):
        result = ((x > self.bound[:, 0]) & (x <= self.bound[:, 1])).sum()
        y_pred = int(result != x.shape[0])
        return y_pred
