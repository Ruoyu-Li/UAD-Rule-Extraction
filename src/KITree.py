import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
from ExtBound import ExtBound
import utils
import re
from global_var import *

class KITree():
    """
    max_level: maxmum level of a tree
    gap_limit: minimum gap between model predictions for splitting
    max_iter: maximum iteration of ExtBound
    n_beam: number of explorers
    n_sampling: number of auxiliary explorers for each explorer
    rho: coefficient of variance, radius of sampling
    eta: factor of iteration stride
    eps: minimum change of model prediction to judge contour line
    """ 
    def __init__(self, func, func_thres, level=0, max_level=5, gap_limit=-np.inf, 
                 max_iter=100, n_beam=10, n_sampling=50, rho=0.3, eta=0.1, eps=0.01):
        self.func = func
        self.func_thres = func_thres
        self.data = None
        self.score = None
        self.score_norm = None
        self.level = level

        self.max_level = max_level
        self.gap_limit = gap_limit
        self.max_iter = max_iter
        self.n_beam = n_beam
        self.n_sampling = n_sampling
        self.rho = rho
        self.eta = eta
        self.eps = eps

        self.left = None
        self.right = None
        self.feature_id = None
        self.feat_thres = None
        self.FP_examples = None

        self.bound = None
    
    def fit(self, X, s, bound=None):
        """
        train a tree model
        """
        FQA_outfile = open("FQA_outfile.txt", "a")  # 打开文件以便写入

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

            self.left = KITree(self.func, self.func_thres, self.level + 1, self.max_level, self.gap_limit, self.max_iter, self.n_beam, self.n_sampling, self.rho, self.eta, self.eps)
            self.right = KITree(self.func, self.func_thres, self.level + 1, self.max_level, self.gap_limit, self.max_iter, self.n_beam, self.n_sampling, self.rho, self.eta, self.eps)
            self.left.fit(self.data[left_idx], self.score[left_idx], left_bound)
            self.right.fit(self.data[right_idx], self.score[right_idx], right_bound)

        else:
            # print('start ExtBound; number of samples:', X.shape[0])
            self.ext = ExtBound(self.func, self.func_thres, self.max_iter, self.n_beam, self.n_sampling, self.rho, self.eta, self.eps)
            self.ext.fit(X, s, self.bound)
            self.ext.set_bound()

    def normalize_score(self, score):
        """
        New: 每次分裂前对score标准化然后使用Sigmoid变为0-1的数值
        """
        score_norm = StandardScaler().fit_transform(score.reshape(-1, 1)).reshape(-1, )
        return 1 / (1 + np.exp(-score_norm))

    def require_split(self):
        """
        determine if the anomaly scores can be splitted by clustering (stop criterion)
        """
        if type(self.score) != np.ndarray:
            return False
        elif self.data.shape[0] < 2:
            return False
        elif self.level >= self.max_level:
            return False
        else:
            self.score_norm = self.normalize_score(self.score)
            sort_score = np.sort(self.score_norm)
            for i in range(sort_score.shape[0] - 1):
                if sort_score[i + 1] - sort_score[i] > self.gap_limit:
                    return True
            return False
    
    def cal_criterion(self, sort_idx, sp, criterion='gini'):
        """
        calculate splitting criterion given feature_id and threshold
        """
        gini_root = self.cal_soft_gini(range(self.score_norm.shape[0]))
        gini_left = self.cal_soft_gini(sort_idx[:sp+1]) * (sp + 1) / sort_idx.shape[0]
        gini_right = self.cal_soft_gini(sort_idx[sp+1:]) * (sort_idx.shape[0] - sp - 1) / sort_idx.shape[0]
        return gini_root - (gini_left + gini_right)
    
    def cal_soft_gini(self, idx):
        L = self.score_norm[idx]
        p = L.sum() / L.shape[0]
        gini = 1 - p**2 - (1 - p)**2
        return gini
    
    def update_bound(self, dr):
        new_bound = self.bound.copy()
        if dr == 0:
            new_bound[self.feature_id, 1] = min(new_bound[self.feature_id, 1], self.feat_thres)
        else:
            new_bound[self.feature_id, 0] = max(new_bound[self.feature_id, 0], self.feat_thres)
        return new_bound
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            curr = self
            while curr.feature_id != None:
                curr, _ = curr.next_node(x)
            y_pred[i] = curr.ext.predict_sample(x)
        return y_pred
    
    def predict_rule(self, x, normalizer, feat_list=None):
        """
        print decision paths of given sample
        """
        if feat_list == None:
            feat_list = range(x.shape[0])
        curr = self
        while curr.feature_id != None:
            nx, indicator = curr.next_node(x)
            if indicator == 0:
                symbol = '<='
            else:
                symbol = '>'
            thres = utils.inverse_norm(normalizer, curr.feature_id, curr.feat_thres)
            subrule = f'dim {curr.feature_id} feat {feat_list[curr.feature_id]} {symbol} {thres}'
            print(' ' * (curr.level), end='')
            print(f'-> {subrule}', end='\n')
            curr = nx
        for i, bound in enumerate(curr.ext.ext_bound):
            x_value = utils.inverse_norm(normalizer, i, x[i])
            try:
                upper_value = utils.inverse_norm(normalizer, i, bound[1])
            except:
                upper_value = bound[1]
            try:
                lower_value = utils.inverse_norm(normalizer, i, bound[0])
            except:
                lower_value = bound[0]
            if x[i] >= bound[1]:
                print(f'dim {i} feat {feat_list[i]} exceed upper bound: {x_value} >= {upper_value}')
            if x[i] < bound[0]:
                print(f'dim {i} feat {feat_list[i]} under lower bound: {x_value} < {lower_value}')

    def next_node(self, x):
        """
        determine the next node (decision path)
        """
        indicator = 0
        if x[self.feature_id] <= self.feat_thres:
            curr = self.left
            indicator = 0
        else:
            curr = self.right
            indicator = 1
        return curr, indicator
    
    def get_depth(self):
        """
        recursively get maximum depth of the tree
        """
        if self.left != None:
            left_depth = self.left.get_depth()
        else:
            return self.level
        if self.right != None:
            right_depth = self.right.get_depth()
        else:
            return self.level
        return max(left_depth, right_depth)
    
    def get_rule_num(self):
        """
        recursively get total rule number (from root to a leaf)
        """
        if self.left != None:
            left_rule = self.left.get_rule_num()
            right_rule = self.right.get_rule_num()
            return left_rule + right_rule
        else:
            return 1
    
    def get_rules(self, feat_list, normalizer, rule=[]):
        """
        print all rules
        """
        if self.feature_id != None:
            rule_copy = copy.copy(rule)
            thres = utils.inverse_norm(normalizer, self.feature_id, self.feat_thres)

            subrule = f'dim {self.feature_id} feat {feat_list[self.feature_id]} <= {thres}'
            rule.append(subrule)
            self.left.get_rules(feat_list, normalizer, rule)

            subrule = f'dim {self.feature_id} feat {feat_list[self.feature_id]} > {thres}'
            rule_copy.append(subrule)            
            self.right.get_rules(feat_list, normalizer, rule_copy)
            return rule + rule_copy
        elif type(self.data) != np.ndarray:
            return
        # else:
            # print(rule[0], end='\n')
            # for i, subrule in enumerate(rule[1:]):
            #     print(' ' * (i+1), end='')
            #     print(f'-> {subrule}', end='\n')
            # print(self.ext.int_bound)
    
    def get_rules_dict(self, feat_list, normalizer, rule=[]):
        """
        get all rules in dictionary format
        """
        if self.feature_id != None:
            rule_copy = copy.copy(rule)
            thres = utils.inverse_norm(normalizer, self.feature_id, self.feat_thres)

            subrule = f'dim {self.feature_id} feat {feat_list[self.feature_id]} <= {thres}'
            rule.append(subrule)
            left_rule = self.left.get_rules_dict(feat_list, normalizer, rule)

            subrule = f'dim {self.feature_id} feat {feat_list[self.feature_id]} > {thres}'
            rule_copy.append(subrule)            
            right_rule = self.right.get_rules_dict(feat_list, normalizer, rule_copy)

            return {**left_rule, **right_rule}

        elif type(self.data) != np.ndarray:
            return
        else:
            return {tuple(rule): self.ext.ext_bound}

    def extract_thresholds_from_pred_rules(pred_rules):
        rules_info = []

        for line in pred_rules:
            match = re.match(r"dim (\d+) feat (\w+) (exceed upper bound|under lower bound): ([\d.]+)\s*(<|<=|>|>=)\s*([\d.]+)", line)

            if match:
                feature_id = int(match.group(1))
                feature_name = match.group(2)
                comparison_type = match.group(3)
                threshold = float(match.group(6))
                value = float(match.group(4))
                comparison = match.group(5)

                if comparison_type == "exceed upper bound":
                    if comparison == "<":
                        comparison = ">="
                else:
                    if comparison == ">":
                        comparison = "<="

                rule_info = {
                    "feature_id": feature_id,
                    "feature_name": feature_name,
                    "comparison": comparison,
                    "value": value,
                    "threshold": threshold,
                }

                rules_info.append(rule_info)

        # 使用集合来删除重复项
        unique_rules_info = list({(info['feature_id'], info['feature_name'], info['comparison'], info['value'], info['threshold']): info for info in rules_info}.values())

        return unique_rules_info

    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)

    def predict_rule_out(self, x, normalizer, feat_list=None):
        """
        Return decision paths of given sample as a list of rules
        """
        if feat_list is None:
            feat_list = range(x.shape[0])
        
        rules = []
        subrules = []
        curr = self
        
        while curr.feature_id is not None:
            nx, indicator = curr.next_node(x)
            if indicator == 0:
                symbol = '<='
            else:
                symbol = '>'
            
            thres = utils.inverse_norm(normalizer, curr.feature_id, curr.feat_thres)

            rule_info = {
                    "feature_id": curr.feature_id,
                    "feature_name": feat_list[curr.feature_id],
                    "comparison": symbol,
                    "value": x[curr.feature_id],
                    "threshold": thres,
            }

            subrules.append(rule_info)

        # 使用集合来删除重复项
            unique_subrules_info = list({(info['feature_id'], info['feature_name'], info['comparison'], info['value'], info['threshold']): info for info in subrules}.values())
            
            curr = nx
        
        for i, bound in enumerate(curr.ext.ext_bound):
            x_value = utils.inverse_norm(normalizer, i, x[i])
            try:
                upper_value = utils.inverse_norm(normalizer, i, bound[1])
            except:
                upper_value = bound[1]
            try:
                lower_value = utils.inverse_norm(normalizer, i, bound[0])
            except:
                lower_value = bound[0]
            
            if x[i] >= bound[1]:
                symbol = '>='
                thres = upper_value
            
            if x[i] < bound[0]:
                symbol = '<'
                thres = lower_value
            
            rule_infos = {
                    "feature_id": i,
                    "feature_name": feat_list[i],
                    "comparison": symbol,
                    "value": x_value,
                    "threshold": thres,
            }

            rules.append(rule_infos)
        

        # 删除掉-inf，然后使用集合来删除重复项
        rules = [rule for rule in rules if rule['threshold'] != float('-inf')]
        unique_rules_info = list({(info['feature_id'], info['feature_name'], info['comparison'], info['value'], info['threshold']): info for info in rules}.values())
        
        return unique_subrules_info, unique_rules_info