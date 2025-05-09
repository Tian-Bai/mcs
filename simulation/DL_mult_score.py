import numpy as np
import pandas as pd 
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys
import copy
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from utility import boundary_map_convex, boundary_map_nonconvex, radius_map_convex, radius_map_nonconvex, BH, gen_data_kd
import argparse
from tqdm import tqdm

class conf_model(nn.Module):
    # let's first test a simple 2-layer MLP
    # the dimension should be R^d -> R
    def __init__(self, input_size, hidden_sizes=[32, 32], bn=True):
        super(conf_model, self).__init__()

        self.hidden_sizes = hidden_sizes
        sizes = [input_size] + hidden_sizes + [1]
        layers = []

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if bn: # add batchnorm
                layers.append(nn.BatchNorm1d(sizes[i+1]))
            if i != len(sizes) - 2:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def criterion(Y, shape): # indicator of whether in R_1
    N, y_dim = Y.shape

    if shape == 0:
        return (Y > boundary_map_convex[y_dim]).all(axis=1)
    
    if shape == 1:
        return (Y < boundary_map_nonconvex[y_dim]).any(axis=1)
    
    if shape == 2:
        return torch.linalg.norm(Y - torch.ones_like(Y) * 2, axis=1) < radius_map_convex[y_dim]
    
    if shape == 3:
        # center is (2, 2, ..., 2)
        return torch.linalg.norm(Y - torch.ones_like(Y) * 2, axis=1) > radius_map_nonconvex[y_dim]

def conf_pval(calib_scores, test_scores): # torch version
    ntest = len(test_scores)
    pval = torch.zeros(ntest)
    for j in range(ntest):
        pval[j] = (torch.sum(calib_scores < test_scores[j]) + (torch.sum(calib_scores == test_scores[j]) + 1) * torch.rand(1)) / (len(calib_scores) + 1)
    return pval

def eval(criterion, Y, sel): # torch version
    true_reject = torch.sum(criterion(Y)).item()
    if len(sel) == 0:
        fdp = 0
        power = 0
    else:
        corr_sel = torch.sum(criterion(Y[sel])).item()
        fdp = 1 - corr_sel / len(sel)
        power = corr_sel / true_reject if true_reject != 0 else 0
    return fdp, power

def V(cm: conf_model, criterion, X, Y):
    """ 
    Compute the conformity score as M * 1{criterion(Y)} - cm(X) for calibration, and -cm(X) for test.
    if the argument Y is None, the function treats X as test data. Otherwise it treats X, Y as calibration data.
    """
    if Y is None:
        return -cm(X).flatten()
    else:
        with torch.no_grad():
            ind = 1000 * criterion(Y).double()
        return ind - cm(X).flatten()
    
def soft_pval_sort(cm: conf_model, criterion, Xcalib, Ycalib, Xtest):
    """
    Compute the smoothened conformal p-values.
    """
    from fast_soft_sort.pytorch_ops import soft_rank

    assert len(Xcalib) == len(Ycalib)

    N, M = len(Xcalib), len(Xtest)

    calib_score = V(cm, criterion, Xcalib, Ycalib) # (N, )
    calib_score_exp = calib_score.unsqueeze(0).repeat(M, 1) # (M, N)
    test_score = V(cm, criterion, Xtest, None) # (M, )

    combined = torch.cat([calib_score_exp, test_score.unsqueeze(1)], dim=1)

    ranks = soft_rank(combined, regularization_strength=0.1)[:, -1]

    return (ranks + 1) / (N + 1)

def soft_BH_size_sort(pval, q, tau=10):
    from fast_soft_sort.pytorch_ops import soft_sort

    M = len(pval)
    pval = soft_sort(pval.unsqueeze(0), regularization_strength=0.1).squeeze(0)
    crit_value = torch.arange(1, M+1).double() * q / M

    s = torch.sigmoid((crit_value - pval) * tau)
    sel = s * torch.arange(1, M+1).double()

    smooth_max = torch.logsumexp(sel * tau, dim=0) / tau
    return smooth_max

parser = argparse.ArgumentParser(description='')
parser.add_argument('beginseed', type=int)
parser.add_argument('endseed', type=int)

parser.add_argument('ntrain_mu', type=int)
parser.add_argument('ntrain_f', type=int)
parser.add_argument('nvalid', type=int)
parser.add_argument('ncalib', type=int)
parser.add_argument('ntest', type=int)
parser.add_argument('modelid', type=int)
parser.add_argument('setting', type=int)
parser.add_argument('shape', type=int) # (of R_1), 0, 1 - rectangular, convex/nonconvex; 2, 3 - spherical, convex/nonconvex
parser.add_argument('y_dim', type=int)
parser.add_argument('x_dim', type=int)

args = parser.parse_args()

beginseed = args.beginseed
endseed = args.endseed
ntrain_mu = args.ntrain_mu
ntrain_f = args.ntrain_f
nvalid = args.nvalid
ncalib = args.ncalib
ntest = args.ntest
modelid = args.modelid
setting = args.setting
shape = args.shape
y_dim = args.y_dim
x_dim = args.x_dim

sigma, covar = 0.5, 0.1
# q_list = np.round(np.linspace(0.05, 0.5, 10), 2)
q_list = np.array([0.3])

if y_dim not in [2, 5, 10, 30]:
    raise NotImplementedError

all_df = pd.DataFrame()

for seed in tqdm(range(beginseed, endseed+1)):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if modelid == 0:
    #     init_model = OracleRegressorkd(setting=setting)
    if modelid == 1:
        init_model = LinearRegression()
    elif modelid == 2:
        init_model = RandomForestRegressor()
    elif modelid == 3: 
        init_model = MultiOutputRegressor(SVR(kernel='rbf', gamma=0.1))

    cm = conf_model(input_size=x_dim + y_dim, hidden_sizes=[32, 32], bn=False).double()
    optimizer = torch.optim.Adam(cm.parameters(), lr=1e-3, weight_decay=1e-5)

    Xtrain_mu, Ytrain_mu = gen_data_kd(setting, y_dim, ntrain_mu, sigma, covar, x_dim)
    Xtrain, Ytrain = gen_data_kd(setting, y_dim, ntrain_f, sigma, covar, x_dim)
    Xvalid, Yvalid = gen_data_kd(setting, y_dim, nvalid, sigma, covar, x_dim)
    Xcalib, Ycalib = gen_data_kd(setting, y_dim, ncalib, sigma, covar, x_dim)
    Xtest, Ytest = gen_data_kd(setting, y_dim, ntest, sigma, covar, x_dim)

    # pretrain a mu
    mu = init_model
    mu.fit(Xtrain_mu, Ytrain_mu)

    T = 1000
    best_model = [copy.deepcopy(cm) for q in q_list]
    best_power = [-np.inf for q in q_list]

    for epoch in tqdm(range(T)):
        with torch.no_grad():
            idxs = np.random.choice(len(Xtrain), size=int(len(Xtrain) * 0.6), replace=False)
            idx_test = idxs[int(len(Xtrain) * 0.4):]
            idx_calib = idxs[:int(len(Xtrain) * 0.4)]
            Xcalib_t, Ycalib_t = Xtrain[idx_calib], Ytrain[idx_calib]
            Xtest_t, Ytest_t = Xtrain[idx_test], Ytrain[idx_test]

            Xcalib_t = torch.tensor(np.column_stack((Xcalib_t, mu.predict(Xcalib_t)))).double() # use Ycalib_t or hat{Y}calib_t?
            Xtest_t = torch.tensor(np.column_stack((Xtest_t, mu.predict(Xtest_t)))).double()
            Ycalib_t = torch.tensor(Ycalib_t).double()
            Ytest_t = torch.tensor(Ytest_t).double()

        cm.train()
        pvals = soft_pval_sort(cm, lambda y: criterion(y, shape), Xcalib_t, Ycalib_t, Xtest_t)
        bh_size = soft_BH_size_sort(pvals, q=0.3)
        loss = -bh_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0: # validate
            cm.eval()

            with torch.no_grad():
                power_avg = [0 for q in q_list]
                for itr in range(100):
                    Xcalib_v, Xtest_v, Ycalib_v, Ytest_v = train_test_split(Xvalid, Yvalid, train_size=0.3)
                    Xcalib_v = torch.tensor(np.column_stack((Xcalib_v, mu.predict(Xcalib_v)))).double() # use Ycalib_v or hat{Y}calib_v?
                    Xtest_v = torch.tensor(np.column_stack((Xtest_v, mu.predict(Xtest_v)))).double()
                    Ycalib_v = torch.tensor(Ycalib_v).double()
                    Ytest_v = torch.tensor(Ytest_v).double()

                    calib_scores = V(cm, lambda y: criterion(y, shape), Xcalib_v, Ycalib_v)
                    test_scores = V(cm, lambda y: criterion(y, shape), Xtest_v, None)

                    pvals = conf_pval(calib_scores, test_scores)

                    for qid, q in enumerate(q_list):
                        sel = BH(pvals, q)
                        FDP, power = eval(lambda y: criterion(y, shape), Ytest_v, sel)
                        power_avg[qid] += power / 100

            for qid, q in enumerate(q_list):
                if power_avg[qid] > best_power[qid]:
                    best_power[qid] = power_avg[qid]
                    best_model[qid] = copy.deepcopy(cm)
    
    Xcalib = torch.tensor(np.column_stack((Xcalib, mu.predict(Xcalib)))).double() # use Ycalib or hat{Y}calib?
    Xtest = torch.tensor(np.column_stack((Xtest, mu.predict(Xtest)))).double()
    Ycalib = torch.tensor(Ycalib).double()
    Ytest = torch.tensor(Ytest).double()
    
    # conduct the selections and evaluate performances
    for qid, q in enumerate(q_list):
        cm_best = best_model[qid]

        calib_scores = V(cm_best, lambda y: criterion(y, shape), Xcalib, Ycalib)
        test_scores = V(cm_best, lambda y: criterion(y, shape), Xtest, None)

        pvals = conf_pval(calib_scores, test_scores)
        sel = BH(pvals, q)
        DL_FDP, DL_power = eval(lambda y: criterion(y, shape), Ytest, sel)

        df_res = pd.DataFrame({
            'DL_FDP': [DL_FDP], 
            'DL_power': [DL_power],
            'q': [q],
            'setting': [setting],
            'shape': [shape],
            'model': [['oracle', 'linear', 'rf', 'svm'][modelid]], 
            'xdim': [x_dim],
            'ydim': [y_dim],
            'seed': [seed],
            'ntest': [ntest],
            'ntrain_mu': [ntrain_mu],
            'ntrain_f': [ntrain_f],
            'nvalid': [nvalid],
            'ncalib': [ncalib],
        })

        all_df = pd.concat((all_df, df_res))

if not os.path.exists('results'):
    os.makedirs('results')

all_df.to_csv(os.path.join('results', f"DL_m, setting={setting}, shape={shape}, xdim={x_dim}, ydim={y_dim}, modelid={modelid}, ntest={ntest}, ntrain_f={ntrain_f}, ncalib={ncalib}, seed={beginseed}_{endseed}.csv"))