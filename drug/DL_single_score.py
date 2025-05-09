import numpy as np
import pandas as pd 
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys
import copy
from sklearn.model_selection import train_test_split
from utility import thresholds, centers, radius_convex, radius_nonconvex, BH
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
            if i == len(sizes) - 2:
                # layers.append(nn.Sigmoid()) # to constraint the output values
                pass
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def criterion(Y, shape): # indicator of whether in R_1
    if shape == 0:
        t = torch.tensor(list(thresholds.values()))
        return (Y > t).all(axis=1)
    if shape == 1:
        c = torch.tensor(list(centers.values()))
        return torch.linalg.norm(Y - c, axis=1) < radius_convex
    if shape == 2:
        c = torch.tensor(list(centers.values()))
        return torch.linalg.norm(Y - c, axis=1) > radius_nonconvex

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

parser.add_argument('ntrain_f', type=int)
parser.add_argument('nvalid', type=int)
parser.add_argument('ncalib', type=int)
parser.add_argument('shape', type=int)
parser.add_argument('--family', type=int, default=3)
parser.add_argument('--retrain', type=int, default=0)

args = parser.parse_args()

beginseed = args.beginseed
endseed = args.endseed
ntrain_f = args.ntrain_f
nvalid = args.nvalid 
ncalib = args.ncalib # the rest is ntest. The three should sum to 8000
family = args.family
shape = args.shape
retrain = bool(args.retrain)

# for testing the stability
random_partition = True
# random_partition = False

# q_list = np.round(np.linspace(0.05, 0.5, 10), 2)
q_list = np.array([0.5])

all_df = pd.DataFrame()

for seed in tqdm(range(beginseed, endseed+1)):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_labeled = pd.read_csv(os.path.join('data', 'ADMET_labeled.csv'), index_col=0)
    df_labeled_pred = pd.read_csv(os.path.join('data', 'ADMET_labeled_pred.csv'), index_col=0) # y hat
    df_features = pd.read_csv(os.path.join('data', 'ADMET_features.csv'), index_col=0) # x

    Ylabeled = df_labeled.drop(columns='smiles').to_numpy()
    Ylabeled_pred = df_labeled_pred.to_numpy()
    Xlabeled = df_features.drop(columns=['SMILES', 'Label']).to_numpy()
    n, y_dim = Ylabeled.shape
    x_dim = Xlabeled.shape[1]
    ntest = n - ncalib - ntrain_f - nvalid

    ntest = 200 # fix ntest for the experiment varying ncalib

    # used a fixed Dtrain, Dvalid
    Xtrain, Ytrain, Ypred_train = Xlabeled[:ntrain_f], Ylabeled[:ntrain_f], Ylabeled_pred[:ntrain_f]
    Xvalid, Yvalid, Ypred_valid = Xlabeled[ntrain_f:ntrain_f+nvalid], Ylabeled[ntrain_f:ntrain_f+nvalid], Ylabeled_pred[ntrain_f:ntrain_f+nvalid]
    Xcalib, Ycalib, Ypred_calib = Xlabeled[ntrain_f+nvalid:], Ylabeled[ntrain_f+nvalid:], Ylabeled_pred[ntrain_f+nvalid:]

    # do a random split only for calib+test data
    if random_partition:
        # test_idx = np.random.choice(ncalib+ntest, size=ntest, replace=False)
        # calib_idx = np.setdiff1d(np.arange(ncalib+ntest), test_idx)

        tmp_idxs = np.random.choice(n-ntrain_f-nvalid, size=ntest+ncalib, replace=False)
        test_idx, calib_idx = tmp_idxs[:ntest], tmp_idxs[ntest:]
    else:
        test_idx = np.arange(ntest)
        calib_idx = np.arange(ntest,  ntest+ncalib)

    Xcalib, Xtest = Xcalib[calib_idx], Xcalib[test_idx]
    Ycalib, Ytest = Ycalib[calib_idx], Ycalib[test_idx]
    Ypred_calib, Ypred_test = Ypred_calib[calib_idx], Ypred_calib[test_idx]

    print('prediction data loaded.')

    if family == 1:
        cm_dim = x_dim
    if family == 2:
        cm_dim = y_dim
    if family in [3, 4]:
        cm_dim = x_dim + y_dim
    if family == 5:
        cm_dim = x_dim + 2 * y_dim

    if not retrain:
        best_model = []
        for qid, q in enumerate(q_list):
            mdl = torch.load(os.path.join('model', f'q={q}, family={family}, shape={shape}.pth'))
            best_model.append(mdl)
    else:
        cm = conf_model(input_size=cm_dim, hidden_sizes=[256, 256], bn=False).double()
        optimizer = torch.optim.Adam(cm.parameters(), lr=3e-4, weight_decay=1e-5)

        T = 1500
        best_model = [copy.deepcopy(cm) for q in q_list]
        best_power = [-np.inf for q in q_list]

        for epoch in tqdm(range(T)):
            with torch.no_grad():
                idxs = np.random.choice(len(Xtrain), size=int(len(Xtrain) * 0.1), replace=False)
                idx_test = idxs[int(len(Xtrain) * 0.05):]
                idx_calib = idxs[:int(len(Xtrain) * 0.05)]
                Xcalib_t, Ycalib_t, Ypredcalib_t = Xtrain[idx_calib], Ytrain[idx_calib], Ypred_train[idx_calib]
                Xtest_t, Ytest_t, Ypredtest_t = Xtrain[idx_test], Ytrain[idx_test], Ypred_train[idx_test]

                if family == 1:
                    Xcalib_t = torch.tensor(Xcalib_t).double()
                    Xtest_t = torch.tensor(Xtest_t).double()
                if family == 2:
                    Xcalib_t = torch.tensor(Ypredcalib_t).double()
                    Xtest_t = torch.tensor(Ypredtest_t).double()
                if family == 3:
                    Xcalib_t = torch.tensor(np.column_stack((Xcalib_t, Ypredcalib_t))).double()
                    Xtest_t = torch.tensor(np.column_stack((Xtest_t, Ypredtest_t))).double()
                if family == 4:
                    Xcalib_t = torch.tensor(np.column_stack((Xcalib_t, Ycalib_t))).double()
                    Xtest_t = torch.tensor(np.column_stack((Xtest_t, Ypredtest_t))).double()
                if family == 5:
                    Xcalib_t = torch.tensor(np.column_stack((Xcalib_t, Ypredcalib_t, Ycalib_t))).double()
                    Xtest_t = torch.tensor(np.column_stack((Xtest_t, Ypredtest_t, Ypredtest_t))).double()

                Ycalib_t = torch.tensor(Ycalib_t).double()
                Ytest_t = torch.tensor(Ytest_t).double()

            cm.train()
            pvals = soft_pval_sort(cm, lambda y: criterion(y, shape), Xcalib_t, Ycalib_t, Xtest_t)
            cr = criterion(Ytest_t, shape).float()
            loss = torch.sum(pvals[cr == 1]) + torch.sum(-0.5 * pvals[cr == 0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0: # validate
                cm.eval()

                with torch.no_grad():
                    power_avg = [0 for q in q_list]
                    for itr in range(100):
                        idx_calib = np.random.choice(len(Xvalid), int(len(Xvalid) * 0.3), replace=False) 
                        idx_test = np.setdiff1d(np.arange(len(Xvalid)), idx_calib)

                        Xcalib_v, Ycalib_v, Ypredcalib_v = Xvalid[idx_calib], Yvalid[idx_calib], Ypred_valid[idx_calib]
                        Xtest_v, Ytest_v, Ypredtest_v = Xvalid[idx_test], Yvalid[idx_test], Ypred_valid[idx_test]
                        if family == 1:
                            Xcalib_v = torch.tensor(Xcalib_v).double()
                            Xtest_v = torch.tensor(Xtest_v).double()
                        if family == 2:
                            Xcalib_v = torch.tensor(Ypredcalib_v).double()
                            Xtest_v = torch.tensor(Ypredtest_v).double()
                        if family == 3:
                            Xcalib_v = torch.tensor(np.column_stack((Xcalib_v, Ypredcalib_v))).double()
                            Xtest_v = torch.tensor(np.column_stack((Xtest_v, Ypredtest_v))).double()
                        if family == 4:
                            Xcalib_v = torch.tensor(np.column_stack((Xcalib_v, Ycalib_v))).double()
                            Xtest_v = torch.tensor(np.column_stack((Xtest_v, Ypredtest_v))).double() # Ytest_v should not be revealed
                        if family == 5:
                            Xcalib_v = torch.tensor(np.column_stack((Xcalib_v, Ypredcalib_v, Ycalib_v))).double()
                            Xtest_v = torch.tensor(np.column_stack((Xtest_v, Ypredtest_v, Ypredtest_v))).double() # Ytest_v should not be revealed

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

        for qid, q in enumerate(q_list):
            torch.save(best_model[qid], os.path.join('model', f'q={q}, family={family}, shape={shape}.pth'))
    
    if family == 1:
        Xcalib = torch.tensor(Xcalib).double()
        Xtest = torch.tensor(Xtest).double()
    if family == 2:
        Xcalib = torch.tensor(Ypred_calib).double()
        Xtest = torch.tensor(Ypred_test).double()
    if family == 3:
        Xcalib = torch.tensor(np.column_stack((Xcalib, Ypred_calib))).double()
        Xtest = torch.tensor(np.column_stack((Xtest, Ypred_test))).double()
    if family == 4:
        Xcalib = torch.tensor(np.column_stack((Xcalib, Ycalib))).double()
        Xtest = torch.tensor(np.column_stack((Xtest, Ypred_test))).double() # Ytest should not be revealed
    if family == 5:
        Xcalib = torch.tensor(np.column_stack((Xcalib, Ypred_calib, Ycalib))).double()
        Xtest = torch.tensor(np.column_stack((Xtest, Ypred_test, Ypred_test))).double() # Ytest should not be revealed
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
            'xdim': [x_dim],
            'ydim': [y_dim],
            'seed': [seed],
            'ntest': [ntest],
            'ntrain_f': [ntrain_f],
            'nvalid': [nvalid],
            'ncalib': [ncalib],
        })

        all_df = pd.concat((all_df, df_res))

if not os.path.exists('results'):
    os.makedirs('results')

all_df.to_csv(os.path.join('results', f"DL_s, ntest={ntest}, ntrain_f={ntrain_f}, ncalib={ncalib}, shape={shape}, seed={beginseed}_{endseed}, family={family}.csv"))