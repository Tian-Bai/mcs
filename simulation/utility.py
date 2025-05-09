import numpy as np
import pandas as pd 
import math
from scipy.stats import multivariate_t

# constants
boundary_map_convex = {2: 1, 5: 0.2, 10: -0.2, 30: -0.6}
boundary_map_nonconvex = {2: -0.5, 5: -0.8, 10: -1.1, 30: -1.6} 
radius_map_convex = {2: 1.5, 5: 2.6, 10: 4.1, 30: 7.5}
radius_map_nonconvex = {2: 3, 5: 4, 10: 5.5, 30: 9.5}

def gen_data_kd(setting, ydim, n, sig, covar, dim=10):
    '''
    Generate k-dimensional data with `dim` covariates.
    '''
    X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
    Y = np.zeros((n, ydim))
    cov = np.ones((ydim, ydim)) * sig * covar
    np.fill_diagonal(cov, sig)
    rng = np.random.default_rng(33)

    if setting == 1:
        # linear
        for i in range(ydim):
            Y[:,i] = X[:,(i) % dim] * 2 + X[:,(i+1) % dim] * (-0.5) + X[:,(i+2) % dim] + 1.5
        if sig != 0:
            Y += rng.multivariate_normal(mean=np.zeros(ydim), cov=cov, size=n)
        return X, Y
    
    if setting == 2:
        # weak nonlinearity
        for i in range(ydim):
            Y[:,i] = X[:,(i) % dim] + X[:,(i+2) % dim] ** 2 + 0.5
        if sig != 0:
            Y += rng.multivariate_normal(mean=np.zeros(ydim), cov=cov, size=n)
        return X, Y
    
    if setting == 3:
        # strong nonlinearity
        for i in range(ydim):
            Y[:,i] = (X[:,(i) % dim] * X[:,(i+1) % dim] >  0) * (X[:,(i+2) % dim] >  0.5) * (0.25 + X[:,(i+2) % dim]) \
                   + (X[:,(i) % dim] * X[:,(i+1) % dim] <= 0) * (X[:,(i+2) % dim] < -0.5) * (X[:,(i+2) % dim] - 0.25) + 0.75
        if sig != 0:
            Y += rng.multivariate_normal(mean=np.zeros(ydim), cov=cov, size=n)
        return X, Y
    
    if setting == 4:
        # linear
        for i in range(ydim):
            Y[:,i] = X[:,(i) % dim] * 2 + X[:,(i+1) % dim] * (-0.5) + X[:,(i+2) % dim] + 1.5
        if sig != 0:
            rv = multivariate_t([0] * ydim, cov, df=3)
            Y += rv.rvs(size=n)
        return X, Y
    
    if setting == 5:
        # weak nonlinearity
        for i in range(ydim):
            Y[:,i] = X[:,(i) % dim] + X[:,(i+2) % dim] ** 2 + 0.5
        if sig != 0:
            rv = multivariate_t([0] * ydim, cov, df=3)
            Y += rv.rvs(size=n)
        return X, Y
    
    if setting == 6:
        # strong nonlinearity
        for i in range(ydim):
            Y[:,i] = (X[:,(i) % dim] * X[:,(i+1) % dim] >  0) * (X[:,(i+2) % dim] >  0.5) * (0.25 + X[:,(i+2) % dim]) \
                   + (X[:,(i) % dim] * X[:,(i+1) % dim] <= 0) * (X[:,(i+2) % dim] < -0.5) * (X[:,(i+2) % dim] - 0.25) + 0.75
        if sig != 0:
            rv = multivariate_t([0] * ydim, cov, df=3)
            Y += rv.rvs(size=n)
        return X, Y

def eval(criterion, Y, sel):
    '''
    Evaluate the selection performace: power and FDP. The target region is the first quadrant.
    '''
    true_reject = np.sum(criterion(Y))
    if len(sel) == 0:
        fdp = 0
        power = 0
    else:
        corr_sel = np.sum(criterion(Y[sel]))
        fdp = 1 - corr_sel / len(sel)
        power = corr_sel / true_reject if true_reject != 0 else 0
    return fdp, power

def conf_pval(calib_scores, test_scores):    
    ntest = len(test_scores)
    pval = np.zeros(ntest)
    for j in range(ntest):
        pval[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1) * np.random.uniform(0, 1)) / (len(calib_scores) + 1)
    return pval

def BH(pvals, q):
    ''' 
    Given a list of p-values and nominal FDR level q, apply BH procedure to get a rejection set.
    '''

    ntest = len(pvals)
         
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return np.array([])
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller) + 1)])
        return idx_sel
    
def eBH(evals, q):
    '''
    Given a list of e-values and nominal FDR level q, apply base eBH procedure (no pruning) to get a rejection set.
    '''
    return BH(1 / evals, q)