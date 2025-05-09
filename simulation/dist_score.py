import numpy as np
import pandas as pd 
import random
import os
import copy
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from utility import boundary_map_convex, boundary_map_nonconvex, radius_map_convex, radius_map_nonconvex, eval, conf_pval, BH, gen_data_kd
import argparse
from tqdm import tqdm

def criterion(Y, shape): # indicator of whether in R_1
    N, y_dim = Y.shape

    if shape == 0:
        return (Y > boundary_map_convex[y_dim]).all(axis=1)
    
    if shape == 1:
        return (Y < boundary_map_nonconvex[y_dim]).any(axis=1)
    
    if shape == 2:
        return np.linalg.norm(Y - np.ones_like(Y) * 2, axis=1) < radius_map_convex[y_dim]
    
    if shape == 3:
        # center is (2, 2, ..., 2)
        return np.linalg.norm(Y - np.ones_like(Y) * 2, axis=1) > radius_map_nonconvex[y_dim]

def criterion_decomp(Y, shape, y_dim): # indicator of whether Y_d is satisfying the d-th criterion (if its decomposable)
    assert shape in [0, 1]

    if shape == 0:
        return Y > boundary_map_convex[y_dim]
    
    if shape == 1:
        return Y < boundary_map_nonconvex[y_dim]

def dist(Y, shape): # distance to R_0
    N, y_dim = Y.shape

    if shape == 0:  
        boundary = boundary_map_convex[y_dim]
        has_negative = np.any(Y < boundary, axis=1)
        min_coordinates = np.min(Y - np.ones_like(Y) * boundary, axis=1)
        distances = np.where(has_negative, 0, min_coordinates)
        return distances
    
    if shape == 1: 
        boundary = boundary_map_nonconvex[y_dim]
        Y_pos = np.maximum(boundary - Y, 0)
        # default use norm = 2
        return np.linalg.norm(Y_pos, axis=1, ord=2)

    if shape == 2:
        radius = radius_map_convex[y_dim] # center is 0

        dist_to_center = np.linalg.norm(Y - np.ones_like(Y) * 2, axis=1, ord=2)
        return np.maximum(radius - dist_to_center, 0)
    
    if shape == 3:
        radius = radius_map_nonconvex[y_dim]

        dist_to_center = np.linalg.norm(Y - np.ones_like(Y) * 2, axis=1, ord=2)
        return np.maximum(dist_to_center - radius, 0)

parser = argparse.ArgumentParser(description='')
parser.add_argument('beginseed', type=int)
parser.add_argument('endseed', type=int)

parser.add_argument('ntrain', type=int)
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
ntrain = args.ntrain
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
q_prime_list = np.zeros_like(q_list) # for CS_intersect_search

if y_dim not in [2, 5, 10, 30]:
    raise NotImplementedError

all_df = pd.DataFrame()

for seed in tqdm(range(beginseed, endseed+1)):
    random.seed(seed)
    np.random.seed(seed)

    # if modelid == 0:
    #     init_model = OracleRegressorkd(setting=setting)
    #     bin_init_model = init_model
    if modelid == 1:
        init_model = LinearRegression()
        bin_init_model = init_model
    elif modelid == 2:
        init_model = RandomForestRegressor()
        bin_init_model = init_model
    elif modelid == 3:
        init_model = MultiOutputRegressor(SVR(kernel='rbf', gamma=0.1))
        bin_init_model = SVR(kernel='rbf', gamma=0.1)

    Xtrain, Ytrain = gen_data_kd(setting, y_dim, ntrain, sigma, covar, x_dim)
    Xcalib, Ycalib = gen_data_kd(setting, y_dim, ncalib, sigma, covar, x_dim)
    Xtest, Ytest = gen_data_kd(setting, y_dim, ntest, sigma, covar, x_dim)

    ''' The purposed method (clipped score) '''

    reg = copy.deepcopy(init_model)
    reg.fit(Xtrain, Ytrain)
    Ypred_calib = reg.predict(Xcalib)
    Ypred_test = reg.predict(Xtest)

    calib_scores = 1000 * criterion(Ycalib, shape) - dist(Ypred_calib, shape)
    test_scores = -dist(Ypred_test, shape)

    mcs_pval = conf_pval(calib_scores, test_scores)

    ''' The purpose method (first score)'''

    calib_scores = dist(Ycalib, shape) - dist(Ypred_calib, shape)
    test_scores = -dist(Ypred_test, shape)

    mcs_alt_pval = conf_pval(calib_scores, test_scores)

    ''' Fitting a binary classifier '''
    
    reg = copy.deepcopy(bin_init_model)
    reg.fit(Xtrain, criterion(Ytrain, shape))
    Ypred_calib = reg.predict(Xcalib)
    Ypred_test = reg.predict(Xtest)

    calib_scores = 1000 * criterion(Ycalib, shape) - Ypred_calib
    test_scores = -Ypred_test

    binary_pval = conf_pval(calib_scores, test_scores)

    ''' Conduct k selections and intersect - impossible with shape 2 and 3 '''

    if shape in [0, 1]:
        reg = copy.deepcopy(init_model)

        reg.fit(Xtrain, criterion_decomp(Ytrain, shape, y_dim))

        Ypred_calib = reg.predict(Xcalib)
        Ypred_test = reg.predict(Xtest)
        intersect_pval = np.zeros((y_dim, ntest))

        for d in range(y_dim):
            calib_scores = 1000 * criterion_decomp(Ycalib[:, d], shape, y_dim) - Ypred_calib[:, d]
            test_scores = -Ypred_test[:, d]

            intersect_pval[d] = conf_pval(calib_scores, test_scores)

    ''' CS_intersect_search ''' 

    if shape in [0, 1]:
        Xvalid, Xcalib, Yvalid, Ycalib = train_test_split(Xcalib, Ycalib, test_size=0.5, random_state=seed)
        # use Xvalid, Yvalid to search for q_prime
        Ypred_calib = reg.predict(Xcalib)
        Ypred_valid = reg.predict(Xvalid)
        Ypred_test = reg.predict(Xtest)
        intersect_pval_valid = np.zeros((y_dim, len(Yvalid))) # valid: p-values for validation set
        intersect_pval_vtest = np.zeros((y_dim, ntest)) # vtest: final p-values for test set

        # get pvalues
        for d in range(y_dim):
            calib_scores = 1000 * criterion_decomp(Ycalib[:, d], shape, y_dim) - Ypred_calib[:, d]
            valid_scores = -Ypred_valid[:, d]
            test_scores = -Ypred_test[:, d]

            intersect_pval_valid[d] = conf_pval(calib_scores, valid_scores)
            intersect_pval_vtest[d] = conf_pval(calib_scores, test_scores)

        # using validation set, search for q_prime
        for q_test in np.linspace(0, q_list.max(), num=500):
            inter_sel_valid = set(range(len(Yvalid)))
            for d in range(y_dim):
                d_sel = BH(intersect_pval_valid[d], q_test)
                if shape == 0:
                    inter_sel_valid &= set(d_sel)
                elif shape == 1:
                    inter_sel_valid |= set(d_sel)

            inter_sel_valid = np.array(list(inter_sel_valid))
            inter_fdp_valid, inter_power_valid = eval(lambda y: criterion(y, shape), Yvalid, inter_sel_valid)

            q_prime_list[q_list >= inter_fdp_valid] = q_test # update q_prime

    # conduct the selections and evaluate performances
    for qid, q in enumerate(q_list):
        mcs_sel = BH(mcs_pval, q)
        mcs_alt_sel = BH(mcs_alt_pval, q)
        binary_sel = BH(binary_pval, q)

        mcs_fdp, mcs_power = eval(lambda y: criterion(y, shape), Ytest, mcs_sel)
        mcs_alt_fdp, mcs_alt_power = eval(lambda y: criterion(y, shape), Ytest, mcs_alt_sel)
        binary_fdp, binary_power = eval(lambda y: criterion(y, shape), Ytest, binary_sel)

        if shape in [0, 1]:
            inter_sel = set(range(ntest))
            inter_corr_sel = set(range(ntest)) # Bonferroni corrected
            inter_valid_sel = set(range(ntest)) # CS_intersect_search

            for d in range(y_dim):
                d_sel = BH(intersect_pval[d], q)
                if shape == 0:
                    inter_sel &= set(d_sel) # intersect
                elif shape == 1:
                    inter_sel |= set(d_sel) # union

                d_corr_sel = BH(intersect_pval[d], q / y_dim)
                if shape == 0:
                    inter_corr_sel &= set(d_corr_sel)
                elif shape == 1:
                    inter_corr_sel |= set(d_corr_sel)

                d_valid_sel = BH(intersect_pval_vtest[d], q_prime_list[qid])
                if shape == 0:
                    inter_valid_sel &= set(d_valid_sel)
                elif shape == 1:
                    inter_valid_sel |= set(d_valid_sel)
            
            inter_sel = np.array(list(inter_sel))
            inter_corr_sel = np.array(list(inter_corr_sel))
            inter_valid_sel = np.array(list(inter_valid_sel))

            inter_fdp, inter_power = eval(lambda y: criterion(y, shape), Ytest, inter_sel)
            inter_corr_fdp, inter_corr_power = eval(lambda y: criterion(y, shape), Ytest, inter_corr_sel)
            inter_valid_fdp, inter_valid_power = eval(lambda y: criterion(y, shape), Ytest, inter_valid_sel)

        df_res = pd.DataFrame({
            'MCS_FDP': [mcs_fdp], 
            'MCS_power': [mcs_power],
            'MCS_alt_FDP': [mcs_alt_fdp],
            'MCS_alt_power': [mcs_alt_power],
            'Binary_FDP': [binary_fdp],
            'Binary_power': [binary_power],
            'q': [q],
            'setting': [setting],
            'shape': [shape],
            'model': [['oracle', 'linear', 'rf', 'svm'][modelid]], 
            'xdim': [x_dim],
            'ydim': [y_dim],
            'seed': [seed],
            'ntest': [ntest],
            'ntrain': [ntrain],
            'ncalib': [ncalib],
        })

        if shape in [0, 1]:
            df_res = pd.concat((df_res, pd.DataFrame({
                'Intersect_FDP': [inter_fdp],
                'Intersect_power': [inter_power],
                'Intersect_corr_FDP': [inter_corr_fdp],
                'Intersect_corr_power': [inter_corr_power],
                'Intersect_valid_FDP': [inter_valid_fdp],
                'Intersect_valid_power': [inter_valid_power],
            })), axis=1)

        all_df = pd.concat((all_df, df_res))

if not os.path.exists('results'):
    os.makedirs('results')

all_df.to_csv(os.path.join('results', f"MCS, setting={setting}, shape={shape}, xdim={x_dim}, ydim={y_dim}, modelid={modelid}, ntest={ntest}, ntrain={ntrain}, ncalib={ncalib}, seed={beginseed}_{endseed}.csv"))

