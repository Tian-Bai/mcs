import numpy as np
import pandas as pd 
import random
import os
from sklearn.model_selection import train_test_split
from utility import thresholds, centers, radius_convex, radius_nonconvex, eval, conf_pval, BH
import argparse
from tqdm import tqdm

def criterion(Y, shape): # indicator of whether in R_1
    if shape == 0:
        t = np.array(list(thresholds.values()))
        return (Y > t).all(axis=1)
    if shape == 1:
        c = np.array(list(centers.values()))
        return np.linalg.norm(Y - c, axis=1) < radius_convex
    if shape == 2:
        c = np.array(list(centers.values()))
        return np.linalg.norm(Y - c, axis=1) > radius_nonconvex

def criterion_decomp(Y, shape, d): # indicator of whether Y_d is satisfying the d-th criterion (if its decomposable)
    assert shape == 0
    t = np.array(list(thresholds.values()))
    return Y > t[d]

def dist(Y, shape): # distance to R_0
    if shape == 0:
        N, y_dim = Y.shape

        boundary = np.array(list(thresholds.values()))
        has_negative = np.any(Y < boundary, axis=1)
        min_coordinates = np.min(Y - boundary, axis=1)
        distances = np.where(has_negative, 0, min_coordinates)
        return distances
    
    if shape == 1:
        dist_to_center = np.linalg.norm(Y - np.array(list(centers.values())), axis=1, ord=2)
        return np.maximum(radius_convex - dist_to_center, 0)
    
    if shape == 2:
        dist_to_center = np.linalg.norm(Y - np.array(list(centers.values())), axis=1, ord=2)
        return np.maximum(dist_to_center - radius_nonconvex, 0)

parser = argparse.ArgumentParser(description='')
parser.add_argument('beginseed', type=int)
parser.add_argument('endseed', type=int)
parser.add_argument('ncalib', type=int)
parser.add_argument('shape', type=int)
args = parser.parse_args()

beginseed = args.beginseed
endseed = args.endseed
ncalib = args.ncalib # ntest is automatically the rest
shape = args.shape

# q_list = np.round(np.linspace(0.05, 0.5, 10), 2)
q_list = np.array([0.3, 0.35, 0.4, 0.45, 0.5])
q_prime_list = np.zeros_like(q_list) # for CS_intersect_search

# for testing the stability
# random_partition = True
random_partition = False

all_df = pd.DataFrame()

for seed in tqdm(range(beginseed, endseed+1)):
    random.seed(seed)
    np.random.seed(seed)

    df_labeled = pd.read_csv(os.path.join('data', 'ADMET_labeled.csv'), index_col=0)
    df_labeled_pred = pd.read_csv(os.path.join('data', 'ADMET_labeled_pred.csv'), index_col=0) # y hat
    df_labeled_bin = pd.read_csv(os.path.join('data', 'ADMET_labeled_bin.csv'), index_col=0) # prob(y in R)

    Ylabeled = df_labeled.drop(columns='smiles').to_numpy()
    Ylabeled_pred = df_labeled_pred.to_numpy()
    Ylabeled_bin = df_labeled_bin.to_numpy()
    n, y_dim = Ylabeled.shape
    ntest = n - ncalib

    if random_partition:
        test_idx = np.random.choice(n, size=ntest, replace=False)
        calib_idx = np.setdiff1d(np.arange(n), test_idx)
    else:
        test_idx = np.arange(ntest)
        calib_idx = np.arange(ntest, ntest+ncalib)

    Ycalib, Ytest = Ylabeled[calib_idx], Ylabeled[test_idx]
    Ypred_calib, Ypred_test = Ylabeled_pred[calib_idx], Ylabeled_pred[test_idx]
    Ybin_calib, Ybin_test = Ylabeled_bin[calib_idx], Ylabeled_bin[test_idx]

    ''' The purposed method (clipped score) '''

    calib_scores = 1000 * criterion(Ycalib, shape) - dist(Ypred_calib, shape)
    test_scores = -dist(Ypred_test, shape)

    mcs_pval = conf_pval(calib_scores, test_scores)

    ''' The purpose method (first score)'''

    calib_scores = dist(Ycalib, shape) - dist(Ypred_calib, shape)
    test_scores = -dist(Ypred_test, shape)

    mcs_alt_pval = conf_pval(calib_scores, test_scores)

    ''' Fitting a binary classifier '''

    calib_scores = 1000 * criterion(Ycalib, shape) - Ybin_calib
    test_scores = -Ybin_test

    binary_pval = conf_pval(calib_scores, test_scores)

    ''' Conduct k selections and intersect - impossible with shape 1 '''

    if shape == 0:
        intersect_pval = np.zeros((y_dim, ntest))
        for d in range(y_dim):
            calib_scores = 1000 * criterion_decomp(Ycalib[:, d], shape, d) - Ypred_calib[:, d]
            test_scores = -Ypred_test[:, d]

            intersect_pval[d] = conf_pval(calib_scores, test_scores)

    ''' CS_intersect_search ''' 

    if shape == 0:
        Yvalid, Ycalib, Ypred_valid, Ypred_calib = train_test_split(Ycalib, Ypred_calib, test_size=0.5, random_state=seed)
        # use Xvalid, Yvalid to search for q_prime
        intersect_pval_valid = np.zeros((y_dim, len(Yvalid))) # valid: p-values for validation set
        intersect_pval_vtest = np.zeros((y_dim, ntest)) # vtest: final p-values for test set

        # get pvalues
        for d in range(y_dim):
            calib_scores = 1000 * criterion_decomp(Ycalib[:, d], shape, d) - Ypred_calib[:, d]
            valid_scores = -Ypred_valid[:, d]
            test_scores = -Ypred_test[:, d]

            intersect_pval_valid[d] = conf_pval(calib_scores, valid_scores)
            intersect_pval_vtest[d] = conf_pval(calib_scores, test_scores)

        # using validation set, search for q_prime
        for q_test in np.linspace(0, q_list.max(), num=500):
            inter_sel_valid = set(range(len(Yvalid)))
            for d in range(y_dim):
                d_sel = BH(intersect_pval_valid[d], q_test)
                inter_sel_valid &= set(d_sel)

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

        if shape == 0:
            inter_sel = set(range(ntest))
            inter_corr_sel = set(range(ntest)) # Bonferroni corrected
            inter_valid_sel = set(range(ntest)) # CS_intersect_search

            for d in range(y_dim):
                d_sel = BH(intersect_pval[d], q)
                inter_sel &= set(d_sel) # intersect

                d_corr_sel = BH(intersect_pval[d], q / y_dim)
                inter_corr_sel &= set(d_corr_sel)

                d_valid_sel = BH(intersect_pval_vtest[d], q_prime_list[qid])
                inter_valid_sel &= set(d_valid_sel)
            
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
            'ydim': [y_dim],
            'seed': [seed],
            'ntest': [ntest],
            'ncalib': [ncalib],
            'shape': [shape]
        })

        if shape == 0:
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

all_df.to_csv(os.path.join('results', f"Drug_MCS, ntest={ntest}, ncalib={ncalib}, shape={shape}, seed={beginseed}_{endseed}.csv"))

