import numpy as np
import pandas as pd 

# constants
thresholds = {'CL_microsome_human': 4, 'CL_microsome_mouse': 4, 'CL_microsome_rat': 4, 'CL_total_dog': 0.5, 'CL_total_human': 0, 'CL_total_monkey': 0.5,
              'CL_total_rat': 1, 'CYP2C8_inhibition': 3.5, 'CYP2C9_inhibition': 3.5, 'CYP2D6_inhibition': 3.5, 'CYP3A4_inhibition': 3.5, 'Papp_Caco2': 0.8, 'Pgp_human': -0.2, 
              'hERG_binding': 3.5, 'LogD_pH_7.4': 2}
centers = thresholds
radius_convex = 2.4
radius_nonconvex = 3.4

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
    u = np.random.uniform(0, 1, size=ntest)
    for j in range(ntest):
        pval[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1) * u[j]) / (len(calib_scores) + 1)
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