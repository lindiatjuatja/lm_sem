"""
Iteratively tries to find optimal point to split on to maximize accuracy
Adds column 'thresh_pred' to scores for labeling at best threshold
"""

import numpy as np

def find_optimal_threshold(df):
    min = df['LL_delta'].min()
    max = df['LL_delta'].max()
    max_num_correct = 0
    best_thresh = 0
    for thresh in np.arange(min, max, 0.1):
        df.loc[df['LL_delta'] < thresh, 'thresh_pred'] = 'patient'
        df.loc[df['LL_delta'] >= thresh, 'thresh_pred'] = 'agent'
        num_correct = len(df[df['semantic role'] == df['thresh_pred']].index)
        if num_correct > max_num_correct:
            best_thresh = thresh
            max_num_correct = num_correct
    df.loc[df['LL_delta'] < best_thresh, 'thresh_pred'] = 'patient'
    df.loc[df['LL_delta'] >= best_thresh, 'thresh_pred'] = 'agent'
    return df, min, max, best_thresh
