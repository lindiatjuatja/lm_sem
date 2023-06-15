import numpy as np
import pandas as pd
from statsmodels.stats import inter_rater

df = pd.read_csv('annotator_responses.csv')

corr_sum = 0
random_seeds = [1,2,3,5,7,13,42]
for seed in random_seeds:
    df_sample = df.sample(n=9, random_state=seed)
    df_remaining = df.drop(df_sample.index)
    df_sample = df_sample.T
    df_remaining = df_remaining.T
    df_sample['mean'] = df_sample.mean(axis=1)
    df_remaining['mean'] = df_remaining.mean(axis=1)
    corr_sum += np.corrcoef(df_sample['mean'].to_numpy(), df_remaining['mean'].to_numpy())[0,1]
avg_corr = corr_sum/7
print(avg_corr)