"""
Calculates correlation between LL_delta (from models) and human ratings as well as ngram counts
"""

import argparse
import numpy as np
import pandas as pd
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--score_folder", type=str, default="../scores/base/")

model_order = {'bloom-560m':0, 'bloom-1b1':1, 'bloom-1b7':2, 'bloom-3b':3, 'bloom-7b1':4,
               'gpt2-small':5, 'gpt2-medium':6, 'gpt2-large':7, 'gpt2-xl':8,
               'gpt3-ada':9, 'gpt3-babbage':10, 'gpt3-curie':11, 'gpt3-davinci':12,
               'text-ada-001':13, 'text-babbage-001':14, 'text-curie-001':15, 'text-davinci-001':16,
               'text-davinci-003':17}

def format_scores(score_folder):
    df_ngrams = pd.read_csv('syntactic_ngrams/complete_counts.csv')
    df_ngrams = df_ngrams.sort_values('noun')
    df_ngrams.reset_index(drop=True, inplace=True)
    df_human = pd.read_csv('noun_annotation/human-noun_and_stats.csv')
    df_human['noun'] = df_human['noun'].apply(str.lower)
    df_human = df_human.sort_values('noun')
    df_human.reset_index(drop=True, inplace=True)
    assert df_ngrams['noun'].equals(df_human['noun'])
    
    exp_folders = glob(score_folder + '*')
    for exp in exp_folders:
        noun_folder = exp + '/noun_scores/*'
        noun_score_files = glob(noun_folder)
        for filepath in noun_score_files:
            df = pd.read_csv(filepath)
            df['noun'] = df['noun'].apply(str.lower)
            df = df.sort_values('noun')
            df.reset_index(drop=True, inplace=True)
            assert df_ngrams['noun'].equals(df['noun'])
            df['LL_delta'] = df['LL_agent']-df['LL_patient']
            df['subj_ratio'] = df_ngrams['subj_ratio']
            df['human_rating'] = (df_human['avg']-1)/4
            df['human_rating_var'] = df_human['var']
            df.to_csv(filepath, index=False)

def calculate_noun_avg_LL_delta(df, dir):
    df_noun_acc = pd.DataFrame(columns=['noun', 'avg_LL_delta'])
    nouns = df['noun'].unique()
    for noun in nouns:
        df_noun = df[df['noun']==noun]
        avg_LL_delta = df_noun['LL_delta'].mean()
        row = [noun, avg_LL_delta]
        df_noun_acc.loc[len(df_noun_acc.index)] = row
    df_noun_acc = df_noun_acc.sort_values(['noun'])
    df_noun_acc.to_csv(dir + '/noun_LL_delta.csv', index=False)
    return

def main():
    args = parser.parse_args()
    score_folder = args.score_folder
    format_scores(score_folder)
    exp_folders = glob(score_folder + '*')
    for exp in exp_folders:
        print(exp)
        exp_dir = os.path.join(exp.split('/')[-2], exp.split('/')[-1])
        exp_results_dir = os.path.join('../results/', exp_dir)
        print(exp_results_dir)
        if not os.path.exists(exp_results_dir):    
            os.makedirs(exp_results_dir)
        df_corr = pd.DataFrame(columns=['model', 'corr_human', 'corr_subj'])
        noun_folder = exp + '/noun_scores/*'
        noun_score_files = glob(noun_folder)
        for filepath in noun_score_files:
            filename = os.path.basename(filepath)
            df = pd.read_csv(filepath)
            model_name = os.path.splitext(filename)[0]
            model_results_dir = os.path.join(exp_results_dir, model_name)
            if not os.path.exists(model_results_dir):    
                os.makedirs(model_results_dir)
            calculate_noun_avg_LL_delta(df, model_results_dir)
            corr_human = np.corrcoef(df['LL_delta'].to_numpy(), df['human_rating'].to_numpy())[0,1]
            df.dropna(inplace=True) # drop nouns with no counts in syntactic ngrams (roadtripper)
            corr_subj = np.corrcoef(df['LL_delta'].to_numpy(), df['subj_ratio'].to_numpy())[0,1]
            row = [model_name, corr_human, corr_subj]
            df_corr.loc[len(df_corr.index)] = row
        df_corr.set_index('model', inplace=True)
        df_corr.rename(index = {'gpt2-x': 'gpt2-small'}, inplace=True)
        df_corr.sort_values(by='model', key=lambda x: x.map(model_order), inplace=True)
        df_corr.to_csv(exp_results_dir + '/noun_corr.csv')
            
if __name__ == "__main__":
    main()  