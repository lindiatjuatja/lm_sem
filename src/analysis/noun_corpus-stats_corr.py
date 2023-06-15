"""
Calculates correlation between corpus stats and human ratings
Note that the Propbank correlation only includes nouns with over 4 occurences with an arg0 or arg1 label
"""

import numpy as np
import pandas as pd

def human_subj_corr():
    df_ngrams = pd.read_csv('../../data/syntactic_ngrams/complete_counts.csv')
    df_ngrams = df_ngrams.sort_values('noun')
    df_ngrams.reset_index(drop=True, inplace=True)
    df_human = pd.read_csv('noun_annotation/human-noun_and_stats.csv')
    df_human['noun'] = df_human['noun'].apply(str.lower)
    df_human = df_human.sort_values('noun')
    df_human.reset_index(drop=True, inplace=True)
    df = pd.DataFrame()
    df['ngrams'] = df_ngrams['subj_ratio']
    df['human'] = df_human['avg']
    df.dropna(inplace=True)
    assert df_ngrams['noun'].equals(df_human['noun'])
    corr = np.corrcoef(df['ngrams'].to_numpy(), df['human'].to_numpy())[0,1]
    print(f'Correlation between humans and Ngrams: {corr}')
    
def human_propbank_corr():
    df_propbank = pd.read_csv('../../data/probank/propbank_counts.csv')
    df_propbank = df_propbank.sort_values('noun')
    df_propbank.reset_index(drop=True, inplace=True)
    df_propbank['arg_counts'] = df_propbank['arg0_count'] + df_propbank['arg1_count']
    df_propbank_filtered = df_propbank[df_propbank['arg_counts'] >= 4]
    df_propbank_filtered.reset_index(drop=True, inplace=True)
    df_propbank_filtered.to_csv('../../data/probank/propbank-filtered.csv', index=False)
    df_human = pd.read_csv('noun_annotation/human-noun_and_stats.csv')
    df_human = df_human.sort_values('noun')
    df_human.reset_index(drop=True, inplace=True)
    assert df_propbank['noun'].equals(df_human['noun'])
    df = pd.DataFrame()
    df['noun'] = df_human['noun']
    df['human'] = df_human['avg']
    df['propbank'] = df_propbank['arg0_prob']
    df['propbank_count'] = df_propbank['arg_counts']
    df_filtered = df[df['propbank_count'] >= 4]
    df_filtered.reset_index(drop=True, inplace=True)
    corr = np.corrcoef(df_filtered['propbank'].to_numpy(), df_filtered['human'].to_numpy())[0,1]
    print(f'Correlation between humans and Propbank: {corr}')

def main():
    human_subj_corr()
    human_propbank_corr()
            
if __name__ == "__main__":
    main()  