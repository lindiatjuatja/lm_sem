"""
Calculates:
- accuracy of labels (predicted label determined by comparing log probs of "agent" or "patient")
- consistency between predicted label for corresponding noun out of context and predicted label in intransitive
- correlation between LL_delta in the intransitive and the correspinding noun
"""

import argparse
import numpy as np
import pandas as pd
from glob import glob
import os
from sklearn import metrics
from label_threshold import find_optimal_threshold

parser = argparse.ArgumentParser()
parser.add_argument("--score_folder", type=str, default="../scores/base/")

model_order = {'bloom-560m':0, 'bloom-1b1':1, 'bloom-1b7':2, 'bloom-3b':3, 'bloom-7b1':4,
               'gpt2-small':5, 'gpt2-medium':6, 'gpt2-large':7, 'gpt2-xl':8,
               'gpt3-ada':9, 'gpt3-babbage':10, 'gpt3-curie':11, 'gpt3-davinci':12,
               'text-ada-001':13, 'text-babbage-001':14, 'text-curie-001':15, 'text-davinci-001':16,
               'text-davinci-003':17}

def format_scores(filepath):
    df = pd.read_csv(filepath)
    df['noun'] = df['noun'].apply(str.lower)
    df['LL_delta'] = df['LL_agent']-df['LL_patient']
    df, min, max, thresh = find_optimal_threshold(df)
    df = df.sort_values(['id'])
    df.to_csv(filepath, index=False)
    return [min, max, thresh]

def calculate_noun_accuracy_and_overlap(df, noun_pred_info, dir):
    # overlap is percentage of labels that match between intransitive pred and noun pred
    df_noun_acc = pd.DataFrame(columns=['noun', 'num_counts', 'accuracy', 'overlap', 'avg_LL_delta'])
    nouns = df['noun'].unique()
    for noun in nouns:
        df_noun = df[df['noun']==noun]
        noun_total = df_noun.shape[0]
        num_correct = df_noun[df_noun['predicted']==df_noun['semantic role']].shape[0]
        overlap = df_noun[df_noun['predicted']==noun_pred_info[noun]['predicted']].shape[0]
        avg_LL_delta = df_noun['LL_delta'].mean()
        row = [noun, noun_total, num_correct/noun_total, overlap/noun_total, avg_LL_delta]
        df_noun_acc.loc[len(df_noun_acc.index)] = row
    df_noun_acc = df_noun_acc.sort_values(['noun'])
    df_noun_acc.to_csv(dir + '/noun_accuracy_intr.csv', index=False)
    return

def calculate_per_verb_accuracy(df, dir):
    df_verb_acc = pd.DataFrame(columns=['verb', 'num_agents', 'num_patients', 'agent_acc', 'patient_acc', 'overall_acc', 'agent_avg_LL_delta', 'patient_avg_LL_delta'])
    verbs = df['verb'].unique()
    for verb in verbs:
        df_verb = df[df['verb']==verb]
        verb_total = df_verb.shape[0]
        num_correct = df_verb[df_verb['predicted']==df_verb['semantic role']].shape[0]
        df_verb_agents = df_verb[df_verb['semantic role']=='agent']
        agent_avg_LL_delta = df_verb_agents['LL_delta'].mean()
        df_verb_patients = df_verb[df_verb['semantic role']=='patient']
        patient_avg_LL_delta = df_verb_patients['LL_delta'].mean()
        num_agents = df_verb_agents.shape[0]
        num_patients = df_verb_patients.shape[0]
        num_correct_agents = df_verb_agents[df_verb_agents['predicted'] == 'agent'].shape[0]
        num_correct_patients = df_verb_patients[df_verb_patients['predicted'] == 'patient'].shape[0]
        row = [verb, num_agents, num_patients, num_correct_agents/num_agents, num_correct_patients/num_patients, num_correct/verb_total, agent_avg_LL_delta, patient_avg_LL_delta]
        df_verb_acc.loc[len(df_verb_acc.index)] = row
    df_verb_acc = df_verb_acc.sort_values(['verb'])
    df_verb_acc.to_csv(dir + '/verb_accuracy_intr.csv', index=False)
    return           

def calculate_accuracy_f1(df):
    num_total = df.shape[0]
    num_correct = df[df['predicted'] == df['semantic role']].shape[0]
    num_thresh_correct = df[df['thresh_pred'] == df['semantic role']].shape[0]
    df_agents = df[df['semantic role'] == 'agent']
    num_agents = df_agents.shape[0]
    df_patients = df[df['semantic role'] == 'patient']
    num_patients = df_patients.shape[0]
    num_correct_agents = df_agents[df_agents['predicted'] == 'agent'].shape[0]
    num_correct_patients = df_patients[df_patients['predicted'] == 'patient'].shape[0]
    macro_f1 = metrics.f1_score(df['semantic role'].to_numpy(), df['predicted'].to_numpy(), average='macro')
    thresh_macro_f1 = metrics.f1_score(df['semantic role'].to_numpy(), df['thresh_pred'].to_numpy(), average='macro')
    acc = [num_correct_agents/num_agents, num_correct_patients/num_patients, num_correct/num_total, macro_f1, num_thresh_correct/num_total, thresh_macro_f1]
    return acc

def get_noun_info(noun_score_path):
    noun_pred_info = {}
    df_noun_stats = pd.read_csv(noun_score_path)
    for idx, row in df_noun_stats.iterrows():
        noun_pred_info[row['noun']] = {'LL_delta': row['LL_delta'], 'predicted': row['predicted'], 'human_rating': row['human_rating'], 'subj_ratio': row['subj_ratio']}
    return noun_pred_info    

def corr_with_noun_stats(df, noun_pred_info):
    """
    Inputs:
    -df: intr scores for a model and 
    -noun_pred_info: dict with LL_delta, predicted label, human rating, and subj ratio for each noun in dataset
    Outputs:
    -list containing 
        1. correlation between intr LL_delta and noun LL_delta
        2. correlation between intr LL_delta and human ratings
        3. correlation between intr LL_delta and ngrams
    """
    df['noun_LL_delta'] = df.apply(lambda x: (noun_pred_info[x['noun']]['LL_delta']), axis=1)
    df['noun_human_rating'] = df.apply(lambda x: (noun_pred_info[x['noun']]['human_rating']), axis=1)
    df['noun_subj_ratio'] = df.apply(lambda x: (noun_pred_info[x['noun']]['subj_ratio']), axis=1)
    
    corr_noun_LL = np.corrcoef(df['LL_delta'].to_numpy(), df['noun_LL_delta'].to_numpy())[0,1]
    corr_noun_human = np.corrcoef(df['LL_delta'].to_numpy(), df['noun_human_rating'].to_numpy())[0,1]
    df.dropna(inplace=True)
    corr_subj_ratio = np.corrcoef(df['LL_delta'].to_numpy(), df['noun_subj_ratio'].to_numpy())[0,1]
    corrs = [corr_noun_LL, corr_noun_human, corr_subj_ratio]
    return corrs

def main():
    args = parser.parse_args()
    score_folder = args.score_folder
    exp_folders = glob(score_folder + '*')
    for exp in exp_folders:
        print(exp)
        exp_dir = os.path.join(exp.split('/')[-2], exp.split('/')[-1])
        exp_results_dir = os.path.join('../results/', exp_dir)
        print(exp_results_dir)
        if not os.path.exists(exp_results_dir):    
            os.makedirs(exp_results_dir)
        df_corr = pd.DataFrame(columns=['model', 'corr_noun_LL', 'corr_human', 'corr_subj'])
        df_acc = pd.DataFrame(columns=['model', 'agent_acc', 'patient_acc', 'overall_acc', 'macro_f1', 
                                       'thresh_overall_acc', 'thresh_macro_f1', 'min_LL', 'max_LL', 'thresh'])
        intr_folder = exp + '/intr_scores/*'
        intr_score_files = glob(intr_folder)
        for filepath in intr_score_files:
            thresh_values = format_scores(filepath)
            filename = os.path.basename(filepath)
            noun_score_path = (exp + '/noun_scores/' + filename)
            model_name = os.path.splitext(filename)[0]
            df = pd.read_csv(filepath)
            model_results_dir = os.path.join(exp_results_dir, model_name)
            if not os.path.exists(model_results_dir):    
                os.makedirs(model_results_dir)
            noun_pred_info = get_noun_info(noun_score_path)
            calculate_noun_accuracy_and_overlap(df, noun_pred_info, model_results_dir)
            calculate_per_verb_accuracy(df, model_results_dir)
            correlations = corr_with_noun_stats(df, noun_pred_info)
            row = [model_name] + correlations
            df_corr.loc[len(df_corr.index)] = row
            accuracies = calculate_accuracy_f1(df)
            row = [model_name] + accuracies + thresh_values
            df_acc.loc[len(df_acc.index)] = row 
        df_corr.set_index('model', inplace=True)
        df_corr.rename(index = {'gpt2-x': 'gpt2-small'}, inplace=True)
        df_corr.sort_values(by='model', key=lambda x: x.map(model_order), inplace=True)
        df_corr.to_csv(exp_results_dir + '/intr_corr.csv')
        df_acc.set_index('model', inplace=True)
        df_acc.rename(index = {'gpt2-x': 'gpt2-small'}, inplace=True)
        df_acc.sort_values(by='model', key=lambda x: x.map(model_order), inplace=True)
        df_acc.to_csv(exp_results_dir + '/intr_acc.csv')
        
if __name__ == "__main__":
    main() 