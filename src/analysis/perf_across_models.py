"""
Formats results from experiments per noun and verb across all models,
    including average LL_delta across nouns and across verbs (split by label)
    and accuracies for intr and trans experiments
Must run intr_analysis.py and trans_analysis.py on experiment folder before running this
"""

import argparse
import pandas as pd
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, default="../results/base/")

def noun_exp(exp, df_noun_stats):
    noun_LL_models = [file for file in glob.glob(exp+'/**/noun_LL_delta.csv', recursive=True)]
    df_noun_LL = pd.DataFrame()
    df_noun_LL['noun'] = df_noun_stats['noun']
    df_noun_LL['semantic role'] = df_noun_stats['semantic role']
    for file in noun_LL_models:
        model_name = file.split('/')[-2]
        df_model = pd.read_csv(file)
        df_model = df_model.sort_values('noun')
        assert df_model['noun'].equals(df_noun_LL['noun'])
        df_noun_LL[model_name] = df_model['avg_LL_delta']
    df_noun_LL.to_csv(os.path.join(exp,'avg_noun_LL.csv'), index=False)

def intr_exp(exp, df_noun_stats):
    noun_model_results = [file for file in glob.glob(exp+'/**/noun_accuracy_intr.csv', recursive=True)]
    verb_model_results = [file for file in glob.glob(exp+'/**/verb_accuracy_intr.csv', recursive=True)]
    df_noun_acc_intr = pd.DataFrame()
    df_noun_acc_intr['noun'] = df_noun_stats['noun']
    df_noun_acc_intr['semantic role'] = df_noun_stats['semantic role']
    df_verb_acc_intr = pd.DataFrame()
    for file in noun_model_results:
        model_name = file.split('/')[-2]
        df_model = pd.read_csv(file)
        df_model = df_model.sort_values('noun')
        assert df_model['noun'].equals(df_noun_acc_intr['noun'])
        df_noun_acc_intr[model_name + '_acc'] = df_model['accuracy']
        df_noun_acc_intr[model_name + '_avg_LL_delta'] = df_model['avg_LL_delta']
    df_noun_acc_intr.to_csv(os.path.join(exp,'intr_per-noun.csv'), index=False)
    for file in verb_model_results:
        model_name = file.split('/')[-2]
        df_model = pd.read_csv(file)
        df_model = df_model.sort_values('verb')
        df_verb_acc_intr['verb'] = df_model['verb']
        df_verb_acc_intr[model_name + '_acc'] = df_model['overall_acc']
        df_verb_acc_intr[model_name + '_agent_acc'] = df_model['agent_acc']
        df_verb_acc_intr[model_name + '_patient_acc'] = df_model['patient_acc']
        df_verb_acc_intr[model_name + '_agent_avg_LL_delta'] = df_model['agent_avg_LL_delta']
        df_verb_acc_intr[model_name + '_patient_avg_LL_delta'] = df_model['patient_avg_LL_delta']
    df_verb_acc_intr.to_csv(os.path.join(exp,'intr_per-verb.csv'), index=False)
        
def trans_exp(exp, df_noun_stats):
    noun_subj_model_results = [file for file in glob.glob(exp+'/**/noun_accuracy_trans_subj.csv', recursive=True)]
    verb_subj_model_results = [file for file in glob.glob(exp+'/**/verb_accuracy_trans_subj.csv', recursive=True)]
    noun_obj_model_results = [file for file in glob.glob(exp+'/**/noun_accuracy_trans_obj.csv', recursive=True)]
    verb_obj_model_results = [file for file in glob.glob(exp+'/**/verb_accuracy_trans_obj.csv', recursive=True)]
    df_noun_acc_trans = pd.DataFrame()
    df_noun_acc_trans['noun'] = df_noun_stats['noun']
    df_noun_acc_trans['semantic role'] = df_noun_stats['semantic role']
    df_verb_acc_trans = pd.DataFrame()
    for file_subj in noun_subj_model_results:
        for file_obj in noun_obj_model_results:
            model_name = file_subj.split('/')[-2]
            df_subj_model = pd.read_csv(file_subj)
            df_subj_model = df_subj_model.sort_values('noun')
            df_obj_model = pd.read_csv(file_obj)
            df_obj_model = df_obj_model.sort_values('noun')
            assert df_subj_model['noun'].equals(df_noun_acc_trans['noun'])
            assert df_obj_model['noun'].equals(df_noun_acc_trans['noun'])
            df_noun_acc_trans[model_name + '_subj_acc'] = df_subj_model['accuracy']
            df_noun_acc_trans[model_name + '_subj_avg_LL_delta'] = df_subj_model['avg_LL_delta']
            df_noun_acc_trans[model_name + '_obj_acc'] = df_obj_model['accuracy']
            df_noun_acc_trans[model_name + '_obj_avg_LL_delta'] = df_obj_model['avg_LL_delta']
    df_noun_acc_trans.to_csv(os.path.join(exp,'trans_per-noun.csv'), index=False)
    for file_subj in verb_subj_model_results:
        for file_obj in verb_obj_model_results:
            model_name = file_subj.split('/')[-2]
            df_subj_model = pd.read_csv(file_subj)
            df_subj_model = df_subj_model.sort_values('verb')
            df_obj_model = pd.read_csv(file_obj)
            df_obj_model = df_obj_model.sort_values('verb')
            df_verb_acc_trans['verb'] = df_subj_model['verb']
            df_verb_acc_trans[model_name + '_subj_acc'] = df_subj_model['overall_acc']
            df_verb_acc_trans[model_name + '_subj_avg_LL_delta'] = df_subj_model['avg_LL_delta']
            df_verb_acc_trans[model_name + '_obj_acc'] = df_obj_model['overall_acc']
            df_verb_acc_trans[model_name + '_obj_avg_LL_delta'] = df_obj_model['avg_LL_delta']
    df_verb_acc_trans.to_csv(os.path.join(exp,'trans_per-verb.csv'), index=False)
            
def main():
    args = parser.parse_args()
    results_folder = args.results_folder
    exp_folders = glob.glob(results_folder + '*')
    df_noun_stats = pd.read_csv('noun_stats.csv')
    df_noun_stats = df_noun_stats.sort_values('noun')
    df_noun_stats.reset_index(inplace=True)
    for exp in exp_folders: # 1/2
        noun_exp(exp, df_noun_stats)
        intr_exp(exp, df_noun_stats)
        trans_exp(exp, df_noun_stats)
        
if __name__ == "__main__":
    main()