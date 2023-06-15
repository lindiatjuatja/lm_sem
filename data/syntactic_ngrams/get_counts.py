import json
import pandas as pd
import os
from tqdm import tqdm

prompts_path = '../../src/prompts.json'

with open (prompts_path, 'r') as f:
    prompts = json.load(f)
labels = {}
for prompt_template in prompts:
        for noun_and_label in prompt_template['nouns_and_labels']:
            noun = noun_and_label[0].lower()
            if noun not in labels:
                labels[noun] = noun_and_label[1]

counts_df = pd.DataFrame(columns=['noun', 'nsubj', 'dobj', 'subj_ratio', 'label'])
counts_df.noun = labels.keys()
counts_df['nsubj'] = counts_df['nsubj'].fillna(0)
counts_df['dobj'] = counts_df['dobj'].fillna(0)
counts_df['subj_ratio'] = counts_df['dobj'].fillna(0)

print(counts_df.head())

dir = 'count_jsons'
for filename in tqdm(os.listdir(dir)):
    with open(os.path.join(dir,filename)) as json_file:
        counts = json.load(json_file)
        for idx, row in counts_df.iterrows():
            noun = row['noun']
            old_subj_count = row['nsubj']
            old_obj_count = row['dobj']
            if noun in counts:
                noun_counts = counts.get(noun)
                counts_df.at[idx, 'nsubj'] = old_subj_count + noun_counts.get('subj')
                counts_df.at[idx, 'dobj'] = old_obj_count + noun_counts.get('obj')
                counts_df.at[idx, 'label'] = labels[noun]
        counts_df['subj_ratio'] = counts_df['nsubj'] / (counts_df['nsubj'] + counts_df['dobj'])

counts_df = counts_df.sort_values(by='subj_ratio')
counts_df.to_csv('complete_counts.csv', index=False)
