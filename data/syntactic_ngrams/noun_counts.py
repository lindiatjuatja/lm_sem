import pandas as pd
import re
import os
import warnings
from tqdm import tqdm
import argparse
import json

warnings.filterwarnings("ignore")

def format_dataframe(filepath):
    data_file_delimiter = "\t"
    largest_column_count = 0
    with open(filepath, 'r') as temp_f:
        lines = temp_f.readlines()
        for l in lines:
            column_count = len(l.split(data_file_delimiter)) + 1
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count
    column_names = [i for i in range(0, largest_column_count)]
    df = pd.read_csv(filepath, header=None, delimiter=data_file_delimiter, names=column_names, engine='python')
    df = df.iloc[:, :3]
    df = df.rename(columns={0:"head", 1:"ngram", 2:"counts"})
    return df

def count_subj(df_subj, counts):
    for row in df_subj.itertuples(index=False):
        ngram = row.ngram
        row_count = row.counts
        regex = r'\b\w+\/\w+\/nsubj\/\w+\b'
        subj_list = re.findall(regex, ngram)
        for word in subj_list:
            noun = word.split('/')[0]
            if noun.isalpha():
                if noun in counts:
                    old_noun_counts = counts.pop(noun)
                    noun_counts = {"subj": old_noun_counts["subj"] + row_count, "obj": old_noun_counts["obj"]}
                else:
                    noun_counts = {"subj": row_count, "obj": 0}
                counts.update({noun: noun_counts})
    return counts
    
def count_obj(df_obj, counts):
    for row in df_obj.itertuples(index=False):
        ngram = row.ngram
        row_count = row.counts
        regex = r'\b\w+\/\w+\/dobj\/\w+\b'
        obj_list = re.findall(regex, ngram)
        for word in obj_list:
            noun = word.split('/')[0]
            if noun.isalpha():
                if noun in counts:
                    old_noun_counts = counts.pop(noun)
                    noun_counts = {"subj": old_noun_counts["subj"], "obj": old_noun_counts["obj"] + row_count}
                else:
                    noun_counts = {"subj": 0, "obj": row_count}
                counts.update({noun: noun_counts})
    return counts

parser = argparse.ArgumentParser()
parser.add_argument('--folder_digit', type=int)

args = parser.parse_args()

data_dir = 'unzipped/split'

counts = {}     # keys are nouns, values are dicts of the form {subj: x, obj: x}
skipped_files = []
for subdir, dirs, files in os.walk(data_dir):
    if os.path.basename(os.path.normpath(subdir))[0] == str(args.folder_digit):
        print(subdir, flush=True)
        for file in tqdm(files):
            path = os.path.join(subdir, file)
            print(path, flush=True)
            try:
                df = format_dataframe(path)
            except:
                print("unable to format file, skipped: " + file, flush=True)
                skipped_files.append(file)
            else:
                df_subj = df[df['ngram'].str.contains("/nsubj/")]
                counts = count_subj(df_subj, counts)
                df_obj = df[df['ngram'].str.contains("/dobj/")]
                counts = count_obj(df_obj, counts)
        nouns = list(counts.keys())
        sorted_nouns = sorted(nouns)
        sorted_counts = {i: counts[i] for i in sorted_nouns}
        counts_json = json.dumps(sorted_counts, indent=4)
        with open("count_jsons/noun_counts_" + str(args.folder_digit) + ".json", "w") as outfile:
            outfile.write(counts_json)
print(skipped_files, flush=True)
        
        
