# !/bin/bash

# Beware! Running this script will charge your OpenAI account!
# You need an OpenAI API key to run.
model='text-davinci-003'
example_number='1'
experiment_name='base'
example_folder="../examples/${example_number}"
score_folder="../../scores/${experiment_name}/${example_number}"
data_path='../../prompts.json'

mkdir -p $score_folder
mkdir -p $score_folder'/noun_scores/'
mkdir -p $score_folder'/intr_scores/'
mkdir -p $score_folder'/trans_subj_scores/'
mkdir -p $score_folder'/trans_obj_scores/'

echo $model $example_number
python ../score_noun_intr_trans-gpt3.py \
    --data_path $data_path \
    --example_folder $example_folder \
    --score_folder $score_folder \
    --model_name $model 