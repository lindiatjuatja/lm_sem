# !/bin/bash

# This script needs to be run within the same directory
# The cache dir should be changed to fit your setup
filename='../../models.txt'
example_number='1'
experiment_name='base'
example_folder="../examples/${example_number}"
score_folder="../../scores/${experiment_name}/${example_number}"
data_path='../../prompts.json'
cache_dir="/projects/ogma3/ltjuatja/huggingface/models/"

mkdir -p $score_folder
mkdir -p $score_folder'/noun_scores/'
mkdir -p $score_folder'/intr_scores/'
mkdir -p $score_folder'/trans_subj_scores/'
mkdir -p $score_folder'/trans_obj_scores/'

while read line; do
    IFS=' '
    read -ra Arr <<< $line
    model=${Arr[0]}
    size=${Arr[1]}
    echo $model $size $example_number 
    python ../score_noun_intr_trans-alt.py \
        --data_path $data_path \
        --example_folder $example_folder \
        --score_folder $score_folder \
        --model_name $model \
        --model_size $size \
        --cache_dir $cache_dir
done < $filename