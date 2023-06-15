#!/bin/bash
#SBATCH --job-name=lm-agentivity
#SBATCH --output ./slurm-out/gpt2-bloom-score_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32GB
#SBATCH --exclude tir-0-[32,36],tir-1-[32,36],tir-1-28
#SBATCH --time 1-00:00:00

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate lm_sem

# change working dir depending on where you are running this script from
working_dir='/home/ltjuatja'

filename="${working_dir}/lm-agentivity/src/models.txt"
example_number='1'
experiment_name='base'
example_folder="${working_dir}/lm-agentivity/src/get_model_probs/examples/${example_number}"
score_folder="${working_dir}/lm-agentivity/src/scores/${experiment_name}/${example_number}"
data_path="${working_dir}/lm-agentivity/src/prompts.json"

HOST=`hostname`
echo ${HOST}

mkdir -p /compute/${HOST::-4}/${USER}/huggingface/cache/model

mkdir -p ${score_folder}
mkdir -p ${score_folder}/noun_scores/
mkdir -p ${score_folder}/intr_scores/
mkdir -p ${score_folder}/trans_subj_scores/
mkdir -p ${score_folder}/trans_obj_scores/

while read line; do
    IFS=' '
    read -ra Arr <<< $line
    model=${Arr[0]}
    size=${Arr[1]}
    echo $model $size
    python ${working_dir}/lm-agentivity/src/get_model_probs/score_noun_intr_trans-gpu.py \
        --data_path $data_path \
        --example_folder $example_folder \
        --score_folder $score_folder \
        --model_name $model \
        --model_size $size \
        --cache_dir /compute/${HOST::-4}/${USER}/huggingface/cache/model    # this may also need to be changed
done < $filename
