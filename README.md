# Syntax and Semantics Meet in the "Middle": Probing the Syntax-Semantics Interface of LMs Through Agentivity

The paper associated with this code is [here](https://arxiv.org/abs/2305.18185).

## Noun-verb-adverb prompt templates
Our curated set of noun-verb-adverb combinations is located in `src/prompts.json`. Each entry is a verb with its associated nouns (along with their label, which is determined by its use in the intransitive) and adverbs.

## Getting scores for models
- For BLOOM and GPT-2 models, use either `src/get_model_probs/scripts/score_noun_intr_trans-cpu.sh` or `src/get_model_probs/scripts/score_noun_intr_trans-gpu.sh`. These scripts will run scoring on all models listed in `src/models.txt`.
- For GPT-3 models, use `src/get_model_probs/scripts/score_noun_intr_trans-gpt3.sh`. Note that you will need an OpenAI key to run this. You will also need to specify which model to run in the script, unlike the above (which runs all models from the txt file).
- For both, you can change the ordering of the examples in the prompt prefixes. The prompt prefixes are in `src/get_model_probs/examples`.  
- The resulting scores for running both of the above will be put in the scores directory `src/scores`.

## Analyzing results
Once you have results from running the scoring scripts, you can run analysis on them. These scripts are in `src/analysis`. The outputs of these files will be in `src/results`.

## "Ground truth" data
The data we use as ground truth comparisons (human annotations and corpus stats) is located in `data`. Human rating data as well as the code to generate the Google Forms used to collect this data is located in `data/noun_annotation`.
