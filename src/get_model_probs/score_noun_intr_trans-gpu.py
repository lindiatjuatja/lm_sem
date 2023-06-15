""" Script to score {'agent', 'patient'} given the:
-noun
-associated intransitive/middle ("This [noun] [verb]+s [adverb]")
-transitive sentence with noun in subj position ("This [noun] [verb]+s something/X [adverb]")
-transitive sentence with noun in obj position ("Something/X [verb]+s this [noun] [adverb]") 
Generates a .csv file for each of the four above
Outputs: score_folder/model_name-model_size_folder/{nouns, intr, trans-subj, trans-obj}.csv
"""

import argparse
import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)   
torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="prompts.json")
    parser.add_argument("--example_folder", type=str)
    parser.add_argument("--score_folder", type=str, default="scores/")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--model_size", type=str, default="x")
    parser.add_argument("--cache_dir", type=str)
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

def get_model_and_tokenizer(cache_path, model_name, model_size):
    name_and_size = model_name + '-' + model_size
    if model_size == 'x':
        name_and_size = model_name
    if model_name == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(name_and_size, cache_dir=cache_path, torch_dtype="auto", device_map="auto")
        tokenizer = GPT2Tokenizer.from_pretrained(name_and_size, cache_dir=cache_path)
    elif model_name == 'bloom':
        model = BloomForCausalLM.from_pretrained("bigscience/" + name_and_size, cache_dir=cache_path, torch_dtype="auto", device_map="auto") 
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/" + name_and_size, cache_dir=cache_path)
    else:   
        print("Model not available")
    return model,tokenizer

def format_data(data_path):
    with open (data_path, 'r') as f:
        prompts = json.load(f)
    templates = []
    id = 0
    for prompt_template in prompts:
        verb = prompt_template['verb']
        for adverb in prompt_template['adverbs']:
            for noun_and_label in prompt_template['nouns_and_labels']:
                noun = noun_and_label[0]
                theta = noun_and_label[1]
                template = {
                    'id': id,
                    'verb': verb,
                    'noun': noun,
                    'adverb': adverb,
                    'theta': theta
                    }
                templates.append(template)
                id += 1
    return templates

def format_nouns(data_path):
    with open (data_path, 'r') as f:
        prompts = json.load(f)
    unique_nouns_and_labels = []
    for prompt_template in prompts:
        for noun_and_label in prompt_template['nouns_and_labels']:
            if noun_and_label not in unique_nouns_and_labels:
                unique_nouns_and_labels.append(noun_and_label)
    return unique_nouns_and_labels  

def get_LL(prefix, score_text, tokenizer, model):
    template = prefix + score_text
    enc = tokenizer(template, return_tensors="pt")
    enc_len = tokenizer(score_text, return_tensors="pt").input_ids.size(1)
    input_ids = enc.input_ids[:, 0:]
    target_ids = input_ids.clone()
    target_ids[:, :-enc_len] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        LL = outputs[0] * -1
    return LL.item()  

def score_nouns(example_folder, nouns_and_labels, tokenizer, model):
    df = pd.DataFrame(columns=['noun', 'LL_agent', 'LL_patient', 'semantic role', 'predicted'])
    with open(example_folder + '/noun_examples.txt', 'r') as file:
        examples = file.read() + '\n'
    for noun_and_label in tqdm(nouns_and_labels):
        noun = noun_and_label[0]
        label = noun_and_label[1]
        noun_prefix = "noun: " + noun + "\n" + "agent/patient: "
        prefix = examples + noun_prefix
        ll_agent = get_LL(prefix, 'agent', tokenizer, model)
        ll_patient = get_LL(prefix, 'patient', tokenizer, model)
        pred = 'agent'
        if ll_patient > ll_agent:
            pred = 'patient'
        row = pd.DataFrame({
            'noun': noun,
            'LL_agent': ll_agent,
            'LL_patient': ll_patient,
            'semantic role': label,
            'predicted': pred}, index=[0])
        df = pd.concat([row,df])    
    return df

def score_sent_prefix(prefix, sentence, noun, tokenizer, model):
    sent_prefix = "Sentence: " + sentence + "\n"
    question_prefix = "Is " + noun + " an agent or a patient?: "
    prefix += sent_prefix + question_prefix
    ll_agent = get_LL(prefix, 'agent', tokenizer, model)
    ll_patient = get_LL(prefix, 'patient', tokenizer, model)
    pred = 'agent'
    if ll_patient > ll_agent:
        pred = 'patient'
    return ll_agent, ll_patient, pred

def score_intr(example_folder, data, tokenizer, model):
    df = pd.DataFrame(columns=['id', 'verb', 'noun', 'adverb', 'sentence', 'LL_agent', 'LL_patient', 'semantic role', 'predicted'])
    with open(example_folder + '/intr_examples.txt', 'r') as file:
        examples = file.read() + '\n'
    for template in tqdm(data):
        sentence = "This " + template['noun'] + " " + template['verb'] + " " + template['adverb'] + "."
        ll_agent, ll_patient, pred = score_sent_prefix(examples, sentence, template['noun'], tokenizer, model)
        row = pd.DataFrame({
            'id': template['id'],
            'verb': template['verb'],
            'noun': template['noun'],
            'adverb': template['adverb'],
            'sentence': sentence,
            'LL_agent': ll_agent,
            'LL_patient': ll_patient,
            'semantic role': template['theta'],
            'predicted': pred}, index=[0])
        df = pd.concat([row,df])    
    return df

def score_trans_subj(example_folder, data, tokenizer, model):
    df = pd.DataFrame(columns=['id', 'verb', 'noun', 'adverb', 'sentence', 'LL_agent', 'LL_patient', 'semantic role', 'predicted'])
    with open(example_folder + '/trans_examples.txt', 'r') as file:
        examples = file.read() + '\n'
    for template in tqdm(data):
        sentence = "This " + template['noun'] + " " + template['verb'] + " something " + template['adverb'] + "."
        ll_agent, ll_patient, pred = score_sent_prefix(examples, sentence, template['noun'], tokenizer, model)
        row = pd.DataFrame({
            'id': template['id'],
            'verb': template['verb'],
            'noun': template['noun'],
            'adverb': template['adverb'],
            'sentence': sentence,
            'LL_agent': ll_agent,
            'LL_patient': ll_patient,
            'semantic role': 'agent',
            'predicted': pred}, index=[0])
        df = pd.concat([row,df])    
    return df

def score_trans_obj(example_folder, data, tokenizer, model):
    df = pd.DataFrame(columns=['id', 'verb', 'noun', 'adverb', 'sentence', 'LL_agent', 'LL_patient', 'semantic role', 'predicted'])
    with open(example_folder + '/trans_examples.txt', 'r') as file:
        examples = file.read() + '\n'
    for template in tqdm(data):
        sentence = "Something " + template['verb'] + " this " + template['noun'] + " " + template['adverb'] + "."
        ll_agent, ll_patient, pred = score_sent_prefix(examples, sentence, template['noun'], tokenizer, model)
        row = pd.DataFrame({
            'id': template['id'],
            'verb': template['verb'],
            'noun': template['noun'],
            'adverb': template['adverb'],
            'sentence': sentence,
            'LL_agent': ll_agent,
            'LL_patient': ll_patient,
            'semantic role': 'patient',
            'predicted': pred}, index=[0])
        df = pd.concat([row,df])    
    return df

def main():
    args = get_args()
    data_path = args.data_path
    example_folder = args.example_folder
    data = format_data(data_path)
    nouns_and_labels = format_nouns(data_path)
    model_name = args.model_name
    model_size = args.model_size
    cache_path = args.cache_dir
    model,tokenizer = get_model_and_tokenizer(cache_path, model_name, model_size)
    noun_scores = score_nouns(example_folder, nouns_and_labels, tokenizer, model)
    noun_scores.to_csv(args.score_folder + "/noun_scores/" + model_name + "-" + model_size + '.csv', index=False)
    intr_scores = score_intr(example_folder, data, tokenizer, model)
    intr_scores.to_csv(args.score_folder + "/intr_scores/" + model_name + "-" + model_size + '.csv', index=False)
    trans_subj_scores = score_trans_subj(example_folder, data, tokenizer, model)
    trans_subj_scores.to_csv(args.score_folder + "/trans_subj_scores/" + model_name + "-" + model_size + '.csv', index=False)
    trans_obj_scores = score_trans_obj(example_folder, data, tokenizer, model)
    trans_obj_scores.to_csv(args.score_folder + "/trans_obj_scores/" + model_name + "-" + model_size + '.csv', index=False)

if __name__ == "__main__":
    main()
