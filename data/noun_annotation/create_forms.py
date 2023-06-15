from apiclient import discovery
import argparse
import json
from httplib2 import Http
from oauth2client import client, file, tools
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='../../src/prompts.json', help='path to prompt json')
parser.add_argument("--num_forms", default=1, help='number of forms to create')

def make_noun_txt(data_path: str):
    with open (data_path, 'r') as f:
        prompts = json.load(f)
    unique_nouns = []
    f = open ('noun_annotation/nouns.txt', 'w')
    for prompt_template in prompts:
            for noun_and_label in prompt_template['nouns_and_labels']:
                noun = noun_and_label[0]
                if noun not in unique_nouns:
                    unique_nouns.append(noun)
                    f.write(noun + '\n')

def get_noun_list(txt: str):
    f = open(txt, "r")
    list = f.readlines()
    return list

def make_noun_entry(noun: str, idx: int, num_breaks: int):
    noun_entry = { "createItem": {
                        "item": {
                            "title": noun,
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "scaleQuestion": {
                                        "low": 1,
                                        "high": 5,
                                        "lowLabel": "very unlikely to be an agent",
                                        "highLabel": "very likely to be an agent"
                                    }
                                }
                            }
                        },
                        "location": {
                            "index": idx+2+num_breaks
                        }
                    } 
                  }
    return noun_entry
    
def make_section(description: str, idx: int, num_breaks: int):
    page_break = { "createItem": {
                        "item": {
                            "title": "Noun agentivity annotation",
                            "description": description,
                            "pageBreakItem": {
                            }
                        },
                        "location": {
                            "index": idx+2+num_breaks
                        }
                    }
                }
    return page_break
    
def main():
    args = parser.parse_args()
    make_noun_txt(args.data_path)
    noun_list = get_noun_list('nouns.txt')
    with open ('noun_senses.json', 'r') as f:
        noun_senses_list = json.load(f)
    noun_senses = {k: v for d in noun_senses_list for k, v in d.items()}
    
    SCOPES = "https://www.googleapis.com/auth/drive"
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    store = file.Storage('token.json')
    creds = None
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_secrets.json', SCOPES)
        creds = tools.run_flow(flow, store)

    form_service = discovery.build('forms', 'v1', http=creds.authorize(
        Http()), discoveryServiceUrl=DISCOVERY_DOC, static_discovery=False)
    
    annotation_instructions = """ An agent is something that initiates an action, possibly with some degree of volition. In other words, nouns that tend to be agents have a tendency to do things.

A patient is something that undergoes an action and often experiences a change. In other words, nouns that tend to be patients have a tendency to have things done to it.

In this form, you are tasked to annotate how "agentive" you think a noun typically is -- in other words, how likely it is to be an agent or a patient when an action involving both an agent and a patient occur. 

Ex: The plant was watered by John.
The plant = patient
John = agent

Ex: The sun burns John.
The sun = agent
John = patient 

A more formal definition is given by Dowty (1991), who outlines contributing properties of agents and patients:
(1) Contributing properties for the Agent Proto-Role:
— volitional involvement in the event or state — sentience (and/or perception)
— causing an event or change of state in another participant
— movement (relative to the position of another participant)
— (exists independently of the event named by the verb)
(2) Contributing properties for the Patient Proto-Role:
— undergoes change of state
— incremental theme (something that changes incrementally over the course of an event)
— causally affected by another participant
— stationary relative to movement of another participant
— (does not exist independently of the event, or not at all)

For the sake of simplicity, disregard events described by reflexives (such as John shaved himself).

For each of the following nouns, rate it on the following scale:
1 = very unlikely to be an agent
2 = somewhat unlikely to be an agent
3 = no preference between agent and patient
4 = somewhat likely to be an agent
5 = very likely to be an agent
    """
    
    link_to_instructions = """For each of the following nouns, rate it on the following scale:
1 = very unlikely to be an agent
2 = somewhat unlikely to be an agent
3 = no preference between agent and patient
4 = somewhat likely to be an agent
5 = very likely to be an agent

Please refer back to the original instructions, which can be found here: https://docs.google.com/document/d/1beN-_SpqfquhoAK3gJ5Nwx1RI9-7gd75kzhJUjCNmRA/edit?usp=sharing
    """
    
    for i in range(args.num_forms):
        initial_form = { "info": {
                            "title": f'Noun Agentivity Annotation Form (#{i})',
                            "documentTitle": f'noun_annotation_{i}'
                            }     
        }
        result = form_service.forms().create(body=initial_form).execute()
        update = { "requests":  [
                    {
                        "updateFormInfo": {
                            "info": {
                                "description": annotation_instructions
                            },
                            "updateMask": "description"
                        }
                    },
                    { 
                        "createItem": {
                            "item": {
                                "title": "Full name",
                                "questionItem": {
                                    "question": {
                                        "required": True,
                                        "textQuestion": {}
                                    }
                                }
                            },
                            "location": {
                                "index": 0
                            }
                        }
                    },
                    {
                        "createItem": {
                            "item": {
                                "title": "What's your Andrew ID?",
                                "description": "Enter your email address if you are not a CMU affiliate.",
                                "questionItem": {
                                    "question": {
                                        "required": True,
                                        "textQuestion": {}
                                    }
                                }
                            },
                            "location": {
                                "index": 1
                            }
                        }
                    }
                ]}
        entries = []
        random.shuffle(noun_list)
        num_breaks = 0
        for idx, noun in enumerate(noun_list):
            noun = noun.strip()
            if (idx%10 == 0):
                entries.append(make_section(link_to_instructions, idx, num_breaks))
                num_breaks += 1
            if noun in noun_senses:
                sense = noun_senses.get(noun)
                noun = noun + " " + sense
            entries.append(make_noun_entry(noun, idx, num_breaks))
        update["requests"].extend(entries)
        set_questions = form_service.forms().batchUpdate(formId=result["formId"], body=update).execute()
        get_result = form_service.forms().get(formId=result["formId"]).execute()
        
if __name__ == "__main__":
    main()
        
        