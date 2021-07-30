import json
import argparse
import collections
import os
from spacy.lang.en import English
import spacy

parser = argparse.ArgumentParser()
parser.add_argument("-canard", "--path_canard", default="train_canard.json", type=str)
parser.add_argument("-quac-tr", "--path_quac_train", default="train_quac.json", type=str)
parser.add_argument("-quac-va", "--path_quac_val", default="val_quac.json", type=str)
parser.add_argument("-conv_qa", "--path_conv_qa", default="train_convqa.json", type=str)
parser.add_argument("-out", "--path_output", default="train_canard+.json", type=str)
parser.add_argument("--spacy", action="store_true", default=False)
parser.add_argument("--keywords", action="store_true", default=False)
args = parser.parse_args()


def convert_quac_to_conv_qa(args):
    data = open(args.path_quac_train, 'r')
    quac = json.load(data)['data']
    data = open(args.path_quac_val, 'r')
    quac = quac + json.load(data)['data']

    conversational_qa = open(args.path_conv_qa, 'w')
    conv_qa_dict = collections.defaultdict()

    for i_topic, topic in enumerate(quac):
        # Topic related data
        background = topic['background']
        title = topic['title']
        section_title = topic['section_title']

        # Turn related data
        content = topic['paragraphs'][0]
        context = content['context']
        for i_turn, turn in enumerate(content['qas']):
            question = turn['question']
            # The "natrual language-like answer'
            #answer = turn['answers'][0]
            # THe "Original" answert from the given context
            orig_answer = turn['orig_answer']['text']
            
            question_id = turn['id']
            conversation_id, turn_id = question_id.split("_q#")
            # Some turn index is wrong in QuAC
            if int(turn_id) != i_turn:
                print("Mismatch found in {}, Corrected from q#{} to q#{}".format(conversation_id, turn_id, i_turn))
                question_id = "{}_q#{}".format(conversation_id, i_turn) 
            conv_qa_dict[question_id] = {"context": context, "question": question, "answer": orig_answer}
    
    json.dump(conv_qa_dict, conversational_qa) 
    print("{} have been converted...".format(args.path_conv_qa))

# History 1: Using the lag "answer " passage for exapnding context
def combine_utterance_response(utterances, responses, pre_history, current_i=-100):
    '''Indicate the i-th turn would consist i-1, i-2, i-3'''
    output = list()
    for i, (u, r) in enumerate(zip(utterances[:-1], responses[:-1])):
        if i >= (current_i - 1):
            output.append(u)
            output.append(r)
            # output.append("{} : {}".format(u, r))
        else:
            output.append(u)
    # If we need the pre-history like title or descprition.
    if len(pre_history):
        output = pre_history + output

    output.append(utterances[-1])
    output = " ||| ".join(output)

    return output

# History 2: Using the lag "entities" of lag turn's answer for expanding context
# def combine_utterance_entity(utterances, responses, pre_history, current_i=-100):
#     '''Indicate the i-th turn would consist i-1, i-2, i-3, 
#     and also using spacy to extract the entites as response'''
#
#     output = list()
#     for i, (u, r) in enumerate(zip(utterances[:-1], responses[:-1])):
#         if i >= (current_i - 3):
#             # output.append(u)
#             # entities = [token.text for token in nlp(r)]
#             entities = [token for token in r.split() if token.lower() not in u.lower()]
#             # print(entities)
#             output.append(u + " ||| " + " ".join(entities))
#         else:
#             output.append(u)
#     
#     # If we need the pre-history like title or descprition.
#     if len(pre_history):
#         output = pre_history + output
#
#     output.append(utterances[-1])
#     output = " ||| ".join(output)
#     
#     return output

# Rewrite 1
def omission_tokens(rewrite_query, raw_query): 
    '''Extract the additional token with information.
    '''
    rewrite_tokens = rewrite_query.lower().split()
    raw_tokens = raw_query.lower().split()

    diff_tokens = [token for token in rewrite_tokens if token not in raw_tokens]
    output = pos(diff_tokens)
    entities = [token.text for token in diff_tokens \
            if (token.pos_ in ['NOUN', 'PROPN'])]
    return " ".join(output)

def merge(args):

    conv_qa = json.load(open(args.path_conv_qa, 'r'))
    canard = json.load(open(args.path_canard, 'r'))
    output = open(args.path_output, 'w')
    quac_id = ""

    for dict_canard in canard:
        if dict_canard['QuAC_dialog_id'] != quac_id:
            new_topic = True
        # Although the history is already containt the Q and A
        history = dict_canard['History'][:2]
        question = dict_canard['Question']
        rewrite = dict_canard['Rewrite']
        quac_id = dict_canard['QuAC_dialog_id']
        turn_id = int(dict_canard['Question_no']) - 1
        quac_turn_id = "{}_q#{}".format(quac_id, turn_id)
        
        # If new topic, clean the answers list, and new another answer list
        if new_topic:
            answers = list()
            questions = list()
            new_topic = False
        qa = conv_qa[quac_turn_id]
        context = qa['context']
        questions += [question]
        answers += [qa['answer']]

        # coreference resolution
        src_coref = combine_utterance_response(questions, answers, history)
        tgt_coref = rewrite

        if args.keywords:
            tgt_coref = omission_tokens(rewrite, question)

        if args.spacy:
            src_coref = ' '.join([tok.text for tok in nlp(src_coref)])
            tgt_coref = ' '.join([tok.text for tok in nlp(tgt_coref)])

        output.write("{}\t{}\n".format(src_coref, tgt_coref))

        # question answering
        #example_qa = "Response: {} Query: {} Rewrite:\n".format()

print(args)
if args.spacy:
    nlp = English()
#if os.path.isfile(args.path_conv_qa) is False:
convert_quac_to_conv_qa(args)
merge(args)
print("DONE")
