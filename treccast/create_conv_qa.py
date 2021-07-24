import json
import argparse
import collections
import os
#from spacy.lang.en import English

parser = argparse.ArgumentParser()
parser.add_argument("-canard", "--path_canard", type=str)
parser.add_argument("-quac", "--path_quac", type=str)
parser.add_argument("-conv_qa", "--path_conv_qa", type=str)
parser.add_argument("-out", "--path_output", type=str)
parser.add_argument("--spacy", action="store_true", default=False)
args = parser.parse_args()


def convert_quac_to_conv_qa(path_quac, path_conv_qa):
    data = open(path_quac, 'r')
    quac = json.load(data)['data']

    conversational_qa = open(path_conv_qa, 'w')
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

            conv_qa_dict[turn['id']] = {"context": context, "question": question, "answer": orig_answer}
    
    json.dump(conv_qa_dict, conversational_qa) 
    print("{} have been converted...".format(path_conv_qa))

# Case 1: Using the lag "answer " passage for exapnding context
def combine_utterance_response(utterances, responses, current_i):
    '''Indicate the i-th turn would consist i-1, i-2, i-3'''
    output = list()
    for i, (u, r) in enumerate(zip(utterances[:-1], responses[:-1])):
        if i >= (current_i - 1):
            output.append(u)
            output.append(r)
        else:
            output.append(u)
    output.append(utterances[-1])

    return " ||| ".join(output)

# case 2: Using the lag "entities" of lag turn's answer for expanding context
def merge(path_conv_qa, path_canard, path_output):

    conv_qa = json.load(open(path_conv_qa, 'r'))
    canard = json.load(open(path_canard, 'r'))
    output = open(path_output, 'w')
    answers = list()

    for dict_canard in canard:
        history = dict_canard['History']
        question = dict_canard['Question']
        rewrite = dict_canard['Rewrite']
        quac_id = dict_canard['QuAC_dialog_id']
        quac_turn_id = "{}_q#{}".format(quac_id, dict_canard['Question_no'])
        
        qa = conv_qa[quac_turn_id]
        context = qa['context']
        answers += [qa['answer']]

        # coreference resolution
        src_coref = combine_utterance_response(history[2:]+[question], answers)
        tgt_coref = rewrite
        if args.spacy:
            src_coref = ' '.join([tok.text for tok in nlp(src_coref)])
            tgt_coref = ' '.join([tok.text for tok in nlp(tgt_coref)])

        output.write("{}\t{}\n".format(src_coref, tgt_coref))

        # question answering
        #example_qa = "Response: {} Query: {} Rewrite:\n".format()

print(args)
#if os.path.isfile(args.path_conv_qa) is False:
convert_quac_to_conv_qa(args.path_quac, args.path_conv_qa)
merge(args.path_conv_qa, args.path_canard, args.path_output)
print("DONE")
