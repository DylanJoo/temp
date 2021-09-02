import argparse
import tensorflow.compat.v1 as tf
import json
from spacy.lang.en import English
import re

parser = argsparse.ArgumentParser()
parser.add_argument("-json", "--path_json", type=str)
parser.add_argument("-output_dir", "--dir_output", type=str)
parser.add_argument("-query_dir", "--dir_query", type=str)
parser.add_argument("-cano_p", "--canonical_passage", action="store_true", default=False)
parser.add_argument("-cano_a", "--canonical_answer", action="store_true", default=True)
parser.add_argument("-path_answer", "--path_canonical_answer", type=str)
parser.add_argument("-window", "--lag_window", type=int, default=3)
args = parser.parse_args()

nlp = English()

def combine_utterance_response(utterances, responses_p, responses_a=None, current_i=-100):
    '''Indicate the i-th turn would consist i-1, i-2, i-3'''
    if responses_a is None:
        responses_a = [None] * len(utterances)
    output = list()
    for i, (u, rp, ra) in enumerate(zip(utterances[:-1], responses_p[:-1], responses_a[:-1])):
        if i >= (current_i - 3):
            output.append("{} ||| {}".format(u, rp))
        elif ra:
            output.append("{} ||| {}".format(u, ra))
        else:
            output.append("{}".format(u))
    
    output.append("{}".format(utterances[-1]))
    output = " ||| ".join(output)
    output = " ".join([tok.text for tok in nlp(output)])
    return output

def merge_utterance(utterances):
    '''Only use the raw utterances.'''
    utterances = " ||| ".join(utterances)
    utterances = " ".join([tok.text for tok in nlp(utterances)])

    return utterances

def convert_trecwab_to_t5ntr(args): 
    json_path = args.path_json
    urels_path = args.query_dir + "/urels-passage.trec"
    urels_doc_path = args.query_dir + "/urels-document.trec"
    utterance_path = , args.dir_queries + "/utterance.tsv"
    queries_path_auto_cano = args.dir_queries + "/queries_auto-cano.tsv"
    queries_path_manu = args.dir_queries + "/queries_manu.tsv" 
    history_path_auto = args.dir_output + "/history_auto-cano.txt"
    history_path_auto_cano = args.dir_output + "/history_raw.txt"
    answer_cano_json_path = args.path_canonical_answer 

    with tf.io.gfile.GFile(json_path, 'r') as json_file:
        data = json.load(json_file)

    try:
        answers = json.load(tf.io.gfile.GFile(answer_cano_json_path, 'r'))
        answers_flag = True
        print("Using the processed canonical answer")
    except:
        answers_flag = False
        print("Not using the processed canonical answer, using passage")

    with tf.io.gfile.GFile(utterance_path, 'w') as u_file, \
    tf.io.gfile.GFile(history_path_auto, 'w') as ha_file, \
    tf.io.gfile.GFile(history_path_auto_cano, 'w' ) as hac_file, \
    tf.io.gfile.GFile(queries_path_auto_cano, 'w') as qac_file, \
    tf.io.gfile.GFile(queries_path_manu, 'w') as qm_file, \
    tf.io.gfile.GFile(urels_path, 'w') as urels_file ,\
    tf.io.gfile.GFile(urels_doc_path, 'w') as urels_doc_file:
        # <topic> ||| <subtopic> ||| History utterance ||| utterance
        # TopicID-TurnID \t Rewritten query
        # TopicID-TurnID \t Raw utterance
        for topic_idx, topic in enumerate(data):
            topic_id = topic['number']
            # topic_id = data[topic_idx]['number']
            # history = "<topic> ||| <subtopic>" 
            history = ""
            passages_cano = list()
            answers_cano = list()
            utterances = list()

            for turn_idx, turn in enumerate(topic['turn']):
                turn_id = turn['number']
                if turn_id != turn_idx + 1 : # The wrong query index
                    print("Query id correction: {}-{} to {}-{}".format(topic_id, turn_id, topic_id, turn_idx+1))
                    turn_id = turn_idx + 1
                # id
                topic_turn_id = "{}_{}".format(topic_id, turn_id)
                
                # doucment-passage id
                document_passage_id = "{}-{}".format(turn['canonical_result_id'], turn['passage_id'])
                urels_file.write("{} 0 {} 1\n".format(topic_turn_id, document_passage_id))
                urels_doc_file.write("{} 0 {} 1\n".format(topic_turn_id, turn['canonical_result_id']))

                # rewritten using canonical
                # ground truth
                utterance = turn['raw_utterance'].strip()
                rewritten = turn['automatic_rewritten_utterance'].strip()  
                rewritten_gt = turn['manual_rewritten_utterance'].strip()
                passage_cano = turn['passage'].strip()
                answer_cano = answers[topic_turn_id]

                qac_file.write(rewritten + '\n')
                qm_file.write(rewritten_gt + '\n')
                u_file.write("{}\t{}\n".format(topic_turn_id, utterance))
                                
                # canonical passage
                # utterance 
                passages_cano.append(passage_cano)
                answers_cano.append(answer_cano)
                utterances.append(utterance)

                if turn_idx == 0:
                    ha_example = utterance
                    hac_example = utterance

                else:
                    # Only answer
                    ha_example = merge_utterace(utterances)
                    hac_example = merge_utterace(utterances)

                    # Lag 3 passage
                    if args.canonical_passage:
                        hac_example = combine_utterance_response(
                            utterances = utterances, 
                            responses_p = passages_cano, 
                            current_i = turn_idx)
                    
                    # All answers
                    if args.canonical_answer:
                        hac_example = combine_utterance_response(
                            utterances = utterances, 
                            responses_p = answers_cano)

                    # Lag 3 passage and answer for the other 
                    if args.canonical_answer and args.canonical_passage:
                        hac_example = combine_utterance_response(
                            utterances = utterances, 
                            responses_p = passages_cano, 
                            responses_a = answers_cano, 
                            current_i = turn_idx)
                
                ha_file.write(ha_example + "\n")
                hac_file.write(hac_example + "\n")


convert_trecwab_to_t5ntr(args)
print("DONE")
