import json
import argparse
import io

parser = argparse.ArgumentParser()
parser.add_argument("-conv_qa", "--path_conv_qa", default="train_convqa.json", type=str)
parser.add_argument("-quac_out", "--path_quac_output", default="QuAC_span_qa.train.tsv", type=str)
args = parser.parse_args()

def convert(args):

    conv_qa = json.load(open(args.path_conv_qa, 'r'))
    fout = open(args.path_quac_output, 'w')

    for i, (quac_turn_id, quac_dict) in enumerate(conv_qa.items()):
        context = quac_dict['context']
        question = quac_dict['question']
        answer = quac_dict['answer']

        fout.write("Question: {} Context: {} Answer:\t{}\n".format(
            question, context, answer))
        if i % 1000 == 0:
            print("{} QA example finished...".format(i))

convert(args)
print("Done")