import json
import argparse
import io

parser = argparse.ArgumentParser()
parser.add_argument("-conv_qa", "--path_conv_qa", default="train_convqa.json", type=str)
parser.add_argument("-quac_out", "--path_quac_output", default="QuAC_span_qa.train.tsv", type=str)
parser.add_argument("-orig", "--original_answer", action="store_true", default=False)
args = parser.parse_args()

def convert(args):

    conv_qa = json.load(open(args.path_conv_qa, 'r'))
    fout = open(args.path_quac_output, 'w')

    for i, (quac_turn_id, quac_dict) in enumerate(conv_qa.items()):
        context = quac_dict['context']
        question = quac_dict['question']
        if args.original_answer:
            answer = quac_dict['orig_answer']
        else:
            answer = quac_dict['answer']

        context.replace("CANNOTANSWER", "")
        
        if args.answerable:
            if answer != "I don't know.":
                fout.write("Question: {} Context: {} Answer:\t{}\n".format(
                    question, context, answer))
        else:
            fout.write("Question: {} Context: {} Answer:\t{}\n".format(
                question, context, answer))

        if i % 10000 == 0:
            print("{} QA example finished...".format(i))

convert(args)
print("Done")
