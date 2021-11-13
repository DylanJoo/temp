import os
import re
import string
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument("-sentA", "--path_esnli_sentenceA", default="sentenceA.txt", type=str)
parser.add_argument("-sentB", "--path_esnli_sentenceB", default="sentenceB.txt", type=str)
parser.add_argument("-label", "--path_esnli_labels", default="label.txt", type=str)
parser.add_argument("-highlightA", "--path_esnli_highlightA", default="highlightA.txt", type=str)
parser.add_argument("-highlightB", "--path_esnli_highlightB", default="highlightB.txt", type=str)
parser.add_argument("-explanation", "--path_esnli_explanation", default="explanation.txt", type=str)
parser.add_argument("-out", "--path_output", type=str)
parser.add_argument("-target", "--target_type", type=str)
parser.add_argument("-class", "--class_selected", default="all", type=str)
parser.add_argument("--reverse", action="store_true", default=False)
args = parser.parse_args()

def read_esnli(args):
    
    def readlines(filename):
        f = open(filename, 'r').readlines()
        data = list(map(lambda x: x.strip(), f))
        return data

    data = collections.OrderedDict()
    data['sentA'] = readlines(args.path_esnli_sentenceA)
    data['sentB'] = readlines(args.path_esnli_sentenceB)
    data['highlightA'] = readlines(args.path_esnli_highlightA)
    data['highlightB'] = readlines(args.path_esnli_highlightB)
    data['label'] = readlines(args.path_esnli_labels)

    if os.path.exists(args.path_esnli_explanation):
        data['explanation'] = readlines(args.path_esnli_explanation)

    # highlight generation (whole sentence or keyphrase)
    # if args.target_type == 'highlight_plus':
    #     data['highlightA'] = list(map(prepare_highlight_plus, data['highlightA'], data['label']))
    #     data['highlightB'] = list(map(prepare_highlight_plus, data['highlightB'], data['label']))
    # elif args.target_type == 'extraction':
    #     data['highlightA'] = list(map(extract_marked_token, data['highlightA']))
    #     data['highlightB'] = list(map(extract_marked_token, data['highlightB']))

    # example filtering 
    if args.class_selected != 'all':
        data['sentA'] = [h for (h, l) in zip(data['sentA'], data['label']) if l in args.class_selected]
        data['sentB'] = [h for (h, l) in zip(data['sentB'], data['label']) if l in args.class_selected]
        data['highlightA'] = [h for (h, l) in zip(data['highlightA'], data['label']) if l in args.class_selected]
        data['highlightB'] = [h for (h, l) in zip(data['highlightB'], data['label']) if l in args.class_selected]
        data['label'] = [l for  l in data['label'] if l in args.class_selected]

    return data

def extract_marked_token(sentence): 
    sentence = sentence.strip()
    token_list = []
    p_highlight = re.compile(r"[\*].*?[\*]")
    p_punct = re.compile("[" + re.escape(string.punctuation) + "]")
    findings = p_highlight.findall(sentence)

    for token in findings:
        token = p_punct.sub("", token)
        token_list += [token]

    if len(token_list) == 0:
        return "None"
    else:
        return " ||| ".join(token_list)

def prepare_highlight_plus(sentence, label):

    if label == "entailment":
        sentence = sentence.replace("*", "+")
    elif label == "neutral":
        sentence = sentence.replace("*", "=")
    elif label == "contradiction":
        sentence = sentence.replace("*", "-")
    return sentence

def remove_positive_highlight(sentence, label):

    if label != "contradiction":
        sentence = sentence.replace("*", "")
    return sentence

def create_sent_pairs(args):

    data = read_esnli(args)
    data_length = len(data['label'])
    if all(len(features) == len(data['label']) for features in data.values()) is False:
        print("Inconsist length of data in the dictionary")
        exit(0)

    output = open(args.path_output, 'w')

    # ESNLI provides several type of way to learn with supervised.
    for idx in range(data_length):
        # 1) Naive NLI supervised task
        if args.target_type == "clf":
            example = "Hypothesis: {} Premise: {} Relation:\t{}\n".format(
                    data['sentA'][idx], data['sentB'][idx], 
                    data['label'][idx])
        elif args.target_type == "expl":
            example = "Hypothesis: {} Premise: {} Relation:\t{}\n".format(
                    data['sentA'][idx], data['sentB'][idx], 
                    data['explanation'][idx])
        elif args.target_type == "clf_expl":
            example = "Hypothesis: {} Premise: {} Relation:\t{} ||| {}\n".format(
                    data['sentA'][idx], data['sentB'][idx], 
                    data['label'][idx], data['explanation'][idx])

        # 2) Highlighter supervised task
        if args.target_type == "highlight":
            example = "Sentence1: {} Sentence2: {} Highlight:\t{}\n".format(
                    data['sentA'][idx], data['sentB'][idx],
                    data['highlightB'][idx])

            if args.reverse:
                example += "Sentence1: {} Sentence2: {} Highlight:\t{}\n".format(
                        data['sentB'][idx], data['sentA'][idx],
                        data['highlightA'][idx])

        elif args.target_type == "highlight_ctrl":
            if data['lable'][idx] == 'contradition':
                example = "Sentence1: {} Sentence2: {} Contradiction:\t{}\n".format(data['sentA'][idx], data['sentB'][idx], data['highlightB'][idx])
            if data['lable'][idx] == 'entailment':
                example = "Sentence1: {} Sentence2: {} Entailment:\t{}\n".format(data['sentA'][idx], data['sentB'][idx], data['highlightB'][idx])


        output.write(example)


create_sent_pairs(args)
print("DONE")

