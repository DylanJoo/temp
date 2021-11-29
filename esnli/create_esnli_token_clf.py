import re
import json
import string
import spacy
import argparse
import collections
from spacy.lang.en import English

parser = argparse.ArgumentParser()
parser.add_argument("-out", "--path_output", type=str)
# parser.add_argument("-sentA", "--path_sentenceA", type=str)
# parser.add_argument("-sentB", "--path_sentenceB", type=str)
parser.add_argument("-highlightA", "--path_highlightA", type=str)
parser.add_argument("-highlightB", "--path_highlightB", type=str)
parser.add_argument("-label", "--path_labels", type=str)
parser.add_argument("-class", "--class_selected")
args = parser.parse_args()

nlp = English()

def keyword_extraction(tgtA, tgtB, highlightA=False):
    """ Convert the "starred marks" to the separated list of elements. 

    Note: for simplicity, default highlightA is False, only extract highlight B.
    """
    p_highlight = re.compile(r"[\*].*?[\*]")
    p_punct = re.compile("[" + re.escape(string.punctuation) + "]")
    
    # convert the star labels to tokens (sentenceA/highlightA)
    tokens_A_hl, tokens_B_hl = [], []

    if highlightA:
        findings = p_highlight.findall(tgtA)
        for token in findings:
            token = p_punct.sub("", token)
            tokens_A_hl += [token]

    # convert the star labels to tokens (sentenceB/highlightB)
    findings = p_highlight.findall(tgtB)
    for token in findings:
        token = p_punct.sub("", token)
        tokens_B_hl += [token]

    tgtA = tgtA.replace("*", "")
    tgtB = tgtB.replace("*", "")
    tokens_A = [tok.text for tok in nlp(tgtA)]
    tokens_B = [tok.text for tok in nlp(tgtB)]

    return {'sentA': ' '.join(tokens_A), 
            'sentB': ' '.join(tokens_B), 
            'keywordA': tokens_A_hl, 
            'keywordB': tokens_B_hl}

def create_highlight_list(args):

    def readlines(filename):
        f = open(filename, 'r').readlines()
        data = list(map(lambda x: x.strip(), f))
        return data

    data = collections.OrderedDict()
    data['highlightA'] = readlines(args.path_highlightA)
    data['highlightB'] = readlines(args.path_highlightB)
    data['label'] = readlines(args.path_labels)

    if args.class_selected != 'all':
        data['highlightA'] = [h for (h, l) in zip(data['highlightA'], data['label']) \
                if l in args.class_selected]
        data['highlighB'] = [h for (h, l) in zip(data['highlightB'], data['label']) \
                if l in args.class_selected]
        data['label'] = [l for  l in data['label'] if l in args.class_selected]

    with open(args.path_output, 'w') as f:
        for idx, (hla, hlb) in enumerate(zip(data['highlightA'], data['highlightB'])):
            example = keyword_extraction(tgtA=hla, tgtB=hlb)
            f.write(json.dumps(example) + '\n')

create_highlight_list(args)
print('DONE')
