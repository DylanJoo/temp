import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument("-premise", "--path_esnli_premise", default="s1.train", type=str)
parser.add_argument("-hypothesis", "--path_esnli_hypothesis", default="s2.train", type=str)
parser.add_argument("-label", "--path_esnli_labels", default="UNK_freq_15_preproc1_expl_1_label.train", type=str)
parser.add_argument("-highlight", "--path_esnli_highlight", default="highlight_and_extract.tsv", type=str)
parser.add_argument("-out", "--path_output", type=str)
parser.add_argument("-type", "--target_type", type=str)
args = parser.parse_args()


def convert_label_to_dict(args):

    data = open(args.path_esnli_labels, 'r')
    targets = collections.OrderedDict()

    for i, line in enumerate(data):
        if len(line.strip().split(" ", 1)) != 2:
            label = line.strip()
            explanation = "No explanation."
        else:
            label, explanation = line.strip().split(' ', 1)
        targets[i] = {"label": label, "explanation": explanation}

    print("Total number of targets: {}".format(len(targets)))
    return targets

def convert_highlight_to_dict(args):

    data = open(args.path_esnli_highlight, 'r')
    highlight = collections.OrderedDict()

    for i, line in enumerate(data):
        highlight_p, highlight_h, extract_p, extract_h = line.strip().split('\t')
        highlight[i] = {
                "highlight_hypothesis": highlight_h, "highlight_premise": highlight_p, 
                "extract_hypothesis": extract_h, "extract_premise": extract_p}

    print("Total number of targets: {}".format(len(highlight)))
    return highlight

def preprocess_highlight_plus(highlight_dict, label):
    if label == "entailment":
        highlight_dict['highlight_hypothesis'] = highlight_dict['highlight_hypothesis'].replace("*", "+")
        highlight_dict['highlight_premise'] = highlight_dict['highlight_premise'].replace("*", "+")
    elif label == "neutral":
        highlight_dict['highlight_hypothesis'] = highlight_dict['highlight_hypothesis'].replace("*", "=")
        highlight_dict['highlight_premise'] = highlight_dict['highlight_premise'].replace("*", "=")
    elif label == "contradiction":
        highlight_dict['highlight_hypothesis'] = highlight_dict['highlight_hypothesis'].replace("*", "-")
        highlight_dict['highlight_premise'] = highlight_dict['highlight_premise'].replace("*", "-")

def create_sent_pairs(args):

    targets = convert_label_to_dict(args)
    highlight = convert_highlight_to_dict(args)
    output = open(args.path_output, 'w')

    with open(args.path_esnli_premise) as premise, open(args.path_esnli_hypothesis) as hypothesis:
        for i, (line_p, line_h) in enumerate(zip(premise, hypothesis)):

            text_p = line_p.strip()
            text_h = line_h.strip()
            
            # ESNLI provides several type of way to learn with supervised.
            # 1) Explanation
            if args.target_type == "explanation":
                example_esnli = "Hypothesis: {} Premise: {} Relation:\t{}\n".format(
                        text_h, text_p, targets[i]['explanation'].strip())
            elif args.target_type == "label":
                example_esnli = "Hypothesis: {} Premise: {} Relation:\t{}\n".format(
                        text_h, text_p, targets[i]['label'].strip())
            elif args.target_type == "label_explanation":
                example_esnli = "Hypothesis: {} Premise: {} Relation:\t{} ||| {}\n".format(
                        text_h, text_p, label, targets[i]['explanation'].strip())

            # 2) Highlights
            if args.target_type == "highlight":
                example_esnli = "Hypothesis: {} Premise: {} Highlight:\tHypothesis: {} Premise: {}\n".format(
                        text_h, text_p,
                        highlight[i]['highlight_hypothesis'], highlight[i]['highlight_premise'])

            if args.target_type == "highlight+":
                preprocess_highlight_plus(highlight[i], targets[i]['label'].strip())
                example_esnli = "Hypothesis: {} Premise: {} Highlight:\tHypothesis: {} Premise: {}\n".format(
                        text_h, text_p,
                        highlight[i]['highlight_hypothesis'], highlight[i]['highlight_premise'])

            elif args.target_type == "highlight_extraction":
                example_esnli = "Hypothesis: {} Premise: {} Highlight:\tHypothesis: {} Premise: {}\n".format(
                        text_h, text_p,
                        highlight[i]['extract_hypothesis'], highlight[i]['extract_premise'])

            output.write(example_esnli)


create_sent_pairs(args)
print("DONE")

