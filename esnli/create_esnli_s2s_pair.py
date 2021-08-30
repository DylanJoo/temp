import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument("-premise", "--path_esnli_premise", default="s1.train", type=str)
parser.add_argument("-hypothesis", "--path_esnli_hypothesis", default="s2.train", type=str)
parser.add_argument("-label", "--path_esnli_labels", default="UNK_freq_15_preproc1_expl_1_label.train", type=str)
parser.add_argument("-out", "--path_output", type=str)
args = parser.parse_args()


def convert_label_to_dict(args):

    data = open(args.path_esnli_labels, 'r')
    targets = collections.OrderedDict()

    for i, line in enumerate(data):
        label, explanation = line.strip().split(' ', 1)
        targets[i] = {"label": label, "explanation": explanation}

    print("Total number of targets: {}".format(len(targets)))
    return targets



def create_sent_pairs(args):

    targets = convert_label_to_dict(args)
    output = open(args.path_output, 'w')

    with open(args.path_esnli_premise) as premise, open(args.path_esnli_hypothesis) as hypothesis:
        for i, (line_p, line_h) in enumerate(zip(premises, hypothesis)):

            text_p = line_p.strip()
            text_h = line_h.strip()
            label = targets[i]['label']
            explanation = targets[i]['explanation']
            
            # ESNLI provides several type of way to learn with supervised.
            # 1) Explanation
            example_esnli = "Hypothesis: {} Premise: {} Relation:\t{}\n".format(
                    text_h, text_p, explanation)
            
            # 2) Relation label 
            # example_esnli = "Hypothesis: {} Premise: {} Relation:\t{}".format(
                    # text_h, text_p, label)
            
            output.write(example_esnli)


create_sent_pairs(args)
print("DONE")

