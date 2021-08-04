import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-quora", "--path_quora", default="quora_duplicate_questions.tsv", type=str)
parser.add_argument("-out", "--path_output", type=str)
args = parser.parse_args()


def get_duplicate_question_pairs(args):

    quora = open(args.path_quora, 'r')
    data = open(args.path_output, 'w')
    j = 0

    for i, line in enumerate(quora):
        try:
            _, _, _, question_1, question_2, is_duplicate = line.strip().split("\t")
        
            question_1 = question_1.strip('"')
            question_2 = question_2.strip('"')

            if int(is_duplicate) == 1:
                j += 1
                data.write("{}\t{}\n".format(question_1, question_2))
        except:
           pass 

        if i % 1000 == 0:
            print("{} example is processed, {} duplicated pair ".format(i, j))


get_duplicate_question_pairs(args)
print("DONE")
