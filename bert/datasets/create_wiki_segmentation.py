import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-file", "--text_file", type=str)
parser.add_argument("-csv", "--output_to_csv", action="store_true", default=True)
# parser.add_argument("--balance", "--do_class_balancing")
args = parser.parse_args()

def convert_text_to_segment_pair(args):

    def normalized(strings):
        strings = strings.strip()
        strings = re.sub('"', '', strings)
        strings = re.sub(r"\t", "", strings)
        strings = re.sub(r"\n", " ", strings)
        strings = re.sub(r"\s+", " ", strings)
        return strings

    with open(args.text_file, 'r') as source, \
         open(args.text_file.replace(".txt", ".csv"), 'w') as pair_data:

        if args.output_to_csv:
            pair_data.write("sentA\tsentB\tlabel\n")

        sentences = list()
        t, f = 0, 0
        for i, line in enumerate(source):

            if "========," in line:
                if len(sentences) == 2:
                    pair_data.write("{}\t{}\t{}\n".format(
                        normalized(sentences[0]), 
                        normalized(sentences[1]), "True"))
                    sentences = list() # clean up the sentences buffer
                    t += 1

            else:
                if len(sentences) == 2:
                    pair_data.write("{}\t{}\t{}\n".format(
                        normalized(sentences[0]), 
                        normalized(sentences[1]), "False"))
                    sentences.pop(0) # pop the old sentence
                    f += 1
                
                sentences.append(line)

            if i == 10000:
                print("{} line finished. Class dist.: {}, {}".format(i, t, f))

convert_text_to_segment_pair(args)
print("DONE")
