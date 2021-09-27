import os
import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-train1", "--path_esnli_train_1", default="esnli_train_1.csv")
parser.add_argument("-train2", "--path_esnli_train_2", default="esnli_train_2.csv")
parser.add_argument("-output_dir", "--path_output_dir", default="preprocessed")
args = parser.parse_args()

def normalized(strings, mode=None):
    try:
        strings = strings.strip()
        strings = re.sub('"', '', strings)
        strings = re.sub(r"\t", "", strings)
        strings = re.sub(r"\n", " ", strings)
        strings = re.sub(r"\s+", " ", strings)
        return strings
    except:
        return False

def read_esnli_csv(args):
    # read the csv table, which includes two file
    if args.path_esnli_train_2:
        data = pd.concat([pd.read_csv(args.path_esnli_train_1),
            pd.read_csv(args.path_esnli_train_2)], axis=0)
        data.reset_index(inplace=True)
        data = data.to_dict()

    else:
        print("Lack of the second esnli training csv")
        exit(0)
        
    # Extract the e-snli data
    with open(os.path.join(args.path_output_dir, 'sentenceA.txt'),'w') as s1, \
         open(os.path.join(args.path_output_dir, 'sentenceB.txt'),'w') as s2, \
         open(os.path.join(args.path_output_dir, 'highlightA.txt'),'w') as h1, \
         open(os.path.join(args.path_output_dir, 'highlightB.txt'), 'w') as h2, \
         open(os.path.join(args.path_output_dir, 'label.txt'), 'w') as lbl:

        for index in data['index']:
            sentA = normalized(data['Sentence1'][index], 'sentA')
            sentB = normalized(data['Sentence2'][index], 'sentB')
            highA = normalized(data['Sentence1_marked_1'][index], 'sentA')
            highB = normalized(data['Sentence2_marked_1'][index], 'sentB')
            label = normalized(data['gold_label'][index], 'label')
           
            if sentA and sentB: 
                s1.write(sentA + "\n")
                s2.write(sentB + "\n")
                if highA is False:
                    highA = sentA
                if highB is False:
                    highB = sentB

                h1.write(highA + "\n")
                h2.write(highB + "\n")
                lbl.write(label + "\n")

            if index % 100000 == 0:
                print("Preprocessing instance: {}".format(index))

        print("Total number of data: {}".format(index))

# Check file and download
if os.path.exists(args.path_esnli_train_1) is False:
    os.system(
        'wget https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_train_1.csv')
    os.system(
        'wget https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_train_2.csv')
    print("Download Finished")

# Build folder
if os.path.exists(args.path_output_dir) is False:
    os.system('mkdir preprocessed')
    print("Directory Created")

# Read and preprocessed the file
read_esnli_csv(args)
print("DONE")
