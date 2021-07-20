import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--officail_dev", type=str, required=True)
parser.add_argument("--officail_baseline_run", type=str, required=True)

# y3_automatic_results_1000.v1.0.run
# 2021_automatic_evaluation_topics_v1.0.tsv.corrected

flag = -1
def dev_check(path):
    dev_json = json.load(open(path, 'r'))
    dev_tsv_correct
    for topic in dev_json:
        topicid = topic['number']
        for i, turn in enumerate(topic['turn']):
            docid = str(turn['canonical_result_id']) + "-" + str(turn['passage_id'])
            turnid = str(turn['number'])
            if i+1 != turn['number']:
                print("[CORRECTION]: {}_{} --> {}_{}".format(
                    topicid, turnid, topicid, i+1))
                turnid = i+1
            qid = str(topicid) + "_" + str(turnid)
            u.write("{}\t{}\n".format(qid, docid))

for line in a:
    qid, _, _, rank, _, _ = line.strip().split(' ')
    if flag == 1:
        b.write(line.replace("128_5", "128_N"))
    else:
        b.write(line)
    
    if qid == "128_5" and rank == "1000":
        flag = -flag

a.close()
b.close()

flag = -1

for line in c:
    qid, docid = line.strip().split('\t')
        
    if flag == 1:
        d.write(line.replace("128_5", "128_N"))
    else:
        d.write(line)

    if qid == "128_5":
        flag = -flag

c.close()
d.close()
def convert_urels_to_trec(urels_path, output_path):
    with open(output_path, 'w') as f:
        for line in open(urels_path, 'r'):
            qid, docid = line.strip().split('\t')
            f.write("{} {} {} {}\n".format(qid, 0, docid, 1))

convert_urels_to_trec(
    urels_path = "/content/urels.dev.tsv", 
    output_path = "/content/urels.dev.trec"
)

import json

u = open("/content/2021_automatic_evaluation_topics_v1.0.tsv", 'w')
data = open("/content/2021_automatic_evaluation_topics_v1.0.json", 'r')
json_data = json.load(data)


u.close()
data.close()
