import argparse
import json

parser = argparse.ArgumentParser()
a = open("/content/y3_automatic_results_1000.v1.0.run", 'r')
b = open("/content/y3_automatic_results_1000.v1.0.run.corrected", 'w')
c = open("/content/2021_automatic_evaluation_topics_v1.0.tsv", 'r')
d = open("/content/2021_automatic_evaluation_topics_v1.0.tsv.corrected", 'w')

flag = -1

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

for topic in json_data:
    topicid = topic['number']
    for turn in topic['turn']:
        qid = str(topicid) + "_" + str(turn['number'])
        docid = str(turn['canonical_result_id']) + "-" + str(turn['passage_id'])
        u.write("{}\t{}\n".format(qid, docid))

u.close()
data.close()
