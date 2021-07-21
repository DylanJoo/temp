import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--official_dev_json", type=str)
parser.add_argument("--official_dev_run", type=str)
parser.add_argument("-pnr", "--passage_number_removal", action="store_true", default=True)
parser.add_argument("--trec", action="store_true", default=True)
args = parser.parse_args()

# y3_automatic_results_1000.v1.0.run
# 2021_automatic_evaluation_topics_v1.0.tsv.correcte

def dev_check(path, pnr):
    dev_json = json.load(open(path, 'r'))
    dev_corrected = open(path + "_corrected.trec", 'w')
    if pnr:
        dev_corrected_doc = open(path + "_doc_corrected.trec", 'w')
    for topic in dev_json:
        topicid = topic['number']
        for i, turn in enumerate(topic['turn']):
            docid = str(turn['canonical_result_id'])
            doc_passage_id = str(turn['canonical_result_id']) + "-" + str(turn['passage_id'])
            turnid = str(turn['number'])
            if i+1 != turn['number']:
                print("[CORRECTION]: {}_{} --> {}_{}".format(
                    topicid, turnid, topicid, i+1))
                turnid = i+1
            qid = str(topicid) + "_" + str(turnid)

            dev_corrected.write("{} {} {} {}\n".format(qid, 0, doc_passage_id, 1))
            dev_corrected.write("{}\t{}\n".format(qid, docid))
            # doc only
            if pnr:
                dev_corrected_doc.write("{} {} {} {}\n".format(qid, 0, docid, 1))
                dev_corrected_doc.write("{}\t{}\n".format(qid, docid))

def baseline_run_check(path):
    run_baseline = open(path, 'r')
    run_baseline_corrected = open(path + "_corrected", 'w')
    query_rank_set = set() 
    for line in run_baseline:
        qid, q_0, doc_passage_id, rank, score, name = line.strip().split(' ') 
        topicid, turnid = qid.split("_")
        query_rank = "{}-{}".format(qid, rank)
        if query_rank not in query_rank_set:
            query_rank_set.add(str(query_rank))
            run_baseline_corrected.write(line)
        else:
            # not in query set
            qid = "{}_{}".format(topicid, int(turnid)+1)
            query_rank = "{}-{}".format(qid, rank)
            query_rank_set.add(query_rank)
            print("[CORRECTION]: {}_{} --> {}".format(topicid, turnid, qid))
            line = "{} Q0 {} {} {} {}\n".format(qid, doc_passage_id, rank, score, name)
            run_baseline_corrected.write(line)

if args.official_dev_json:
    dev_check(args.official_dev_json, args.passage_number_removal)
if args.official_dev_run:
    baseline_run_check(args.official_dev_run)
print("DONE")
