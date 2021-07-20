import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--official_dev", type=str)
parser.add_argument("--official_baseline_run", type=str)
parser.add_argument("--trec", action="store_true", default=True)
# y3_automatic_results_1000.v1.0.run
# 2021_automatic_evaluation_topics_v1.0.tsv.corrected

def dev_check(path, trec):
    dev_json = json.load(open(path, 'r'))
    dev_tsv_corrected = open(os.path.join(path, "_corrected"))
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
            if trec:
                dev_tsv_corrected.write("{} {} {} {}\n".format(qid, 0, docid, 1))
            else:
                dev_tsv_corrected.write("{}\t{}\n".format(qid, docid))


def baseline_run_check(path):
    run_baseline = open(path, 'r')
    run_baseline_corrected = open(os.path.join(path, "_corrected"))
    query_set = set() 
    for line in run_baseline:
        qid, q_0, doc_passage_id, rank, score, name = line.strip().split(' ') 
        topicid, turnid = qid.split("_")
        if qid not in query_set:
            query_set.add(qid)
            run_baseline_corrected.write(line)
        else:
            # not in query set
            qid = "{}_{}".format(topicid, int(turnid)+1)
            query_set.add(qid)
            line = "{} Q0 {} {} {} reranking\n".format(qid, doc_passage_id, rank, score))
            run_baseline_corrected.write(line)

dev_check(args.official_dev, args.trec)
baseline_run_check(args.official_baseline_run, args.trec)
print("DONE")
