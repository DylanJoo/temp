import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-queries", "--path_queries", type=str)
parser.add_argument("-out", "--path_output", type=str)
args = parser.parse_args()

def load_queries(path):
    query_dict = collections.defaultdict(list)
    data = open(args.path_queries, 'r')
    for line in data:
        qid, query = line.strip().split('\t')
        topic_id, turn_id = qid.split('_')
        query_dict[topic_id].append((turn_id, query))
    return query_dict

def aggregate_lag_one_query(args):
    output = open(args.path_output, 'w')
    queries = load_queries(args.path_queries)
    topic_id = -1
    for topic_id, topic in queries.items():
        for (_, turn_prev), (_, turn_current) in zip(topic, topic[1:]):
            output.write("{} ||| {} Follow-up:\n".format(turn_prev, turn_current))
        output.write("_ ||| _ Follow-up:\n")

aggregate_lag_one_query(args)
print("DONE")
