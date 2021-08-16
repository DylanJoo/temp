import collections
import tensorflow.compat.v1 as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-flogits", "--path_false_logit", type=str)
parser.add_argument("-tlogits", "--path_true_logit", type=str)
parser.add_argument("-score", "--path_score", type=str)
parser.add_argument("-qqpair", "--path_queries_autoregressive_pair", type=str)
parser.add_argument("-query", "--path_queries", type=str)
parser.add_argument("-runs", "--path_runs", type=str)
parser.add_argument("-fusion_runs", "--path_fusion_runs", type=str)
parser.add_argument("-topp", default=1000, type=int, 
                     help="p indicate the top-p relevant passage, which will be fused by follow-up score")
parser.add_argument("--resoftmax", action="store_true", default=True)
#parser.add_argument("--trec", action="store_true", default=True)
args = parser.parse_args() 

def convert_logit_to_prob(args):

    query_followup = collections.defaultdict(float)

    with tf.io.gfile.GFile(args.path_true_logit, "r") as true_logits, \
    tf.io.gfile.GFile(args.path_false_logit, "r") as false_logits, \
    tf.io.gfile.GFile(args.path_queries, "r") as query_file:

        for i, (true_logit, false_logit, query_line) in enumerate(zip(true_logits, false_logits, query_file)):
            true_prob = np.exp(float(true_logit))
            false_prob = np.exp(float(false_logit))
            sum = true_prob + false_prob
            
            if args.resoftmax:
                true_prob = true_prob / sum
                false_prob = false_prob / sum

            qid, qtext = query_line.split('\t')
            if qid.split("_")[1] == "1":
                # No previous query, assume NO followup
                query_followup[qid] = (0, 1, 1)
            else:
                query_followup[qid] = (true_prob, false_prob, np.add(true_prob, false_prob))

            #if i % 1000000 == 0:
            #    print("[Folloup prediction] {} query-passage pair had been scored.".format(i))

def ranklist_fusion(args):

    with tf.io.gfile.GFile(args.path_runs, 'r') as baseline_run_file, \
    tf.io.gfile.GFile(args.path_fusion_runs, 'w') as f:
        
    pass
    # query_candidates = collections.defaultdict(list) 
    # with tf.io.gfile.GFile(args.path_score, 'r') as score_file, \
    # tf.io.gfile.GFile(args.path_runs, "r") as baseline_run_file:
    #
    #     for i, (score_line, run_line) in enumerate(zip(score_file, baseline_run_file)):
    #         true_prob, false_prob, _ = score_line.rstrip().split('\t')
    #         qid, docid, order = run_line.rstrip().split('\t')
    #         if int(order) <= args.topk:
    #             query_candidates[qid].append((docid, true_prob, false_prob))


if tf.io.gfile.exists(args.path_runs) is False:
  print("Invalid path of run file")
  exit(0)

convert_logit_to_prob(args)
print("Score finished")
#ranklist_fusion(args)
print("DONE")
