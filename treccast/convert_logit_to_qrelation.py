import collections
import tensorflow.compat.v1 as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-flogits", "--path_false_logit", type=str)
parser.add_argument("-tlogits", "--path_true_logit", type=str)
parser.add_argument("-score", "--path_score", type=str)
parser.add_argument("-qqpair", "--path_queries_autoregressive_pair", type=str)
parser.add_argument("-topk", default=1000, type=int)
parser.add_argument("--resoftmax", action="store_true", default=True)
#parser.add_argument("--trec", action="store_true", default=True)
args = parser.parse_args() 

def convert_logit_to_prob(args):

    with tf.io.gfile.GFile(args.path_score, 'w') as f, \
    tf.io.gfile.GFile(args.path_true_logit, "r") as true_logits, \
    tf.io.gfile.GFile(args.path_false_logit, "r") as false_logits:

        for i, (true_logit, false_logit) in enumerate(zip(true_logits, false_logits)):
            true_prob = np.exp(float(true_logit))
            false_prob = np.exp(float(false_logit))
            sum = true_prob + false_prob
            
            if args.resoftmax:
                true_prob = true_prob / sum
                false_prob = false_prob / sum

            f.write("{:.16f}\t{:.16f}\t{:.16f}\n".format(
                true_prob, false_prob, np.add(true_prob, false_prob)))

            if i % 1000000 == 0:
                print("[Re-ranker] {} query-passage pair had been scored.".format(i))

def ranklist_fusion(args):
    pass

if tf.io.gfile.exists(args.path_runs) is False:
  print("Invalid path of run file")
  exit(0)

convert_logit_to_prob(args)
print("Score finished")
rerank_runs(args)
print("DONE")
