import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-runs1", "--baseline_run", type=str)
parser.add_argument("-runs2", "--reference_run", type=str)
parser.add_argument("-rels", "--groundtruth_trec", type=str)
args = parser.parse_args()

def convert_run_to_dict(run_file):

    runs = collections.defaultdict(list)
    for i, line in enumerate(run_file):
        qid, pid, rank = line.strip().split('\t')
        runs[qid].append((pid, rank))
    
    for i, (qid, plist) in enumerate(runs.items()):
        runs[qid] = [pid for (pid, rank) in sorted(plist, key=lambda x: x[1])]

    return runs

def eval_relevance(args):

    baseline = open(args.baseline_run, 'r')
    reference = open(args.reference_run, 'r')
    relevance = open(args.groundtruth_trec, 'r') 
    eval_out = open("evaluation.tsv", 'w')
    
    overall_eval = collections.defaultdict(list)
    WTL = collections.defaultdict(int)
    baseline = convert_run_to_dict(baseline)
    reference = convert_run_to_dict(reference)

    for i, line in enumerate(relevance):
        qid, _, pid, _ = line.strip().split()
        try:
            rank_baseline = baseline[qid].index(pid) + 1
            overall_eval['baseline'].append(rank_baseline)
        except:
            rank_baseline = -1
        
        try:
            rank_reference = reference[qid].index(pid) + 1
            overall_eval['reference'].append(rank_reference)
        except:
            rank_reference = -1
        
        if rank_baseline < rank_reference:
            WTL['lose'] +=1
        elif rank_baseline > rank_reference:
            WTL['win'] += 1
        else:
            WTL['tie'] += 1

        eval_out.write("{}\t{}\t{}\t{}\n".format(qid, pid, rank_baseline, rank_reference))

    print("Total query: {}\nBasline hit: {}\nReference hit: {}".format(
        i+1, len(overall_eval['baseline']), len(overall_eval['reference'])))
    print(WTL.items())

eval_relevance(args)
print("DONE")

