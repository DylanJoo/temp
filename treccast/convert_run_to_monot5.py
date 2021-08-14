import collections
import re
import argparse
import json
from passage_chunker import SpacyPassageChunker

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--queries", type=str, required=True,
                    help="tsv file with two columns, <query_id> and <query_text>")
parser.add_argument("-run", "--run", type=str, required=False,
                    help="tsv file with three/six columns including <query_id>, <doc_id> and <rank>")
parser.add_argument("--trec", action="store_true", default=False,
                    help="Using trec file as fun file")
parser.add_argument("-corpus", "--corpus", type=str, required=True, 
                    help="json/tsv file with <doc_id> and <doc_text> or <passage_id> and <passage_text>")
parser.add_argument("-d", "--doc_level", action="store_true", default=True,  
                    help="Document level identifier, segmented the docuemnt if needed.")
parser.add_argument("-k", "--top_k", type=int, default=1000,
                    help="Selectd top k candidate documents/passages to create the pair.")
parser.add_argument("--output_text_pair", type=str, required=True,
                    help="path to the query-(candidate) passage pair with monot5 format.")
parser.add_argument("--output_id_pair", type=str, required=True,
                    help="path to the query-(candidate) passage pair with ids")
args = parser.parse_args()


def load_queries(path):

    queries_dict = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            qid, query = line.rstrip().split('\t')
            queries_dict[qid] = query
        
            if i % 10000 == 0:
                print('Loading queries...{}'.format(i))

    return queries_dict

def load_corpus(path, doc_level, candidates):

    corpus_type = path.rsplit(".", 1)[-1]
    collection_dict = collections.defaultdict() 
    title_dict = {}

    if corpus_type == "trecweb":
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                doc_dict = json.loads(line)
                docid, doctext, doctitle = doc_dict["id"], doc_dict["contents"], doc_dict["title"]
                
                if docid in candidates:
                    if doc_level:
                        # Document to passage, usong PassageChunker
                        passageChunker.sentence_tokenization(doctext)
                        passages = passageChunker.create_passages()
                    else:
                        # Passage
                        passages = doctext
                    
                    collection_dict[docid] = passages
                    title_dict[docid] = doctitle
                    # Remove the docuemnt in candidate list
                    candidates.remove(docid)

                if i % 100000 == 0:
                    print('Loading collections...{}'.format(i))
                if len(candidates) == 0:
                    break

    # Passage only (document is not yet supported.)
    if corpus_type == "tsv":
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                docid, doctext = line.strip().split('\t')
                collection_dict[docid] = doctext

                if docid in candidates:
                    collection_dict[docid] = doctext

                    # Remove the docuemnt in candidate list
                    candidates.remove(docid)

                if i % 10000 == 0:
                    print('Loading collections...{}'.format(i))
                if len(candidates) == 0:
                    break

    return collection_dict, title_dict

def load_run(path, topk, trec=False):
    candidate_docs = set()
    run = collections.OrderedDict()
    if trec:
        with open(path) as f:
            for i, line in enumerate(f):
                qid, _, docid, rank, _, _ = line.split(' ')
                if int(rank) <= topk:
                    # log the candidate docs 
                    candidate_docs.add(docid)
                    if qid not in run:
                        run[qid] = []
                    run[qid].append((docid, int(rank)))
    else:
        with open(path) as f:
            for i, line in enumerate(f):
                qid, docid, rank = line.split('\t')
                if int(rank) <= topk:
                    # log the candidate docs 
                    candidate_docs.add(docid)
                    if qid not in run:
                        run[qid] = []
                    run[qid].append((docid, int(rank)))

    print('Sorting candidate docs by rank...')
    sorted_run = collections.OrderedDict()
    for i, (qid, doc_ids_ranks) in enumerate(run.items()):
        doc_ids_ranks = sorted(doc_ids_ranks, key=lambda x: x[1])
        docids = [docid for docid, _ in doc_ids_ranks]
        sorted_run[qid] = docids

    print('Loading run...{}'.format(i))
    print("================\nOverlapped Document in this run under top-{}: {}\n================".format(
        topk, len(candidate_docs)))
    return sorted_run, candidate_docs

def normalized(strings, strings_title="No Title"):
    if strings_title != "No Title":
        strings = strings_title + " " + strings 
    strings = re.sub(r"\n", " ", strings)
    # strings = re.sub(r"\s{2, }", " ", strings)
    return strings.strip()

# Load requirements (corpus, queries, runs)
runs, candidate_docs = load_run(path=args.run, topk=args.top_k, trec=args.trec)
queries = load_queries(path=args.queries)

# Load only the collection that within the run file and chunk.
passageChunker = SpacyPassageChunker()
corpus, titles = load_corpus(path=args.corpus, doc_level=args.doc_level, candidates=candidate_docs)
n_passage = 0

with open(args.output_text_pair, 'w') as text_pair, open(args.output_id_pair, 'w') as id_pair:
    for i, (qid, docids) in enumerate(runs.items()):
        # Only create for tok_k candidates

        for k, docid in enumerate(docids):
            if args.doc_level:

                for passage in corpus[docid]:
                    text_example = "Query: {} Document: {} Relevant:\n".format(
                            queries[qid], normalized(passage["body"]))
                    id_example = "{}\t{}-{}\t{}\n".format(qid, docid, passage["id"], (k+1))
                    text_pair.write(text_example)
                    id_pair.write(id_example)
                    n_passage += 1
            else:
                text_example = "Query: {} Document: {} Relevant:\n".format(
                        queries[qid], normalized(titles[docid], corpus[docid]))
                id_example = "{}\t{}\t{}\n".format(qid, docid, (k+1))
                text_pair.write(text_example)
                id_pair.write(id_example)
                n_passage += 1
                
        
        if i % 100 == 0:
            print('Loading queries ...{}'.format(i))
            print('Creating T5-qp-ranking-pairs...{}'.format(n_passage))

print("DONE!")
