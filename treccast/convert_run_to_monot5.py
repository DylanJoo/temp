import collections
import argparse
import json
from passage_chunker import SpacyPassageChunker

parser = argparse.ArgumentParser()
parser.add_argument("--queries", type=str, required=True,
                    help="tsv file with two columns, <query_id> and <query_text>")
parser.add_argument("--run", type=str, required=True,
                    help="tsv file with three columns <query_id>, <doc_id> and <rank>")
parser.add_argument("--corpus", type=str, required=True, 
                    help="json/tsv file with <doc_id> and <doc_text> or <passage_id> and <passage_text>")
parser.add_argument("-d", "--doc_level", action="store_false", default=False,  
                    help="Document level identifier, segmented the docuemnt if needed.")
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
        
            if i % 1000 == 0:
                print('Loading queries...{}'.format(i))

    return queries_dict

def load_corpus(path, doc_level):

    corpus_type = path.rsplit(".", 1)[-1]
    collection_dict = {}
    title_dict = {}

    if corpus_type == "json":
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                doc_dict = json.loads(line)
                docid, doctext, doctitle = doc_dict["id"], doc_dict["contents"], doc_dict["title"]
                collection_dict[docid] = doctext
                title_dict[docid] = doctitle

                if i % 10000 == 0:
                    print('Loading collections...{}'.format(i))


    if corpus_type == "tsv":
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                docid, doctext = line.rstrip().split('\t')
                collection_dict[docid] = doctext
            
                if i % 10000 == 0:
                    print('Loading collections...{}'.format(i))

    return collection_dict, title_dict

def load_run(path):
    
    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
            qid, docid, rank = line.split('\t')
            if qid not in run:
                run[qid] = []
            run[qid].append((docid, int(rank)))

    print('Sorting candidate docs by rank...')
    sorted_run = collections.OrderedDict()
    for i, qid, doc_ids_ranks in enumerate(run.items()):
        sorted(doc_ids_ranks, key=lambda x: x[1])
        docids = [docid for docid, _ in doc_ids_ranks]
        sorted_run[qid] = docids

    print('Loading run...{}'.format(i))
    return sorted_run

def normalized(strings):
    strings = re.sub(r"\n", " ", strings)
    strings = re.sub(r"\s{2, }", " ", strings)
    return string.strip()

# Load requirements (corpus, queries, runs)
collections, titles = load_corpus(path=args.corpus)
queries = load_queries(path=args.queries)
runs = load_run(path=args.run)
passageChunker = SpacyPassageChunker()

with open(args.output_text_pair, 'w') as text_pair, open(args.output_id_pair, 'w') as id_pair:
    for rank, (qid, docids) in enumerate(runs.items()):
        for docid in docids:
            passageChunker.sentence_tokenization(collections[docid])
            passages = passageChunker.create_passages()

            for n_paragraph, passage in passages:
                text_example = "Query: {} Document: {} Relevant: ".format(
                        queries[qid], normalized(titles[docid], passage["body"]))
                id_example = "{}\t{}-{}\t{}\n".format(qid, docid, passage["id"], rank+0.001*n_paragraph)
                text_pair.write(text_example)
                id_pair.write(id_example)
        
        if i % 10000 == 0:
                print('Creating T5-qp-ranking-pairs...{}'.format(i))

print("DONE!")
