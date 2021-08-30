import tensorflow as tf
import functools

# msmrco passage ranking
def msmarco_passage_ranking_prep(ds):
    '''The preprocessor of t5 source and target.
    '''
    def normalize_text(text):
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text
    def to_inputs_and_targets(ex):
        return {
            "inputs": normalize_text(ex["qd-pair"]),
            "targets": normalize_text(ex["relevance"])
        }
    return ds.map(to_inputs_and_targets, 
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

def msmarco_passage_ranking_ds(split, shuffle_files):
    '''The tfText dataset pipeline.
    Input: [Query: <q> Document: <d> Relevant:]
    Output: [true] or [false]
    '''
    if split == "full":
        dataset = tf.data.TextLineDataset("gs://castorini/monot5/data/query_doc_pairs.train.tsv")
        # [TODO] upload the full triplet if needed.
    else:
        dataset = tf.data.TextLineDataset("gs://castorini/monot5/data/query_doc_pairs.train.tsv")
    dataset = dataset.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda *ex: dict(zip(["qd-pair", "relevance"], ex)))
    
    return dataset

# msmrco passage to query
def msmarco_passage_to_query_prep(ds):
    '''The preprocessor of t5 source and target.
    '''
    def normalize_text(text):
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text
    def to_inputs_and_targets(ex):
        return {
            "inputs": normalize_text("Document: " + ex["positive_passage"] + " Translate Document to Query:"),
            "targets": normalize_text(ex["query"])
        }
    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

def msmarco_passage_to_query_ds(split, shuffle_files):
    '''The tfText dataset pipeline.
    Input: [Document: <d> Translate Document to Query:]
    Output: [<q>]
    '''
    dataset = tf.data.TextLineDataset("gs://conv-ir/msmarco/doubles.train.qrels.tsv")
    dataset = dataset.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda *ex: dict(zip(["positive_passage", "query"], ex)))
    
    return dataset

