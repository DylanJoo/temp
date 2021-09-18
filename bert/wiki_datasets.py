from dataset import load_datasets

def prepare_features(examples, data_args=None):

    size = len(examples['sentA'])
    assert len(examples['sentA']) == len(examples['sentB']) and size == len(examples['sentA']),\
     'Invalud number of examples'

    # Debugging
    for idx in range(size):
        if examples['sentA'][idx] is None:
            examples['sentA'][idx] = " "
        if examples['sentB'][idx] is None:
            examples['sentB'][idx] = " "
        if examples['label'][idx] is None:
            examples['label'][idx] = " "

    sentence_A = examples['sentA']
    sentence_B = examples['sentB']

    # Tokenized the string to id 
    sentence_features = tokenizer(
        sentence_A, sentence_B,
        max_length=data_args.max_seq_length,
        truncation=True,
        padding="max_length"
    )

    # add the labels in features
    sentence_features['labels'] = [None] * size
    # add the labels in features
    for idx in range(size):
        if examples['label'][idx] is True:
            sentence_features['labels'][idx] = 'True'
        else:
            sentence_features['labels'][idx] = 'False'

    return sentence_features

