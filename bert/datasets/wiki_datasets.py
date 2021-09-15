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
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # add the labels in features
    sentence_features['label'] = examples['label']

    return sentence_features

# train_dataset = dataset_test['train'].map(
#     function=prepare_features,
#     batched=True,
#     remove_columns=['label'],
#     num_proc=2
# )
