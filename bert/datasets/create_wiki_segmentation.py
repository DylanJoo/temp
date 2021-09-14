from dataset import load_datasets

def convert_text_to_segment_pair(file):

    def normalized(strings):
        strings = strings.strip()
        strings = re.sub('"', '', strings)
        strings = re.sub(r"\t", "", strings)
        strings = re.sub(r"\n", " ", strings)
        strings = re.sub(r"\s+", " ", strings)
        return strings

    with open(file, 'r') as source, open(file.replace(".txt", ".csv"), 'w') as pair_data:
        sentences = list()
        t, f = 0, 0
        for i, line in enumerate(source):

            if "========," in line:
                if len(sentences) == 2:
                    pair_data.write("{}\t{}\t{}\n".format(
                        normalized(sentences[0]), 
                        normalized(sentences[1]), "True"))
                    sentences = list()
                    t += 1

            else:
                if len(sentences) == 2:
                    pair_data.write("{}\t{}\t{}\n".format(
                        normalized(sentences[0]), 
                        normalized(sentences[1]), "False"))
                    sentences.pop(0) # pop the old sentence
                    f += 1
                
                sentences.append(line) # append the new one

            if i == 10000:
                print("{} line finished. Class dist.: {}, {}".format(i, t, f))

def prepare_features(examples):

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
