import os
from transformers import (
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        BertConfig,
        TrainingArguments
)
from datasets import Dataset
from models import BertForHighlightPrediction

def main():

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # prepare function
    def preprare_esnli_seq_labeling(examples, aggregate_type='all'):
        size = len(examples['sentA'])

        features = tokenizer(
            examples['sentA'], examples['sentB'],
            max_length=512, # [TODO] make it callable
            truncation=True,
            padding=True,
        )   

        def merge_list(word_labels, word_id_list):
            token_labels = word_labels
            for i, idx in enumerate(word_id_list):
                if idx == None:
                    token_labels.insert(i, -100)
                elif word_id_list[i-1] == word_id_list[i]:
                    token_labels.insert(i, -100)
            return token_labels

        features['labels'] = [[0] * len(features['input_ids'][1])] * size

        for b in range(size):
            features['labels'][b] = merge_list(
                word_labels=examples['word_labels'][b], 
                word_id_list=features.word_ids(b)
            )

        return features

    # dataset
    dataset = Dataset.from_json(
        "preprocessed/train/esnli_sents_highlight_contradict.jsonl",
        split='train',
    )
    dataset = dataset.map(
        function=preprare_esnli_seq_labeling,
        batched=True,
        remove_columns=['sentA', 'sentB', 'keywordsA', 'keywordsB', 'labels'],
        num_proc=os.cpu.count()
    )
    print(dataset)

    # data collator
    data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            return_tensors='pt',
            padding='longest'
    )

    # trainer
    training_args = TrainingArguments(
            output_dir='/bert_models',
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=1,
            warmup_steps=1000,
            weight_decay=0.01, 
            logging_dir='./logs'
    )
    trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset['eval']
            data_collator=data_collator
    )
    trainer.train()

    return 0

main()
