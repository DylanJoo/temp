import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from utils import WikipediaDataset, Fin10KDataset
from datacollator import WikiDataCollator, Fin10KDataCollator
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    Trainer,
    HfArgumentParser,
    BertForSequenceClassification,
)

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='bert-base-uncased')
    model_type: Optional[str] = field(default='bert-base-uncased')
    config_name: Optional[str] = field(default='bert-base-uncased')
    tokenizer_name: Optional[str] = field(default='bert-base-uncased')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Cutomized arguments
    num_labels: float = field(default=2)

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    run_wiki: Optional[bool] = field(default=False)
    test_folder: Optional[str] = field(default='wiki_727/test')
    run_fin10k: Optional[bool] = field(default=True)
    test_file: Optional[str] = field(default='samples.txt')
    max_seq_length: Optional[int] = field(default=512)
    pad_to_max_length: bool = field(default=False)
    pred_dir: str = field(default='./prediction')

@dataclass
class OurTrainingArguments(TrainingArguments):
    output_dir: str = field(default='./models')
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=-1)
    num_train_epochs: float = field(default=1.0)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=5000)
    evaluate_during_training: bool = field(default=True)
    evaluation_strategy: Optional[str] = field(default='steps')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_steps: int = field(default=0)
    remove_unused_columns: Optional[bool] = field(default=False)
    resume_from_checkpoint: Optional[bool] = field(default=None)
    instance_per_example: int = field(default=2)
    negative_sampling: Optional[str] = field(default="hard_min")


def main():
    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_datalcasses()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # config and tokenizers
    # [TODO] If the additional arguments are fixed for putting in the function,
    # make it consistent to the function calls.
    config_kwargs = {
            "num_labels": model_args.num_labels,
            "output_hidden_states": True
    }
    tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir, 
            "use_fast": model_args.use_fast_tokenizer
    }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    model = BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config, 
    )

    # Dataset 
    if not os.path.exists(data_args.pred_dir):
        os.makedirs(data_args.pred_dir)

    # (1) Wikipedia dataset
    if data_args.run_wiki:
        test_dataset = WikipediaDataset(data_args.test_folder)
        test_dataset.filtering()

        data_collator = WikiDataCollator(
                tokenizer=tokenizer,
                max_length=data_args.max_seq_length,
                return_tensors="pt",
                padding=True,
                n_positive_per_example=training_args.instance_per_example,
                n_negative_per_example=training_args.instance_per_example,
                negative_sampling=training_args.negative_sampling
        )
        output_file = data_args.test_folder.split('.')[0] + '-' + model_args.model_name_or_path.split('-')[1]
        fout = open(os.path.join(data_args.pred_dir, output_file), 'w')

    # (2) Fin10K dataset
    if data_args.run_fin10k:
        test_dataset = Fin10KDataset(data_args.test_file)
        data_collator = Fin10KDataCollator(
                tokenizer=tokenizer,
                max_length=data_args.max_seq_length,
                return_tensors="pt",
                padding=True,
        )
        output_file = data_args.test_file.split('.')[0] + '-' + model_args.model_name_or_path.split('-')[1]
        fout = open(os.path.join(data_args.pred_dir, output_file), 'w')

    dataloader = DataLoader(
            test_dataset, 
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator
    )


    model.to(training_args.device)
    softmax = nn.Softmax(dim=1)

    # run prediction
    for b, batch in enumerate(dataloader):
        ids = batch.pop('sent_id')

        for k in batch:
            batch[k] = batch[k].to(training_args.device)

        output = model(**batch)
        probs = softmax(output.logits)
        probs = probs.detach().cpu().numpy()

        if b % 1000 == 0:
            print(f'{b} batch predicted' + f"'<LAST SENTENCE>' | {ids[0]}: {probs[0]}")
        fout.write("\n".join(
            f'{id}\t{pred}\t{prob}' for id, pred, prob in zip(ids, np.argmax(probs, -1), probs[:, 1])
        )+'\n')


if __name__ == '__main__':
    main()
