"""
"""
import sys
import torch
import random
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from utils import WikipediaDataSet

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    Trainer,
    HfArgumentParser,
    BertForSequenceClassification,
)

from transformers.tokenization_utils_base import (
        BatchEncoding, 
        PaddingStrategy, 
        PreTrainedTokenizerBase
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
    train_folder: Optional[str] = field(default='wiki_727/train')
    eval_folder: Optional[str] = field(default='wiki_727/dev')
    test_folder: Optional[str] = field(default='wiki_727/text')
    max_seq_length: Optional[int] = field(default=512)
    pad_to_max_length: bool = field(default=False)

@dataclass
class OurTrainingArguments(TrainingArguments):
    output_dir: str = field(default='./models')
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=-1)
    num_train_epochs: float = field(default=1.0)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=5000)
    evaluate_during_training: bool = field(default=False)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_steps: int = field(default=1000)
    remove_unused_columns: Optional[bool] = field(default=True)
    resume_from_checkpiint: Optional[str] = field(default=None)
    instance_per_example: int = field(default=2)
    negative_sampling: Optional[str] = field(default="hard_max")


def main():
    """
    (1) Prepare parser with the 3 types of arguments
        * Detailed argument parser by kwargs
    (2) Load the corresponding tokenizer and config 
    (3) Load the self-defined models
    (4)
    """

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


    # model 
    model_kwargs = {
            "cache_dir": model_args.cache_dir,
    }
    model = BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config, 
    )

    # Dataset 
    # (1) Wikipedia dataset
    train_dataset = WikipediaDataSet(data_args.train_folder)
    eval_dataset = WikipediaDataSet(data_args.eval_folder)

    @dataclass
    class DataCollatorWithMultipleInput:
        """ Padding with defined maximum length in left/right context

        Note the special tokens are padded in this way:
        wordpiece tokens: [CLS] <sentA> [PAD] ... [PAD] [SEP] <sentB> [PAD] .. [PAD]
        token_types:      0 0 0 ... .... .... ... ...  | 1 1 1 ... ... 0 0 0 0 ... 0
        and each sequence pad with same length
        """
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        truncation: Union[bool, str] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"
        n_positve_per_example: Optional[int] = None
        n_negative_per_example: Optional[int] = None
        padding: Union[bool, str] = True
        negative_sampling: str = 'random'

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

            flat_left_features, flat_right_features, flat_labels = [], [], []

            for i in range(len(features)):
                doc_features = features[i]       
                pos_ind = [ind for ind in doc_features['targets'] if ind == 1]
                pos_ind = random.sample(
                        pos_ind, min(self.n_positive_per_example, len(pos_ind))
                )
                neg_ind = [ind for ind in doc_features['targets'] if ind == 0]

                if "hard" in self.negative_sampling:
                    # haed negative
                    final_neg_ind = []
                    for a in pos_ind:
                        distance = [abs(a-b) for b in neg_ind]
                        if self.negative_sampling == 'hard_max':
                            f = (lambda x: max(x))
                        elif self.negative_sampling == 'hard_min':
                            f = (lambda x: min(x))
                        final_neg_ind.append(neg_ind[distance.index(f(distance))])
                else:
                    # random
                    final_neg_ind = random.sample(
                            neg_ind, min(self.n_negative_per_example, len(neg_ind))
                    )

                # [Warning] Not sure there are > 2 positive label in each document.
                for j in pos_j + final_neg_ind:
                    flat_left_features += [doc_features['left_context'][j]]
                    flat_right_features += [doc_features['right_context'][j]]
                    flat_labels += [doc_features['targets'][j]]

            # Direct using huggingface truncation
            batch = self.tokenizer(
                flat_left_features, flat_right_features,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch['labels'] = torch.tensor(flat_labels)

            return batch

    # (2) data collator: transform the datset into the training mini-batch
    data_collator = DataCollatorWithMultipleInput(
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
            return_tensors="pt",
            padding=True,
            n_positve_per_example=training_args.instance_per_example,
            n_negative_per_example=training_args.instance_per_example,
            negative_sampling=training_args.negative_sampling
    )

    # Trainer
    trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            data_collator=data_collator
    )
    # trainer.model_args = model_args
    
    # ***** strat training *****
    model_path = None #[TODO] parsing the argument model_args.model_name_or_path 
    results = trainer.train(model_path=model_path)

    return results

if __name__ == '__main__':
    main()