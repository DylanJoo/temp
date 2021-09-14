from dataclasses import dataclass, field

from xxx.models import BertForSegmentPrediction
from xxx.trainer import xxxTrainer 

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    TraininngArguments,
    HfArgumentParser
)

@dataclass
class OurModelArguments:

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` 
            (necessary to use this script " "with private models)."
        },
    )
    # Cutomized arguments
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "Different kind of pooler used (default in BertModel)"
        }
    )
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax"
        }
    )

    # unused arguments
    # do_mlm / mlm_weight / mlp_only_train ...

@dataclass
class OurDataTrainingArguments:

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # customized arguments
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training dataset file. So far, only work for csv with tab"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum sequence length after tokenization"
        },
    )
    # unused arguments
    # mlm_probability / padding to max length
    def __post_init__(self):
        """
        use this post init function to process the methods after initialization, 
        e.g. dataset checking
        """
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need to specify the dataset")
        if 

def main():
    
    # [Arguments] Load parser from 3 customized argument dataclasses
    parser = HfArgumentParser((
        OurModelArguments,
        OurDataTrainingArguments, 
        OurTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # [Models] Load models and related configs
    # and load from pretrained weights
    configuration = BertConfig() 
    tokenizer = BertTokenizer()
    # model = BertModel(configuration)
    model = BertForSegmentPrediction(configuration)
    model.from_pretrained("the_name_of_the_model")

    # [Data] Load datasets 
    train_dataset = None
    eval_dataset = None
    data_collator = default_data_collator 
    # collator can be customized for token-level pretraining task
    
    # [Trainer] Wrapup the args/models/data and preprocess them into a hug-face trainer
    trainer = OurTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator
    )


