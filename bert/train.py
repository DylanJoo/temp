from dataclasses import dataclass, field

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    TrainingArguments,
    HfArgumentParser
)
from xxx.models import xxxSegmentation
from xxx.trainer import OurTrainer

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
    pass

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
    pass
    
    def __post_init__(self):
        """
        use this post init function to process the methods after initialization, e.g. dataset checking
        """
        pass

def main():
    
    # [Arguments] Load parser from 3 customized argument dataclasses
    parser = HfArgumentParser((OurModelArguments, OurDataTrainingArguments, OurTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # [Models] Load models and related configs, and load from pretrained weights
    tokenizer = BertTokenizer.from_pretrained()
    configuration = BertConfig.from_pretrained()
    ## [Models (configuration)]
    configuration.output_hidden_states = True
    model = BertModel.from_pretraiend(config=configuration)

    # [Data] Load datasets 
    training_dataset=None
    val_dataset=None
    


    # [Trainer] Wrap up the args/models/data and preprocess them into a hug-face trainer
    trainer = OurTrainer(
            model=model, args=training_args, train_dataset=None, eval_data=None
    )



