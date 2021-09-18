from dataclasses import dataclass, field

from xxx.models import BertForSegmentPrediction
from xxx.trainer import xxxTrainer 
from wiki_datasets import prepare_features

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    TraininngArguments,
    HfArgumentParser
)
from datasets import load_dataset

@dataclass
class OurModelArguments:

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Cutomized arguments
    pooler_type: str = field(default="cls")
    temp: float = field(default=0.05)

    # Unused arguments: do_mlm / mlm_weight / mlp_only_train ...

@dataclass
class OurDataTrainingArguments:

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)

    # Customized arguments
    train_file: Optional[str] = field(default=None)
    max_seq_length: Optional[int] = field(default=512)

    # Unused arguments: mlm_probability / padding to max length
    def __post_init__(self):
        """
        use this post init function to process the methods after initialization, 
        e.g. dataset checking
        """
        pass

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    eval_transfer: bool = field(default=False)

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._n_gpu = torch.cuda.device_count()
        else:
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device



def main():
    
    # [Arguments] Load parser from 3 customized argument dataclasses
    parser = HfArgumentParser((
        OurModelArguments,
        OurDataTrainingArguments, 
        OurTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # [Config] Load models and related configs and load from pretrained weights
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
    }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)

    # [Tokenized]
    tokenizer_kwargs = {
        "cache_dor": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    
    # [Model] Load the model we initiatite
    model = BertForSegmentPrediction.from_pretrained(
                model_args.model_name_or_path, 
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                model_args=model_args
            )

    # [Data] Load datasets 
    datasets = load_dataset(
                'csv',
                data_files={'train': data_args.train_file, 
                            'eval': data_args.eval_file,
                            'test': data_args.test_file},
                delimiter='\t',
                cache_dir='./cache/'
            )
    train_dataset = datasets['train'].map(
                function=prepare_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=['sentA', 'sentB', 'label'],
            )
    eval_dataset = datasets['eval'].map(
                function=prepare_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=['sentA', 'sentB', 'label'],
            )
    # collator can be customized for token-level pretraining task
    data_collator = default_data_collator 

    # [Trainer] Wrapup args/models/data and preprocess them into a hug-face trainer
    trainer = xxxTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator
            )
    # Train
    if training_args.do_train:
        model_path = model_args.model_name_or_path,  
        train_result = trainer.train(model_path=mdoel_path)
        trainer.save_model()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as f:
                logger.info("***** Train Results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info("{} = {}".format(key, value))
                    writer.write("{} = {}\n".format(key, value))
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("***** Evaluate *****")
        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as f:
                logger.info("***** Evaluation Results *****")
                for key, value in sorted(eval_result.items()):
                    logger.info("{} = {}".format(key, value))
                    writer.write("{} = {}\n".format(key, value))

if __name__ == "__main__":
    main()
