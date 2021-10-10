"""
Function for perturbed examples, wrap-up the pipeline for pseduo-label prediction.
And make up the mini-batch huggingface procedure.
"""
from datasets import Dataset
import collections

class PerturbedDataset:
    """For the mini-batch perturbed datset.

    Args:
        data: the TextInstance to be transformed.
        tokenizer: the PretrainedTokenizer tailored for the model.

    [TODO]
        - Besides the huggingface dataset tokenizer, set the bow tokenizer
    """

    def __init__(self, text_instance, tokenizer, bow=False):
        self.data = collection.defaultdict(list)
        self.add_from_instance(text_instance)  # add data_dict
        self.tokenizer = tokenizer
        self.bow = bow # indicate that's embedding layers
    
    def __post_init__(self):
        """Check the dataset correctedness and load into huggingface dataset.
        Check if the text instance is intialized correct and perturbed samples are ready.
        """

        assert self.data.raw == None, 'Incorrect TextInstance object'
        assert self.data.distances == None, 'Invalid perturbed examples, need to generate first.'

        self.dataset = Dataset.from_dict(self.data)

    def add_from_instance(self, instance):
        """Add the data_dict (with two list of strings)
        Note that in this function, it's OK for other additional text instance.
        """

        self.data['sentA'] += instance.perturbed['sentA']
        if instance.pairwise:
            self.data['sentB'] += instance.perturbed['sentB']

    def get_dataset(self):
        """Return the preprocessed/tokenized dataset, which device is cuda"""

       def prepare_feature(x):
            if len(x) > 1:
                return self.tokenizer(x['sentA'], x['sentB'], padding=True)
            else:
                return self.tokenizer(x['sentA'], padding=True)

        preprocessed_dataset = self.dataset.map(
                function=prepare_feature,
                remove_columns=self.dataset.column_names,
                batched=True
        )
        preprocessed_data.set_format('torch', device='cuda:0')

        return preprocessed_dataset
