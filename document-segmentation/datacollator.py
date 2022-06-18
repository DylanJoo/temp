from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
        PaddingStrategy, 
        PreTrainedTokenizerBase
)

@dataclass
class Fin10KDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        left_features = [ft['left_context'] for ft in features]
        right_features = [ft['right_context'] for ft in features]

        batch = self.tokenizer(
            left_features, right_features,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch['sent_id'] = [ft['sent_id'] for ft in features]

        return batch


@dataclass
class WikiDataCollator:
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
    n_positive_per_example: Optional[int] = None
    n_negative_per_example: Optional[int] = None
    padding: Union[bool, str] = True
    negative_sampling: str = 'random'

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        flat_left_features, flat_right_features, flat_labels = [], [], []
        offset = 0

        for i in range(len(features)):
            doc_features = features[i]       
            pos_ind = [ind for ind, lbl in enumerate(doc_features['targets']) if lbl == 1]
            pos_ind = random.sample(
                    pos_ind, min(self.n_positive_per_example, len(pos_ind))
            )
            neg_ind = [ind for ind, lbl in enumerate(doc_features['targets']) if lbl == 0]

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

            # offset compensation (from last example)
            if offset > 0:
                remain = [i for i in range(doc_features) if i not in pos_ind+final_neg_ind]
                remain = random.sample(remain, min(offset, len(remain)))
                offset -= len(remain)
            else:
                remain = []

            # [Warning] Not sure there are > 2 positive label in each document.
            for j in pos_ind + final_neg_ind + remain:
                flat_left_features += [doc_features['left_context'][j]]
                flat_right_features += [doc_features['right_context'][j]]
                flat_labels += [doc_features['targets'][j]]

            offset += self.n_negative_per_example - len(final_neg_ind)

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

