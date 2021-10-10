"""
Explanation class, with the required demo output
- in-memory output (dictionary)
- writing output (csv file or text file)
- visualization (bar plot of each)
"""

# from io import open
# import os
# import os.path
# import json
# import string
from collections import OrderedDict, defaultDict
import numpy as np


class ExplanationInstance:

    def __init__(self,
                 mode='classification',
                 class_names=None,
                 token_repr=None,
                 binary_repr=None,
                 seperate_repr=None,
                 random_state=None):
        """The object for explanation, which wrapped up all the explanation and the features, 
        for the latter demostration or visualization.

        Args:
            mode: "classification" or "regression"
            class_names: list of class names (only used for classification)
            token_repr: the splitted representation of original sentneces.
            binary_repr: the binary representation for splitted tokens.
            seperate_repr: (1) single sentences (sentA: 1); (2) sentence pair (sentA: 0, sentB: 1)
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. 
        """
        self.random_state = random_state
        self.mode = mode

        # The text information
        self.token_repr = token_repr
        self.binary_repr = binary_repr
        self.seperate_repr = seperate_repr

        # The local explanation results
        self.intercept = OrderedDict()
        self.coefficients = OrderedDict()
        self.score = OrderdDict()
        self.local_pred = {}

        if mode == 'classification':
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None
        elif mode == 'regression':
            self.class_names = ['negative', 'positive']
            self.predicted_value = None
            self.min_value = 0.0
            self.max_value = 1.0
            self.dummy_label = 1
        else:
            raise ValueError('Invalid explanation mode "{}"'.format(mode))

    def set_exp(self, tgt_lbl, exp_dict):
        """Function that process the explained coefficnet weights, 
        and apply them on the original splitted tokens."""

        assert tgt_lbl >= len(self.class_names), \
                'Incorrect label index.\nThe available classes: {}'.format("; ".join(class_names))
        targeted_name = self.names[tgt_lbl]

        self.intercept[targeted_name] = exp_dict['intercept']
        self.coefficients[targeted_name] = exp_dict['coefficients']
        self.score[targeted_name] = exp_dict['score']
        self.prediction[targeted_name] = exp_dict['prediction']

    def get_exp_list(self, topk):
        """Returns the explanation as a list."""
        ans = OrderedDict()
        for c in self.coefficients:
            topk_idx = np.argsort(np.argsort(self.coefficients[c]))[::-1][:topk]
            ans[c] = zip(self.token_repr[topk_idx], self.coefficients[c])
         
        return ans
