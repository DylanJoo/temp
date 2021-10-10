"""
Functions for explaining text classifiers.
LimeBase: Lime's explainale model g.
LimeTextExplainer: Lime's locally perturbed explain algorithm.
"""
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import scipy as sp

from functools import partial
from typing import Optional
import itertools
import json
import re
import numpy as np 

from utils import batch_iteri, flatten_listOflist
from models import SimpleLinearRegression
from text_instance import TextInstance
from explanation_instance import ExplanationInstance

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):

        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

        # To be updated
        self.model_g = None

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def _get_new_feature(self, 
                         data, labels, weights, 
                         estimator, 
                         direction, 
                         mask):
        """Function for sequentially add/remove the features, which recored on the mask.
        The esimator is a model_fs, using it to search for the (locally) best feature greedily.
        1 indicates used, vice versa.

        Returns:
            mask (boolean array): The underlying selected feature. (to be add/remove)
        """
        scores = {}

        for feature_index in np.flatnonzero(~mask):
            new_mask = mask.copy() # choose a new one as the new mask
            new_mask[feature_index] = True

            if direction == 'backward':
                new_mask = ~new_mask 

            estimator.fit(data[:, new_mask], labels, sample_weight=weights)

            # maybe some kind of validation is better, though CV is weired.
            scores[feature_index] = model_fs.score(data[:, new_mask],
                                                   labels, 
                                                   sample_weight=weights)

        # Find out the "targeted" one.
        return max(scores, key=lambda idx: scores[idx])

    def feature_selection(self, 
                          data, labels, weights, 
                          num_features=None, 
                          method='auto', 
                          init_mask=None):
        """Funcition for the feature (tokens) selection, which adopted the regularization.

        Args:
            data/labels: the training samples.
            weights: the sampling distribution.
            num_features: pre-defined the maximum number of token to be selected. 
            method: feature selection process.
                - forward/backward: Iteratively adds features to the model.
                - highest_weights: Selected highest K tokens as the model. (left the other zero)
            init_mask: the initial mask for the data (i.e. feature selected), default is all.

        Return:
            selected_features: the index of the features (from columns of data)
            highest_weights: One-shot estimate, and selected by the coefficients values.
        """
        if method == 'auto':
            if num_features / sum(init_mask) > 0.75:
                method = 'backward'
            elif num_features / sum(init_mask) < 0.25:
                method = 'forward'
            else:
                method = 'highest_weights'

        if method == 'highest_weights':
            current_mask = np.ones(data.shape[1], dtype=bool) if init_mask is None else init_mask
            model_fs = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
            model_fs.fit(data[:, current_mask], labels, sample_weight=weights)

            feature_weights = sorted(zip(np.flatnonzero(current_mask), np.abs(model_fs.coef_)),
                                     key=lambda x: x[1], reverse=True)

            return np.array([x[0] for x in feature_weights[:num_features]])

        else:  # forward or backward, start from nothing
            current_mask = np.zeros(data.shape[1], dtype=bool) if init_mask is None else ~init_mask
            model_fs = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)

            n_iterations = num_features if method == 'forward'
            n_iterations = data.shape[1] - num_features if method == 'backward' 

            for _iter in n_iterations:
                selected = self._get_new_feature(
                        data, labels, weights,
                        estimator=model_fs,
                        direction=method, 
                        mask=current_mask
                )
                current_mask[selected] = True

            # In backward: 1 indicates the "to-be-removed", thus reversed.
            used_features = ~current_mask if method == 'backward' else current_mask

            return np.flatnonzero(used_features)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection_method='auto',
                                   model_g_option=None):
        """
        Args:
            neighborhood_data: (N perturbed samples + 1 original) data.
            neighborhood_labels: The predicted probabilities, i.e. label_prob
            distances: The physical distance b/w origianl and perturbeds'(for weighted sampling)
            model_g_option: The underlying model for explaination (e.g. SimpleLinearRegression)

        Procedures:
            - (pre-fit) feature selection: Using the data and labels
                to findout the "potential" features, by directly perform the regression.
            - (Post fit) selected feature importance calculate: Using the remaining coefficents,
                to fit agiain for the coefficients with more precise results.

        Returns: 
            (dictionary of model g estimation)
            - intercept (float) 
            - coefficients (array of coef)
            - score: RSS of all data (float)
            - pred: probabilities (g(x))
        """
        weights = self.kernel_fn(distances) 
        label_probabilities = neighborhood_labels[:, label]

        # (pre-fit) feature selection
        coefficients = np.zeros(
        used_features_idx = self.feature_selection(neighborhood_data,
                                                   neighborhood_labels,
                                                   weights,
                                                   k=num_features,
                                                   method=feature_selection_method)
        # (post-fit) predicting probabilties
        if model_g_option is None:
            model_g = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        else:
            model_g = model_regressor

        model_g.fit(neighborhood_data[:, used_features_idx], 
                    label_probabilities,
                    sample_weight=weights)

        # model g estimates
        coefficients = np.zeros(neighborhood_data.shape[1])
        coefficients[used_features_idx] = model_g.coef_

        # weighted score (of all examples)
        local_fitness = model_g.score(neighborhood_data[:, used_features_idx],
                                     labels_column, 
                                     sample_weight=weights)

        # prediction prob (of original)
        local_pred = model_g.predict(neighborhood_data[0, used_features_idx].reshape(1, -1))

        if self.verbose:
            print('Intercept', self.model_g.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])

        # process the feature weights, assign all the other zero

        return  {'intercept': model_g.intercept_,
                'coefficients': cofficients,
                'score': local_fitness, 
                'pred': local_pred}

class LimeTextExplainer(object):

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=1234,
                 char_level=False):

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = random_state
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.char_level = char_level

        # to be build
        self.label_probs = np.empty((0, len(class_names))
        self.text_instance = None

    def set_classifier(self, trainer: Optional[HuggingfaceTrainer] = None):
        """Set the model f, which includes the trainer's model and tokenizer"""
        self.model = trainer.model 
        self.tokenizer = trainer.tokenizer

    def explain_instance(self,
                         raw_string,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         max_num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):
        """Function call for the raw string explaination pipeline.

        Procedure:
            step 1a: Generate neighborhood data by perturbation.
                - Make text instance, and perturbing.
                - [TODO] more than one raw_string
            step 1b: Generate the inferecing dataset (for BERT)
                - [TODO] Maybe the model finetuning on other function as well
            step 2-: Make sure that the model f is ready (or use the existing one)
            step 2: Predict the pseduo label by huggingface 
            step 3-: Create the explaination objects.
            step 3: Estimate the model g locally.
                - [TODO] How about "globally"
            step 4: Explain the results by coef. 
                - Ouput the explaination object.

        Args:
            raw_string: the original text.
            labels: the original labels (which is used to infer the proba)
            max_num_features: specify the maximun number of features.
            num_samples: specify the number of perturbed samples.
        """
        # Step 1a
        self.text_instance = TextInstance(raw_string=raw_string)
        self.text_instance.perturbed_data_generation(
                num_samples=num_samples,
                perturbed_method='mask'
        )
        perturbed = self.perturbed
        distances = self.perturbed_distances
        data = self.perturbed_data

        # Step 1b
        perturbed_dataset = PerturbedDataset(text_instance=self.text_instance, 
                                             tokenizer=self.tokenizer,
                                             bow=False).get_dataset()

        # Step 2-, [TODO] If None, maybe it will need to train or load any availablde model. 
        assert self.model is None, 'The classifier/model f is not ready.'

        # Step 2: Note that it's the 2-way probabilties
        for batch in batch_iter(perturbed_dataset):
            output = self.model.inference(**batch)
            self.label_probs = np.vstack((self.label_probs, output['probabilities'].numpy())

        # step 3-
        explanation_instance = ExplanationInstance(
            class_names=self.class_names,
            token_repr=flatten_listOflist(self.text_instance.split), 
            binary_repr=flatten_listOflist(self.text_instance.isfeature),
            seperate_repr=self.text_instance.sent_sep
            random_state=self.random_state
        )

        # step 3
        for label in labels:
            explanation_instance =  self.base.explain_instance_with_data(
                neighborhood_data=data, 
                neighborhood_labels=self.label_probs, 
                distances=distances,
                label=label, 
                num_features=max_num_features,
                feature_selection=self.feature_selection,
                model_g_option=None
            )

            explanation_instance.set_exp(
                targeted_class=label,
                targeted_dict=explanation_instance
            )

        # step 4
        return explanation_instance.get_exp_list(topk=max_num_features)

        return ret_exp
