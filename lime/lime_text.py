"""
Functions for explaining text classifiers.
LimeBase: Lime's explainale model g.
LimeTextExplainer: Lime's locally perturbed explain algorithm.
"""
from sklearn.linear_model import Ridge
import scipy as sp

from functools import partial
import itertools
import json
import re
import numpy as np 

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):

        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):

    def feature_selection(self, data, labels, weights, num_features, method):
        if method == 'forward_selection':
            """Iteratively adds features to the model"""
            clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
            used_features = []
            for _ in range(min(num_features, data.shape[1])):
                max_ = -100000000
                best = 0
                for feature in range(data.shape[1]):
                    if feature in used_features:
                        continue
                    clf.fit(data[:, used_features + [feature]], labels,
                            sample_weight=weights)
                    score = clf.score(data[:, used_features + [feature]],
                                      labels,
                                      sample_weight=weights)
                    if score > max_:
                        best = feature
                        max_ = score
                used_features.append(best)
            return np.array(used_features)
        elif method == 'highest_weights':
            """One-shot esimate, and locked the coefficient value"""
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            weighted_data = coef * data[0]
            feature_weights = sorted(
                zip(range(data.shape[1]), weighted_data),
                key=lambda x: np.abs(x[1]),
                reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """
        distance: The physical distance b/w origianl and perturbeds'.
        model_regressor: The pre-defined model coefficients (In many case, Ridge)
        """
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        # Here is similar to post-lasso (first select then inference)
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        # RSS
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)

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
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

        # to be build
        self.text_instance = None

    def explain_instance(self,
                         raw_string,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):
        """Function call for the raw string explaination pipeline.

        Modification:
            step 1a: Generate neighborhood data by perturbation.
                - Make text instance, and perturbing.
            step 1b: Generate the inferecing dataset (for BERT)
                - [TODO] Maybe the model finetuning on other function as well
            step 2: Predict the pseduo label by huggingface 
            step 3: Estimate the model g locally.
                - [TODO] How about "globally"
            step 4: Explain the results by coef. 
                - Ouput the explaination object.
        """
        # Step 1 [TODO]: maybe more than one raw_string.
        self.text_instance = TextInstance(raw_string=raw_string)
        self.text_instance.perturbed_data_generation(
                num_samples=num_samples,
                perturbed_method='mask'
        )
        perturbed, data, distances = self.perturbed, self.perturbed_data, self.perturbed_distances

        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]

        # Step 2 
        tokenizer = PretrainedTokenizer.from_pretrained()
        dataset = PerturbedDataset(text_instance, tokenizer=tokenizer)

        # step 3: huggingface model f inferencing

        # step 4
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]

        # Explanation
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
             # [TODO] Prepare for the local prediction
             # Make the model easier and more readable.
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    # def __data_labels_distances(self,
    #                             text_instance,
    #                             classifier_fn,
    #                             num_samples,
    #                             distance_metric='cosine'):
    #
    #     def distance_fn(x):
    #         return sklearn.metrics.pairwise.pairwise_distances(
    #             x, x[0], metric=distance_metric).ravel() * 100
    #
    #     feature_size = text_instance.num_words()
        # feature size: number of tokens --> model regressors
    #     sample = self.random_state.randint(1, doc_size + 1, num_samples - 1)
        # perturbed 1: Generate "How many token should be masked?" , 
        # i.e. masked n token (in one isntance)
    #     data = np.ones((num_samples, doc_size))
    #     data[0] = np.ones(doc_size) # The first instance is the true (non-perturbed instance)
    #     features_range = range(doc_size)
    #     inverse_data = [indexed_string.raw_string()]
    #
    #     for i, size in enumerate(sample, start=1):
    #         inactive = self.random_state.choice(features_range, size,
    #                                             replace=False)
    #         # perturbed 2: Noted specifically which n tokens should be masked?
    #         # i.e. masked n tokens, and randomly picked in each feature (non-repeated)
    #         data[i, inactive] = 0
    #         # For faster perturbed instance generation, only remove the inactive from origin.
    #         # Prepare the contextualized embeddings
    #         inverse_data.append(indexed_string.inverse_removing(inactive))
    #     labels = classifier_fn(inverse_data)
    #     distances = distance_fn(sp.sparse.csr_matrix(data))
    #     return data, labels, distances
