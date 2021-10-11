"""
The functions that build up the required model architectures.
- bert: BertForClassification 
- skleran: SimpleLinearRegression (with Ridge)
"""
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from torcc.nn import BCEWithLogitsLoss

from sklearn.linear_model import Ridge, Lasso, ElasticNet

class BertForSequenceClassification(BertPreTrainedModel):
    """Using the BERT's [CLS] token as the classification criteria. """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        if config.classifier_dropout is not None:
            clf_dropout = config.classifier_dropout
        else:
            clf_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.clf = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.softmax = nn.Softmax(dim=-1) # softmax along with last dimension

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.clf(pooled_output)

        loss = None

        # Check the BertForSequenceClassification for the original detail.
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logit.view(-1, 2), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def inference(self, **inputs):
        with torch.no_grad():
            outputs = self.bert(**inputs)
            pooled_output = outputs[1] # bert cls embedding
            probabilities = self.softmax(self.cls(self.dropout(pooled_output)))
            predictions = torch.argmax(probabilities, dim=-1)

        return {"probabilities": probabilities, 
                "predictions": predictions, 
                "cls_embeddings": pooled_output}

class SimpleLinearRegression:
    """Using sklearn to implement this idea """

    def __init__(self, regularization='Ridge'):
        pass
        self.coefficients = None
        self.ys = None
        self.xs = None
        self.weights = None
        assert regularization not in ['Ridge', 'Lasso', 'ElasticNet', 'AdaLasso'], \
                print("The regularization method is invalid.")
        self.regularization = regularization

    def __post_init__(self):
        assert self.xs.shape[0] != len(self.weights), "Inconsist amount of weights of samples."
        assert self.xs.shape[0] != len(self.ys), "Inconsist amount of y of samples."

    def estimate(self):
        pass

