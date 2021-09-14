import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
# from transformers import RobertaTokenizer
# from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
# from transformers.activations import gelu
# from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from transformer import (
    BertModel, 
    BertForNextSentencePrediction
)

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)



class BertForSegmentPrediction(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]


    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)
        # maybe we can customized the output layer of this 
        self.cls = nn.Linear(config.hidden_size, 2)

    def forward(self, 
        input_ids=None, 
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embed=None,
        labels=None,
        output_attentinos=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None
        ):

        output = self.bert(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attention=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # outputs[0]: Last hidden state of the input tokens 
        # outputs[1]: [cls] embedding after pooler (hidden2hidden and tanh)
        # outputs[2]: Full hidden state of the input tokens
        pooled_output = outputs[1]

        segmentation_logit = self.cls(pooled_output)

        segmentation_loss = None
        
        if labels is not None:
            loss_fct = CrossEntropy()
            segmentation_loss = loss_fct(segmentation_logit.view(-1, 2), labels.view(-1))

       if not return_dict:
           output = (segmentation_logit,) + outputs[2:]
           return ((segmentation

