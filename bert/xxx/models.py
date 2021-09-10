import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

import transformers
# from transformers import RobertaTokenizer
# from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
# from transformers.activations import gelu
# from transformers.file_utils import (
#     add_code_sample_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     replace_return_docstrings,
# )
from transformer.modeling_outputs import NextSentencePredictorOuput



class xxxForSegmentation(BertPreTrainedModel):

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)
        # This classifier on [CLS] is prepared for determining the sgement [eos]
        # Need to be finetuned
        self.cls = nn.Linear(config.hidden_size, 2)

        if self.model_args.do_mlm:
            self.km_head = BertLMPredictionHead(config)

    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        haed_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None
        ):
           
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
        # output[0]: last_hidden_state, contextualized embed 
        #contextualized_embeddings = outputs[0]

        # output[1]: pooler_output, the logits of [CLS]
        pooled_output = outputs[1]
        segment_clf_logit = self.cls(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLss()
            # first loss: segmentation lss
            sgement_clf_loss = loss_fct(
                    segment_clf_logit.view(-1, 2),
                    labels.view(-1))
            # Second loss: MLM in-domain loss

        if not return_dict:
            output = (segment_clf_loss, ) + outputs[2:]

        return NextSentencePredictorOuput(
            loss=segment_clf_loss,
            logits=segment_clf_logit,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
