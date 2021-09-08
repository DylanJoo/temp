import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions



class xxx(BertPreTrainedModel):

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)

        if self.model_args.do_mlm:
            self.km_head = BertLMPredictionHead(config)

    
    # def forward(self,
    #     input_ids=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     haed_mask=None,
    #     inputs_embeds=None,
    #     labels=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     return_dict=None,
    #     sent_emb=False,
    #     mlm_input_ids=None,
    #     mlm_labels=None):
