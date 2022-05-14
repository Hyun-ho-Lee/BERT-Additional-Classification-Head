# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:19:45 2022

@author: 201
"""

import torch
import torch.nn as nn
from packaging import version
import transformers
from transformers import BertModel, BertForSequenceClassification

#%%
class CustomEmbedding(nn.Module):
    def __init__(self, config):
           super().__init__()
           self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
           self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
           self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
           self.summary_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size,padding_idx=config.pad_token_id) #
    
           # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
           # any TensorFlow checkpoint file
           self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
           self.dropout = nn.Dropout(config.hidden_dropout_prob)
           # position_ids (1, len position emb) is contiguous in memory and exported when serialized
           self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
           self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
           
           if version.parse(torch.__version__) > version.parse("1.6.0"):
               self.register_buffer(
                   "token_type_ids",
                   torch.zeros(self.position_ids.size(), dtype=torch.long),
                   persistent=False,
               )
    
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, summary_ids=None, past_key_values_length=0): #
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
 
        seq_length = input_shape[1]
 
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
 
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
 
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        if summary_ids is not None:
            summary_embeddings = self.summary_embeddings(summary_ids) #
        
            embeddings = inputs_embeds + token_type_embeddings + summary_embeddings #
        else:
            embeddings = inputs_embeds + token_type_embeddings
        
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


#%%
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        
        self.word_embeddings_weight = self.model.bert.embeddings.word_embeddings.weight
        self.position_embeddings_weight = self.model.bert.embeddings.position_embeddings.weight
        self.token_type_embeddings_weight = self.model.bert.embeddings.token_type_embeddings.weight
        # self.layernorm_weight = self.model.bert.embeddings.LayerNorm.weight
        # self.layernorm_bias = self.model.bert.embeddings.LayerNorm.bias
        
        self.model.bert.embeddings = CustomEmbedding(self.model.config)
        self.model.bert.embeddings.word_embeddings.weight = self.word_embeddings_weight
        self.model.bert.embeddings.position_embeddings.weight = self.position_embeddings_weight
        self.model.bert.embeddings.token_type_embeddings.weight = self.token_type_embeddings_weight
        # self.model.bert.embeddings.LayerNorm.weight = self.layernorm_weight
        # self.model.bert.embeddings.LayerNorm.bias = self.layernorm_bias
        
        self.bert = self.model.bert
        self.dropout = self.model.dropout
        self.classifier = self.model.classifier
        self.pooler = self.model.bert.pooler
        
    def forward(self, input_ids,token_type_ids,attention_mask, labels, summary_ids=None, summ=False):
        if summ:
            embedding_output = self.bert.embeddings(input_ids=input_ids, token_type_ids = token_type_ids, summary_ids=summary_ids)
            encoder_outputs = self.bert.encoder(embedding_output)
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)
        
        else:
            outputs = self.bert(input_ids=input_ids,
                                position_ids=attention_mask,
                                )
            pooled_output = outputs[1]    
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return logits,loss
    

