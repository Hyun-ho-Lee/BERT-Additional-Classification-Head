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
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.bert = self.model.bert
        self.dropout = self.model.dropout
        self.classifier = self.model.classifier
        self.projection = nn.Sequential(nn.Linear(768, 768),
                                        nn.Tanh())
        self.p_classifier = nn.Linear(768, 2)
        
    def forward(self, input_ids= None,attention_mask = None,token_type_ids = None,summary_ids=None,labels =None, summ=False):
        outputs = self.bert(input_ids = input_ids,
                            token_type_ids = token_type_ids,
                            attention_mask = attention_mask)
        
        last_hidden_state = outputs[0]
        pooler_output = outputs[1] # [CLS]에서 진행 
        
        loss_fct = nn.CrossEntropyLoss()
        

        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        
        
        loss = loss_fct(logits, labels)
        
        
        
        if summ:
            masked_last_hidden_state = last_hidden_state.mul(summary_ids.unsqueeze(2)) # (batch,seq_len,hidden_size)
            proj = self.avg_pool(masked_last_hidden_state, attention_mask)
            proj = self.projection(proj)
            proj = self.dropout(proj)
            s_logits = self.p_classifier(proj)
            
            s_loss = loss_fct(s_logits, labels)
            
            return s_loss,loss
            

        return logits,loss

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length
        
        return avg_hidden

#%% 


# model = BertModel.from_pretrained("bert-base-uncased")

# model2 = BertClassifier()

# model.embeddings = model2.bert.embeddings

# model.embeddings = model2.bert.embeddings

# model

# model2.bert.embeddings


