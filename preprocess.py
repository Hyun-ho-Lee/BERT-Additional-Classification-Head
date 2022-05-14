# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:45:08 2022

@author: 201
"""
import nltk
import pandas as pd
import numpy as np
import re
import datasets
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from tqdm import tqdm 
from nltk import tokenize

from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification , AddedToken
from transformers import TextClassificationPipeline
from transformers import BertModel
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizer
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoConfig
from transformers import AutoModel

nltk.download('punkt')

device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu") 


#%%


# full_df= pd.read_csv('imdb2.csv')



# from sklearn.model_selection import train_test_split


# train_df,val_df = train_test_split(full_df,test_size=0.2, shuffle=True, random_state=1004)

# train_df.to_csv('train.csv',index=False)
# val_df.to_csv('val.csv',index=False)

# train_df['label'].value_counts()

# val_df['label'].value_counts()

# train_data = load_dataset('csv', data_files='train.csv')
# train_data = train_data['train']

# val_data = load_dataset('csv', data_files='val.csv')
# val_data = val_data['train']

# train_data.save_to_disk('bert_summary/dataset/train')

# val_data.save_to_disk('bert_summary/dataset/validation')


# def save(train_data,val_data,path):
#     train_data.save_to_disk(f'{path}/train')
#     val_data.save_to_disk(f'{path}/val')
    


# save(train_data, val_data, path='./bert_summary')


# train_data.save_to_disk('bert_summary/train')

   

# article_length=512

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# train_data = datasets.load_from_disk('bert_summary/train')

#%% 

def preprocess_imdb(model_name="bert-base-uncased",article_length=512):

    train_data = datasets.load_from_disk('bert_summary/dataset/train')
    val_data = datasets.load_from_disk('bert_summary/dataset/validation')

    article_length=512

    
    tokenizer = AutoTokenizer.from_pretrained(model_name,max_length=512)
    tokenizer.add_special_tokens({'bos_token':AddedToken('<s>', lstrip=True)})
    tokenizer.add_special_tokens({'eos_token':AddedToken('</s>', lstrip=True)})
    summary_masks = []

    for idx in tqdm(range(len(train_data))):
        s_idx = []
        og_text = tokenize.sent_tokenize(train_data['text'][idx])
        s_text = tokenize.sent_tokenize(train_data['bert_s'][idx])
    
        for j in range(len(og_text)):
            for k in range(len(s_text)):
                if og_text[j] ==s_text[k]:
                    s_idx.append(j)
        og_targets = og_text.copy()
        for i in s_idx:
            og_targets[i] = "<s> " + og_targets[i] + "</s>"
        og_targets = ' '.join(og_targets)
        temp = np.array([-10]+tokenizer.encode(og_targets, add_special_tokens=False)+[-20])
        p_idx = {'start':list(np.where(temp==30522)[0]), 'end':list(np.where(temp==30523)[0])}
        
        summary_mask = np.array([0] * len(tokenizer(train_data['text'][idx])['attention_mask']))
        for i in range(len(p_idx['start'])):
            summary_mask[(p_idx['start'][i]-i*2) : (p_idx['end'][i]-(i*2+2)+1)] = 1
        if len(summary_mask) < article_length:
            summary_masks.append(np.append(summary_mask, np.array([0]*(article_length-len(summary_mask)))))
        else:
            summary_masks.append(summary_mask[:article_length])
            
    
    train_data = train_data.add_column('summary_masks', summary_masks)
    
    summary_masks = []
        
    for idx in tqdm(range(len(val_data))):
        s_idx = []
        og_text = tokenize.sent_tokenize(val_data['text'][idx])
        s_text = tokenize.sent_tokenize(val_data['bert_s'][idx])
        
        for j in range(len(og_text)):
            for k in range(len(s_text)):
                if og_text[j] ==s_text[k]:
                    s_idx.append(j)
        og_targets = og_text.copy()
        for i in s_idx:
            og_targets[i] = "<s> " + og_targets[i] + "</s>"
        og_targets = ' '.join(og_targets)
        temp = np.array([-10]+tokenizer.encode(og_targets, add_special_tokens=False)+[-20])
        p_idx = {'start':list(np.where(temp==30522)[0]), 'end':list(np.where(temp==30523)[0])}
        
        summary_mask = np.array([0] * len(tokenizer(val_data['text'][idx])['attention_mask']))
        for i in range(len(p_idx['start'])):
            summary_mask[(p_idx['start'][i]-i*2) : (p_idx['end'][i]-(i*2+2)+1)] = 1
        if len(summary_mask) < article_length:
            summary_masks.append(np.append(summary_mask, np.array([0]*(article_length-len(summary_mask)))))
        else:
            summary_masks.append(summary_mask[:article_length])
            
        
        val_data = val_data.add_column('summary_masks', summary_masks)


    
    
    
    
    
    
    
    train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    remove_columns=["text", "bert_s"])


    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "token_ids",
                                "label", "summary_masks"],)

    val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    remove_columns=["text", "bert_s"])


    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "token_ids",
                                "label", "summary_masks"],)
    


    return train_data,val_data

    



def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
    batch["input_ids"] = inputs.input_ids
    batch['token_ids'] = inputs.token_type_ids
    batch["attention_mask"] = inputs.attention_mask

    return batch


def save(train_data,val_data,path):
    train_data.save_to_disk(f'{path}/train')
    val_data.save_to_disk(f'{path}/val')
    

if __name__ == '__main__':
    train_data, val_data = preprocess_imdb()
    save(train_data, val_data, path='./bert_summary')



