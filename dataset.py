# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:03:29 2022

@author: 201
"""

#import os
import torch
import datasets

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

#from transformers import BertTokenizer


#%%
class BertDataset(Dataset):
    def __init__(self, val=False, test=False):
        self.test = test
        self.val = val
        
        if val:
            self.dataset = datasets.load_from_disk('data/val')
        elif test:
            self.dataset = datasets.load_from_disk('data/test')
        else:
            self.dataset = datasets.load_from_disk('data/train')

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.dataset[idx]['input_ids'])
        token_type_ids = torch.tensor(self.dataset[idx]['token_ids'])
        attention_mask = torch.tensor(self.dataset[idx]['attention_mask'])
        labels = torch.tensor(self.dataset[idx]['label'])
        
        if self.test:
            return input_ids, token_type_ids, attention_mask, labels
        
        summary_ids = torch.tensor(self.dataset[idx]['summary_masks'])
        
        return input_ids,token_type_ids,attention_mask, summary_ids, labels
        #return input_ids, token_ids, attention_mask, labels
    
    def __len__(self):
        return len(self.dataset)


def get_loader(batch_size, model_name,num_workers):
  
    
    train_loader = DataLoader(dataset=BertDataset(),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=BertDataset(val=True),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return train_loader, val_loader

def get_dist_loader(batch_size,model_name, num_workers):

    
    train_dataset = BertDataset()
    val_dataset = BertDataset(val=True)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler,
                              pin_memory=True,
                              batch_size=batch_size,
                              shuffle=None,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=val_dataset,
                            sampler=val_sampler,
                            pin_memory=True,
                            batch_size=batch_size,
                            shuffle=None,
                            num_workers=num_workers)
    
    return train_loader, val_loader, train_sampler, val_sampler


    