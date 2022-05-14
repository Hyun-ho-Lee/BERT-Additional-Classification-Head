# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:41:36 2022

@author: 201
"""
import warnings
warnings.filterwarnings("ignore")


import os
import time
import datasets

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from dataset import get_dist_loader, get_loader
from transformers import AdamW, get_scheduler

from model3 import BertClassifier

from transformers import BertModel, BertForSequenceClassification

from bertmodel import BERT

from dataset import get_loader

from train_utils import (cal_running_avg_loss, eta, progress_bar,
                         time_since, user_friendly_time)






#%%
class Trainer():
    def __init__(self, args):
        self.args = args
#        self.tokenizer = args.tokenizer
        self.model_dir = args.model_dir
        
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None
        self.val_sampler = None
        
        self.model = None
        self.optimizer = None
        
    def make_model_env(self, gpu, ngpus_per_node):
        self.args.gpu = self.args.devices[gpu]
        self.args.rank = self.args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                world_size=self.args.world_size, rank=self.args.rank)
        self.model = BertClassifier()

        
        torch.cuda.set_device(self.args.gpu)
        self.model.cuda(self.args.gpu)
        self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
        self.args.workers = (self.args.workers + ngpus_per_node - 1) // ngpus_per_node
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self.get_data_loader()
        
        param = self.model.parameters()

        
        self.optimizer = AdamW(param, lr=1e-4)
        self.num_training_steps = self.args.num_epochs * len(self.train_loader) 
        self.lr_scheduler = get_scheduler("linear",
                                          optimizer=self.optimizer,
                                          num_warmup_steps=int(0.1*self.num_training_steps),
                                          num_training_steps=self.num_training_steps)
        self.model = DistributedDataParallel(self.model,
                                             device_ids=[self.args.gpu],
                                             find_unused_parameters=True)
        
        cudnn.benchmark = True
        
    def get_data_loader(self):
        # TODO change train file to trans_train_file
        train_loader, val_loader, train_sampler, val_sampler = get_dist_loader(batch_size=self.args.batch_size,
                                                                               num_workers=self.args.workers,
                                                                               model_name=self.args.model_name)
        
        return train_loader, val_loader, train_sampler, val_sampler
    
    def train(self, model_path=None):
        running_avg_loss = 0.0
        running_avg_s_loss = 0.0
        running_avg_b_loss = 0.0
        
        best_loss = 1e5
        batch_nb = len(self.train_loader)
        step = 1
        self.model.zero_grad()
        for epoch in range(1, self.args.num_epochs+1):
            start = time.time()
            self.model.train()
            self.train_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                batch = tuple(v.to(self.args.gpu) for v in batch) # batch : {positive_masks:tensor, negati~~~} => (tensor, tensor, ~~~)
                
                input_ids, token_type_ids, attention_mask, summary_ids, labels = batch
                

                
                s_loss, b_loss  = self.model(input_ids = input_ids,
                                             token_type_ids = token_type_ids,
                                             attention_mask = attention_mask,
                                             summary_ids = summary_ids,
                                             labels = labels,
                                             summ=True
                                             )
                
                loss = s_loss + b_loss
                
                loss.backward()
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()
                
                running_avg_loss = cal_running_avg_loss(loss.item(), running_avg_loss)
                running_avg_b_loss = cal_running_avg_loss(b_loss.item(), running_avg_b_loss)
                running_avg_s_loss = cal_running_avg_loss(s_loss.item(), running_avg_s_loss)
                #msg = "{}/{} {} - ETA : {} - b_loss: {:.4f}".format(
                msg = "{}/{} {} - ETA : {} - loss: {:.4f} b_loss: {:.4f} s_loss: {:.4f}".format(
                    batch_idx, batch_nb,
                    progress_bar(batch_idx, batch_nb),
                    eta(start, batch_idx, batch_nb),
                    running_avg_loss ,
                    running_avg_b_loss,
                    running_avg_s_loss)
                print(msg, end="\r")
                step += 1
                
            # evaluate model on validation set
            if self.args.rank == 0:
                val_loss = self.evaluate(msg)
                if val_loss < best_loss:
                    best_loss = val_loss
                self.save_model(val_loss, epoch)
                # print("Epoch {} took {} -  - Train b_loss: {:.4f} -  - val loss: "
                #       "{:.4f} ".format(epoch,
                #                         user_friendly_time(time_since(start)),
                                     
                #                         running_avg_b_loss,

                #                         val_loss))               

                print("Epoch {} took {} - Train loss: {:.4f} - Train b_loss: {:.4f} - Train s_loss: {:.4f} - val loss: "
                      "{:.4f} ".format(epoch,
                                        user_friendly_time(time_since(start)),
                                        running_avg_loss,
                                        running_avg_b_loss,
                                        running_avg_s_loss,
                                        val_loss))
    
    def evaluate(self, msg):
        val_batch_nb = len(self.val_loader)
        val_losses = []
        self.model.eval()
        for i, batch in enumerate(self.val_loader, start=1):
            batch = tuple(v.to(self.args.gpu) for v in batch)
            
            input_ids, token_type_ids, attention_mask, summary_ids, labels = batch
            
            # extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
            
            with torch.no_grad():
                s_loss, b_loss  = self.model(input_ids = input_ids,
                                             token_type_ids = token_type_ids,
                                             attention_mask = attention_mask,
                                             summary_ids = summary_ids,
                                             labels = labels,
                                             summ=True
                                             )
                
            msg2 = "{} =>   Evaluating : {}/{}".format(msg, i, val_batch_nb)
            print(msg2, end="\r")
#            val_losses.append(b_loss.item())
            val_losses.append(b_loss.item() + s_loss.item())

        val_loss = np.mean(val_losses)

        return val_loss
    
    def save_model(self, loss, epoch):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {"args":self.args,
                "state_dict":model_to_save.state_dict()}
        model_save_path = os.path.join(
            self.model_dir, "{}_{:.4f}.pt".format(epoch, loss))
        torch.save(ckpt, model_save_path)
