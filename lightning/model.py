import re
import os
import gc
import time
import random
import string

import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

## torchmetrics
import torchmetrics

from utils import *

# Pytorch Lightning
import pytorch_lightning as pl

## Transforemr Import
from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig, DataCollatorWithPadding

########### Mean Pooling ################
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
      
      
class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        if model_name == 'google/bigbird-roberta-base':
            self.model = AutoModel.from_pretrained(model_name, attention_type="original_full")
        else: 
            self.model = AutoModel.from_pretrained(model_name) 
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 3) # n_classes = 7
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        outputs = self.softmax(outputs)
        
        return outputs
    
      
############### Pytorch Lightning Model ################      
class PLModel(pl.LightningModule):

    def __init__(self,
                 model_name, # = config['model'],
                 lr, # = config['learning_rate'],
                 wd, # = config['weight_decay'],
                 n_classes # = 3, # n_classes
                 ):
        super().__init__()
        self.model = Model(model_name)
        self.lr = lr
        self.wd = wd
        self.n_classes = n_classes

        # Loss Function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, ids, mask):
        return self.model(ids, mask)

    def shared_step(self, batch, stage):

        # Inputs
        ids = batch['input_ids']
        mask = batch['attention_mask']

        # targets
        targets = batch['target']
 
        # y_preds
        y_preds = self.forward(ids, mask) 
        
        # Loss
        loss = self.loss_fn(y_preds, targets)
        
        # Accuracy (https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html)
        acc = torchmetrics.functional.accuracy(y_preds, targets, task="multiclass", num_classes = self.n_classes)

        # F1 Score
        f1 = torchmetrics.functional.f1_score(y_preds, targets, task="multiclass", num_classes = self.n_classes)
        
        self.log(f'{stage}/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f'{stage}/acc', acc, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f'{stage}/f1', f1, on_epoch=True, on_step=True, prog_bar=True)
        
        return {"loss": loss, "acc": acc, "f1": f1}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = self.lr, weight_decay = self.wd)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
      
      
      
