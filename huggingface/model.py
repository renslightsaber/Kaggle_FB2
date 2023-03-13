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

## Transforemr Import
from transformers import AutoTokenizer, AutoModel, AutoConfig


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
        # self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask, 
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, attention_mask)
        out = self.drop(out)
        outputs = self.fc(out)
        # outputs = self.softmax(outputs)
        return SequenceClassifierOutput(logits=outputs)
      
      
      
