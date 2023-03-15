import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


######################## MyDataset ########################
class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, mode = "train"):
        self.dataset = df
        self.mode = mode
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.discourse_type = df['discourse_type'].values
        self.discourse = df['discourse_text'].values
        self.essay = df['essay_text'].values
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        discourse_type = self.discourse_type[index]
        discourse = self.discourse[index]
        essay = self.essay[index]

        # This is what we tried
        # text = discourse_type + ' ' + discourse + self.tokenizer.sep_token + essay

        # One of Winner's Solution
        text = "</" + discourse_type + '_START>' + " " + discourse + "</" + discourse_type + '_END>' + " " + self.tokenizer.sep_token + " " + essay

        inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            # padding='max_length', 
            max_length = self.max_length, 
            truncation=True, 
            )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
    
        if self.mode == "train":
            y = self.dataset['discourse_effectiveness'][index]
            return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': y}
        else:
            return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],}
          
          
          
#################### Prepare Loader ########################          
def prepare_loader(train, 
                   fold, 
                   tokenizer, # = tokenizer, 
                   max_length, # = config['max_length'], 
                   bs, # = config['train_batch_size'],
                   collate_fn, # = DataCollatorWithPadding(tokenizer)
                   ):
    
    train_df = train[train.kfold != fold].reset_index(drop=True)
    valid_df = train[train.kfold == fold].reset_index(drop=True)

    ## train, valid -> Dataset
    train_ds = MyDataset(train_df, 
                            tokenizer = tokenizer ,
                            max_length = max_length,
                            mode = "train")

    valid_ds = MyDataset(valid_df, 
                            tokenizer = tokenizer ,
                            max_length = max_length,
                            mode = "train")
    
    # Dataset -> DataLoader
    train_loader = DataLoader(train_ds,
                              batch_size = bs, 
                              collate_fn=collate_fn, 
                              num_workers = 2,
                              shuffle = True, 
                              pin_memory = True, 
                              drop_last= True)

    valid_loader = DataLoader(valid_ds,
                              batch_size = bs,
                              collate_fn=collate_fn,
                              num_workers = 2,
                              shuffle = False, 
                              pin_memory = True,)
    
    print("DataLoader Completed")
    return train_loader, valid_loader
  
