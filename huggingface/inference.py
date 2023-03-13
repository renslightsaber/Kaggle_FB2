import re
import os
import gc
import time
import random
import string
import joblib

# import wandb

import argparse
import ast

import copy
from copy import deepcopy

# import torchmetrics
# from torchmetrics.classification import BinaryF1Score
# from torchmetrics.classification import BinaryAccuracy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

## Transforemr Import
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
# from transformers import AdamW
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification

# Utils
from tqdm.auto import tqdm, trange

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from dataloader import *
# from new_trainer import *
from model import *
from utils import *



################## trained_model_paths #####################
def trained_model_paths(n_folds = config['n_folds'], 
                        model_save = config['model_save']):
    print("n_folds: ",n_folds )

    model_paths_f1 = []
    for num in range(0, n_folds):
        model_paths_f1.append(model_save + f"Loss-Fold-{num}_f1.bin")

    print(len(model_paths_f1))
    print(model_paths_f1)
    return model_paths_f1

  
  
##################### inference ###########################  
def inference(model_paths, 
              model_name,
              model_type,
              test_dataset,
              tokenizer, 
              collate_fn,
              bs,
              device):

    final_preds = []
    
    for i, path in enumerate(model_paths):
        if model_type == "AutoModelForSequenceClassification":
            model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        else:
            model = Model(model_name).to(device)
            model.load_state_dict(torch.load(path))
        
        print(f"Getting predictions for model {i+1}")
        training_args = TrainingArguments(output_dir=".",
                                          per_device_eval_batch_size = bs,
                                          label_names=["target"])
        
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator
                          )
        
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions
        final_preds.append(preds)
    
    # 그리고 평균을 내줍니다.
    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds  
  

    
def define():
    p = argparse.ArgumentParser()

    p.add_argument('--base_path', type = str, default = "./data/", help="Data Folder Path")
    p.add_argument('--model_save', type = str, default = "./models/", help="Data Folder Path")
    p.add_argument('--sub_path', type = str, default = "./submission/", help="Data Folder Path")
    
    p.add_argument('--hash', type = str, default = "HuggingFace", help="HASH NAME")
    
    p.add_argument('--model', type = str, default = "microsoft/deberta-v3-base", help="HuggingFace Pretrained Model")    
    p.add_argument('--model_type', type = str, default = "AutoModelForSequenceClassification", help="Fine-Tuning Model Type")
    
    p.add_argument('--n_folds', type = int, default = 3, help="Folds")
    p.add_argument('--n_epochs', type = int, default = 3, help="Epochs")
    
    p.add_argument('--seed', type = int, default = 2022, help="Seed")
    p.add_argument('--valid_bs', type = int, default = 16, help="Batch Size")
    p.add_argument('--max_length', type = int, default = 512, help="Max Length")
    
    p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?")

    config = p.parse_args()
    return config 
  

def main(config):
        
    HASH_NAME = config.hash
    print("HASH_NAME: ", HASH_NAME )
    group_name = f'{HASH_NAME}-Baseline'
    print("Group Name: ", group_name)
    
    ## Data
    test, ss = kaggle_competition_data(base_path = config.base_path, 
                                    stage = "test", 
                                    ratio = config.use_ratio)
    print(test.shape, ss.shape)
    print(train.head(2))
    
    ## Set Seed
    set_seed(config.seed)
    
    ## Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    print("Tokenizer Downloaded")

    # Device
    if config.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
    elif config.device == "cuda":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
    else:
        device = torch.device("cpu")

    print("Device", device)
    
    # test_ds
    test_ds = MyDataset(test, 
                        tokenizer = tokenizer,
                        max_length = config.max_length,
                        mode = "test")
    
    # model_paths
    model_paths_f1 = trained_model_paths(n_folds = config.n_folds,
                                         model_save = config.model_save)
                                        
    ## Inference GoGo
    f1_preds = inference(model_paths_f1, 
                         model_name = config.model, 
                         model_type = config.model_type,
                         test_dataset = test_ds,
                         tokenizer = tokenizer, 
                         collate_fn = DataCollatorWithPadding(tokenizer=tokenizer ),
                         bs = config.valid_bs,
                         device= device)
    
    print("Shape of f1_preds: ", f1_preds.shape)
    
    ## Submission File  
    print("Before completing sample_submission.csv")
    print(ss.shape)
    print(ss.head(3))
    
    ss['Adequate'] = f1_preds[:, 0]
    ss['Effective'] = f1_preds[:, 1]
    ss['Ineffective'] = f1_preds[:, 2]
    
    print("After completing sample_submission.csv")
    print(ss.shape)
    print(ss.head(3))

    # submission.csv saved
    sub_file_name = config.sub_path + "_".join(config.model.split("/")) + "_folds_" + str(config.n_folds) + "_epochs_" + str(config.n_epochs) + ".csv"
    print(sub_file_name)
    
    ss.to_csv(sub_file_name, index=False) 
    print("submission file saved")
    
    
if __name__ == '__main__':
    config = define()
    main(config)
    
    
    
