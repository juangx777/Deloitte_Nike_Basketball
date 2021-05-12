## Data manipulation
import pandas as pd
import numpy as np
import pyrsm as rsm
import os
import glob
import re
import warnings
warnings.filterwarnings("ignore")


## Sklearn module
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import (
    GridSearchCV, 
    cross_val_score, 
    train_test_split,
    ShuffleSplit
)
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, naive_bayes
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


## XGBoost module
import xgboost as xgb


## Language Processing
# NLTK module
import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import sent_tokenize as st
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
# Jieba module
import jieba


## Graphing
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns



##torch
import torch
from torch import nn, optim
from torch.utils.data import (
    Dataset, 
    DataLoader, 
    RandomSampler, 
    SequentialSampler
)

import transformers
from transformers import (
    BertForSequenceClassification,
    BertTokenizer, 
    BertModel,
    AdamW,
    BertConfig
)
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# torch.cuda.set_device(opts.gpu)
df = pd.read_excel('nike_basketball.xlsx', sheet_name = 0, engine='openpyxl')

# print(df.head())
df.head(5).transpose()
df.tail(5).transpose()

## Filter rows with DOCID is NaN
df_clean = df[df.DOCID.notnull()]

## Fill NaN for each Column
df_clean.loc[:, 'AUTHOR'] = df_clean.loc[:, 'AUTHOR'].fillna('Missing Author')
df_clean.loc[:, 'SECTION'] = df_clean.loc[:, 'SECTION'].fillna('Missing Section')
df_clean.loc[:, 'INDICATORWORD'] = df_clean.loc[:, 'INDICATORWORD'].fillna('Missing Indicator')

## Define column to correct data type
df_clean['PUBLISHDATE'] = pd.to_datetime(df_clean['PUBLISHDATE'])
df_clean['PUBTYPE'] = df_clean['PUBTYPE'].astype(str)
df_clean['RELEVANT'] = df_clean['RELEVANT'].astype(str)

## Chcek data set's information again
# df_clean.info()

df_clean.loc[df_clean['DOCID'] == 'news:258l^201902255565029(S:435019146)', 'DOCID'] = "news:040g^201902255786312(S:435019146)"

## Define explanatory variable, response variable, and ID variable
evar = 'SENTENCE'
rvar = 'RELEVANT'
idvar = 'DOCID'

## Copy X and y from df_clean and use it for model training
df_clean['RELEVANT'] = df_clean['RELEVANT'].astype(float)
df_clean['RELEVANT'] = df_clean['RELEVANT'].astype(int)
X = np.asarray(df_clean[evar].copy())
y = np.asarray(df_clean[rvar])




os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

## Specify GPU
device = torch.device("cuda")
os.environ['WANDB_SILENT']="true"

class cDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese")

max_seq_len = 100
randomS = 10
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.3,random_state=randomS,shuffle = True)
test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=.5,random_state=randomS,shuffle = True)


train_encodings = tokenizer.batch_encode_plus(list(train_X), truncation=True, max_length = max_seq_len,pad_to_max_length=True)
val_encodings = tokenizer.batch_encode_plus(list(val_X), truncation=True, max_length = max_seq_len,pad_to_max_length=True)
test_encodings = tokenizer.batch_encode_plus(list(test_X), truncation=True, max_length = max_seq_len,pad_to_max_length=True)




train_dataset = cDataset(train_encodings, train_y)
val_dataset = cDataset(val_encodings, val_y)
test_dataset = cDataset(test_encodings, test_y)




training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=20,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    learning_rate=1e-05,
)


trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

from scipy.special import softmax
trainer.train()
trainer.evaluate()
print("===========================================")
result = trainer.predict(test_dataset)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support,roc_auc_score

def compute_metrics(pred):
    labels = pred.label_ids #correct label 
    preds = pred.predictions.argmax(-1) #prediciton array -> prediction label 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

print(compute_metrics(result))