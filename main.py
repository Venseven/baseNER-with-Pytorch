#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:31:04 2019

@author: venkatesh
"""

import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from NER import BiLSTM_CRF
import NER
import preprocess

EMBEDDING_DIM = 5
HIDDEN_DIM = 4
EPOCHS=30
torch.manual_seed(1)
Z_train,Z_test,word2idx,tag2idx=preprocess.prepare_data("train.txt")

tag2idx["<START>"]=len(tag2idx)-2    
tag2idx["<STOP>"]=len(tag2idx)-1
n_tags=len(tag2idx)
# =============================================================================
# MODEL
# =============================================================================

model = BiLSTM_CRF(len(word2idx), tag2idx, EMBEDDING_DIM, HIDDEN_DIM,n_tags)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)



# =============================================================================
# TRAINING
# =============================================================================
for epoch in range(EPOCHS):
    for sentence, tags in Z_train:
        model.zero_grad()
        sentence_in = NER.prepare_sequence(sentence, word2idx)
  
        targets = torch.tensor([tag2idx[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)

        loss.backward()
        optimizer.step()
    print("Epoch number {}".format(epoch))



# =============================================================================
# TESTING
# =============================================================================
with torch.no_grad():
    precheck_sent = NER.prepare_sequence(test, word2idx)
    print(model(precheck_sent))
