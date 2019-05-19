#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 01:22:49 2019

@author: venkatesh
"""

# =============================================================================
# 
# =============================================================================
def openfile(name):
    data=open(name,"r")
    data=data.read()    
    data=data.split('\n\n')
    return data
# =============================================================================

# =============================================================================
def idxdict(filename) :
    data=openfile(filename)
    words,tags,ner=[],[],{}
    for i in range(len(data)-1):
        ner[data[i].split('\n')[0].split(" ")[0]]=data[i].split('\n')[0].split(" ")[3]
        for j in range(len(data[i].split('\n'))):
            words.append(data[i].split('\n')[j].split(" ")[0])
            tags.append(data[i].split('\n')[j].split(" ")[3])
    tags_=list(set(tags))
    words=list(set(words))
    n_tags = len(tags_)
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags_)}
    return word2idx,tag2idx,n_tags
# =============================================================================
# 
# =============================================================================
def prepare_data(filename):
    data=openfile(filename)    
    word2idx,tag2idx,_=idxdict(filename)
    new_data=[]
    for i in range(len(data)):
        senten=[]
        for j in range(len(data[i].split("\n"))):
            senten.append(data[i].split("\n")[j].split())
        new_data.append(senten)
    new_name_list=[[new_data[j][i][0] for i in range(len(new_data[j]))] for j in range(len(new_data)-1)]
    new_tag_list=[[new_data[j][i][3] for i in range(len(new_data[j]))] for j in range(len(new_data)-1)]
    X=[[new_name_list[i][j] for j in range(len(new_name_list[i]))] for i in range(len(new_name_list)-2)] 
    Y=[[new_tag_list[i][j] for j in range(len(new_tag_list[i]))] for i in range(len(new_tag_list)-2)]
    Z=list(zip(X,Y))
    Z_train=Z[(len(Z)//20):len(Z)]
    Z_test=Z[0:(len(Z)//20)]
    return Z_train,Z_test,word2idx,tag2idx
    
