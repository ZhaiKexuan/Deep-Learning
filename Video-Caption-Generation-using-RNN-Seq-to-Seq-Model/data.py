# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:33:39 2020

@author: kzhai
"""
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence
import numpy as np
import pickle
import os
import json

filters = '`","?!/.()'

class dataset():
    
    def shuffle_data(self):
        np.random.shuffle(self.dataIdx)
    
    def __init__(self,batchSize,feat,label=None,len_maxcap=50):
        self.feat = feat
        self.label = label
        self.batchSize = batchSize
        self.len_maxcap = len_maxcap
        self.dataSize = None
        self.dataIdx = None
        self.id_list = None
        self.batchIdx = 0
        self.cap_list = None
        self.len_cap_list = None
        self.vocab_size = 0
    
    def generate_token(self):
        self.tokenizer = Tokenizer(filters=filters,split=" ")
        total_list = []
        with open(self.label) as f:
            raw_data = json.load(f)
        for vid in raw_data:
            for cap in vid['caption']:
                total_list.append(cap)
        self.tokenizer.fit_on_texts(total_list)
        self.vocab_size = len(self.tokenizer.word_index)
        self.tokenizer.fit_on_texts(['<PAD>','<BOS>','<EOS>','<UNK>'])
    
        
    def process_data(self):
        
        pad = self.tokenizer.texts_to_sequences(['<PAD>'])[0]
        id_list = []
        cap_list = []
        len_cap_list = []
        self.feat_data = {}
        with open(self.label) as f:
            raw_data = json.load(f)
        for vid in raw_data:
            vid_id = vid['id']
            self.feat_data[vid_id] = np.load(self.feat + vid_id + '.npy')

            for caption in vid['caption']:
                w = text_to_word_sequence(caption)
                for i in range(len(w)):
                    if w[i] not in self.tokenizer.word_index:
                        w[i] = '<UNK>'
                w.append('<EOS>')
                one_hot = self.tokenizer.texts_to_sequences([w])[0]
                len_cap = len(one_hot)
                one_hot += pad * (self.len_maxcap - len_cap)
                id_list.append(vid_id)
                cap_list.append(one_hot)
                len_cap_list.append(len_cap)
                
        self.id_list = np.array(id_list)
        self.cap_list = np.array(cap_list)
        self.len_cap_list = np.array(len_cap_list)
        self.dataSize = len(self.cap_list)
        self.dataIdx = np.arange(self.dataSize,dtype=np.int)
        
    def save_vocab(self):
        with open('tokenizer.pickle', 'wb') as process:
            pickle.dump(self.tokenizer, process, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_token(self):
        with open('tokenizer.pickle', 'rb') as process:
            self.tokenizer = pickle.load(process)
        self.vocab_size = len(self.tokenizer.word_index) - 4

        
    def next_batch(self):
        if self.batchIdx + self.batchSize <= self.dataSize:
            idx = self.dataIdx[self.batchIdx:(self.batchIdx+self.batchSize)]
            self.batchIdx += self.batchSize
        else:
            idx = self.dataIdx[self.batchIdx:-1]
            self.batchIdx = 0
        id_batch = self.id_list[idx]
        cap_batch = self.cap_list[idx]
        len_cap_batch = self.len_cap_list[idx]
        batchSuit = []
        for vid_id in id_batch:
            batchSuit.append(self.feat_data[vid_id])
        batchSuit = np.array(batchSuit)
        return id_batch,batchSuit,cap_batch,len_cap_batch
    
    
    def new_process_data(self):
        id_list = []
        self.feat_data = {}
        for filename in os.listdir(self.feat):
            if filename.endswith('.npy'):
                vid_id = os.path.splitext(filename)[0]
                self.feat_data[vid_id] = np.load(self.feat + filename)
                id_list.append(vid_id)
            self.id_list = np.array(id_list)
        self.dataSize = len(self.id_list)
        self.dataIdx = np.arange(self.dataSize,dtype=np.int)
    
    def next_feature_batch(self):
        if self.batchIdx + self.batchSize <= self.dataSize:
            idx = self.dataIdx[self.batchIdx:(self.batchIdx+self.batchSize)]
            self.batchIdx += self.batchSize
        else:
            idx = self.dataIdx[self.batchIdx:]
            self.batchIdx = 0
        id_batch = self.id_list[idx]
        feat_batch = []
        for vid_id in id_batch:
            feat_batch.append(self.feat_data[vid_id])
        feat_batch = np.array(feat_batch)
        return id_batch,feat_batch