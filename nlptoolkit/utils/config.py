#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:01:31 2019

@author: weetee
"""
import sys
sys.path.insert(1, '../')

class Config(object):
    def __init__(self, task):
        if task == 'punctuation_restoration':
            self.data_path = "./data/train.tags.en-fr.en"
            self.level = "bpe"
            self.bpe_word_ratio = 0.7
            self.bpe_vocab_size = 7000
            self.batch_size = 32
            self.d_model = 512
            self.ff_dim = 2048
            self.num = 6
            self.n_heads = 8
            self.max_encoder_len = 200
            self.max_decoder_len = 200
            self.LAS_embed_dim = 512
            self.LAS_hidden_size = 512
            self.num_epochs = 500
            self.lr = 0.0003
            self.gradient_acc_steps = 2
            self.max_norm = 1.0
            self.T_max = 5000
            self.model_no = 0
            self.train = 1
            self.infer = 0

