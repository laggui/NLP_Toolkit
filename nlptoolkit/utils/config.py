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
            self.data_path = "/project/cq-training-1/project2/teams/team14/data/unaligned.en"
            self.level = "bpe"
            self.bpe_word_ratio = 0.7
            self.bpe_vocab_size = 7000
            self.batch_size = 128
            self.d_model = 512
            self.ff_dim = 2048
            self.num = 6
            self.n_heads = 8
            self.max_encoder_len = 96
            self.max_decoder_len = 96
            self.LAS_embed_dim = 512
            self.LAS_hidden_size = 512
            self.num_epochs = 500
            self.lr = 5e-4
            self.gradient_acc_steps = 2
            self.max_norm = 1.0
            self.T_max = 5000
            self.model_no = 1
            self.train = 1
            self.infer = 0
            self.checkpoint_path = './data/'

