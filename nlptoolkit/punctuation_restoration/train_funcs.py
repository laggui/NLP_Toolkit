# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:11:29 2019

@author: tsd
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from .models.Transformer import PuncTransformer, PuncTransformer2
from .models.LSTM_attention_model import puncLAS, puncLAS2
from .models.py_Transformer import pyTransformer
from .utils.misc import load_pickle, save_as_pickle, CosineWithRestarts
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class TwoHeadedLoss(torch.nn.Module):
    def __init__(self, ignore_idx=1, ignore_idx_p=7, gamma=0.2):
        super(TwoHeadedLoss, self).__init__()
        self.ignore_idx = ignore_idx
        self.ignore_idx_p = ignore_idx_p
        self.gamma = gamma
        self.criterion1 = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)
        self.criterion2 = nn.CrossEntropyLoss(ignore_index=self.ignore_idx_p)
    
    def forward(self, pred, pred_p, labels, labels_p):
        total_loss = self.criterion1(pred, labels) + (self.gamma)*(self.criterion2(pred_p, labels_p))
        return total_loss

class TwoHeadedLoss2(torch.nn.Module):
    def __init__(self, ignore_idx=1, ignore_idx_p=7, gamma=0.2):
        super(TwoHeadedLoss2, self).__init__()
        self.ignore_idx = ignore_idx
        self.ignore_idx_p = ignore_idx
        self.gamma = gamma
    
    def forward(self, pred, pred_p, labels, labels_p):
        pred_error = torch.sum((-labels* 
                                (1e-8 + pred.float()).float().log()), 1)
        pred_p_error = torch.sum((-labels_p* 
                                (1e-8 + pred_p.float()).float().log()), 1)
        total_loss = pred_error + self.gamma*pred_p_error
        return total_loss

def load_model_and_optimizer(args, src_vocab_size, trg_vocab_size, trg2_vocab_size, max_features_length,\
                             max_seq_length, mappings, idx_mappings, cuda):
    '''Loads the model (Transformer or encoder-decoder) based on provided arguments and parameters'''
    paper = 1
    if args.model_no == 0:
        logger.info("Loading PuncTransformer...")
        if paper == 0:        
            net = PuncTransformer(src_vocab=src_vocab_size, trg_vocab=trg_vocab_size, trg_vocab2=trg2_vocab_size, \
                                  d_model=args.d_model, ff_dim=args.ff_dim,\
                                    num=args.num, n_heads=args.n_heads, max_encoder_len=max_features_length, \
                                    max_decoder_len=max_seq_length, mappings=mappings, idx_mappings=idx_mappings)
        elif paper == 1:
            net = PuncTransformer2(src_vocab=src_vocab_size, trg_vocab=trg_vocab_size, trg_vocab2=trg2_vocab_size, \
                                  d_model=args.d_model, ff_dim=args.ff_dim,\
                                    num=args.num, n_heads=args.n_heads, max_encoder_len=max_features_length, \
                                    max_decoder_len=max_seq_length, mappings=mappings, idx_mappings=idx_mappings)
    
    elif args.model_no == 1:
        logger.info("Loading encoder-decoder (puncLAS) model...")
        if paper == 0:
            net = puncLAS(vocab_size=src_vocab_size, listener_embed_size=args.LAS_embed_dim, listener_hidden_size=args.LAS_hidden_size, \
                      output_class_dim=trg_vocab_size, output_class_dim2=trg2_vocab_size,\
                      max_label_len=max_seq_length, max_label_len2=max_seq_length)
        elif paper == 1:
            net = puncLAS2(vocab_size=src_vocab_size, listener_embed_size=args.LAS_embed_dim, listener_hidden_size=args.LAS_hidden_size, \
                      output_class_dim=trg_vocab_size, output_class_dim2=trg2_vocab_size,\
                      max_label_len=max_seq_length, max_label_len2=max_seq_length)
    elif args.model_no == 2:
        logger.info("Loading pyTransformer model...")
        net = pyTransformer(src_vocab=src_vocab_size, trg_vocab=trg_vocab_size, trg_vocab2=trg2_vocab_size, \
                                  d_model=args.d_model, ff_dim=args.ff_dim,\
                                    num=args.num, n_heads=args.n_heads, max_encoder_len=max_features_length, \
                                    max_decoder_len=max_seq_length, mappings=mappings, idx_mappings=idx_mappings)
     
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    criterion = TwoHeadedLoss(ignore_idx=1, ignore_idx_p=idx_mappings['pad'], gamma=0.2)
    
    net, optimizer, scheduler, start_epoch, acc = load_state(net, args, load_best=False, load_scheduler=False)

    if cuda:
        net.cuda()

    return net, criterion, optimizer, scheduler, start_epoch, acc

def load_state(net, args, load_best=False, load_scheduler=False):
    """ Loads saved model and optimizer states if exists """
    base_path = args.checkpoint_path #"./data/"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        if load_best:
            net = net.load_model(best_path)
        else:
            net = net.load_model(checkpoint_path)
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = CosineWithRestarts(optimizer, T_max=args.T_max)
        if load_scheduler:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = CosineWithRestarts(optimizer, T_max=args.T_max)
    return net, optimizer, scheduler, start_epoch, best_pred

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch

def decode_outputs(outputs, labels, vocab_decoder, args):
    if labels.is_cuda:
        l = list(labels[:70].cpu().numpy())
        o = list(torch.softmax(outputs, dim=1).max(1)[1][:70].cpu().numpy())
    else:
        l = list(labels[:70].numpy())
        o = list(torch.softmax(outputs, dim=1).max(1)[1][:70].numpy())
    if args.level == "bpe":
        l = [l]
        o = [o]
    print("Sample Output: ", " ".join(vocab_decoder(o)))
    print("Sample Label: ", " ".join(vocab_decoder(l)))
    
def decode_outputs_p(outputs, labels, inv_idx, inv_map):
    inv_map['pad'] = '<pad>'
    inv_map['eos'] = '<eos>'
    inv_map['word'] = '<word>'
    if labels.is_cuda:
        l = list(labels[:70].cpu().numpy())
        o = list(torch.softmax(outputs, dim=1).max(1)[1][:70].cpu().numpy())
    else:
        l = list(labels[:70].numpy())
        o = list(torch.softmax(outputs, dim=1).max(1)[1][:70].numpy())
    ll = []
    for t in l:
        if t in inv_idx.keys():
            ll.append(inv_idx[t])
        else:
            ll.append('pad')
    ll = [inv_map[a] for a in ll]
    oo = []
    for t in o:
        if t in inv_idx.keys():
            oo.append(inv_idx[t])
        else:
            oo.append('pad')
    oo = [inv_map[a] for a in oo]
    print("Sample Output: ", " ".join(oo))
    print("Sample Label: ", " ".join(ll))