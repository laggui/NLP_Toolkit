# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:23:00 2019

@author: WT
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from .models.BERT.modeling_bert import BertForTokenClassification
from .utils.misc_utils import load_pickle, save_as_pickle, CosineWithRestarts
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_model_and_optimizer(args, cuda=False):
    '''Loads the model (Transformer or encoder-decoder) based on provided arguments and parameters'''
    
    if args.model_no == 0:
        logger.info("Loading pre-trained BERT for token classification...")
        net = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=args.num_classes)
        
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    if cuda:
        net.cuda()
        
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding tokens
    optimizer = optim.Adam([{"params":net.bert.parameters(),"lr": args.lr/2},\
                             {"params":net.classifier.parameters(), "lr": args.lr}])
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10,13,15,17,20,23,25], gamma=0.8)
    scheduler = CosineWithRestarts(optimizer, T_max=330)
    
    start_epoch, acc = load_state(net, optimizer, scheduler, args, load_best=False)

    

    return net, criterion, optimizer, scheduler, start_epoch, acc

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
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
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred

def load_state1(net, args, load_best=False, load_scheduler=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
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
        optimizer = optim.Adam([{"params":net.bert.parameters(),"lr": args.lr/5},\
                             {"params":net.classifier.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10,13,15,17,20,23,25], gamma=0.8)
        if load_scheduler:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    else:
        optimizer = optim.Adam([{"params":net.bert.parameters(),"lr": args.lr/5},\
                             {"params":net.classifier.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10,13,15,17,20,23,25], gamma=0.8)
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

def evaluate(output, labels):
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != 0).nonzero().squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]; #print(output.shape, o_labels.shape)
    if len(idxs) > 1:
        return (labels[idxs] == o_labels[idxs]).sum().item()/len(idxs)
    else:
        return (labels[idxs] == o_labels[idxs]).sum().item()

def evaluate_results(net, data_loader, cuda, g_mask1, g_mask2, args):
    acc = 0
    print("Evaluating...")
    with torch.no_grad():
        net.eval()
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            if args.model_no == 0:
                src_input = data[0]
                labels = data[1].contiguous().view(-1)
                src_mask = (src_input != 0).float()
                token_type = torch.zeros((src_input.shape[0], src_input.shape[1]), dtype=torch.long)
                if cuda:
                    src_input = src_input.cuda().long(); labels = labels.cuda().long()
                    src_mask = src_mask.cuda(); token_type=token_type.cuda()
                outputs = net(src_input, attention_mask=src_mask, token_type_ids=token_type)
                outputs = outputs[0]
                
            elif args.model_no == 1:
                src_input, trg_input = data[0], data[1][:, :-1]
                labels = data[1][:,1:].contiguous().view(-1)
                if cuda:
                    src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                outputs = net(src_input, trg_input)
            
            #print(outputs.shape); print(labels.shape)
            outputs = outputs.reshape(-1, outputs.size(-1))
            acc += evaluate(outputs, labels)
    return acc/(i + 1)

def decode_outputs(outputs, labels, vocab_decoder, args, reshaped=True):
    if reshaped:
        if labels.is_cuda:
            l = list(labels[:50].cpu().numpy())
            o = list(torch.softmax(outputs, dim=1).max(1)[1][:50].cpu().numpy())
        else:
            l = list(labels[:50].numpy())
            o = list(torch.softmax(outputs, dim=1).max(1)[1][:50].numpy())
        
        print("Sample Output: ", " ".join(vocab_decoder[oo] for oo in o))
        print("Sample Label: ", " ".join(vocab_decoder[ll] for ll in l))
    
    else:
        if labels.is_cuda:
            l = list(labels[0,:].cpu().numpy())
            o = list(torch.softmax(outputs, dim=2).max(2)[1].cpu().numpy())
        else:
            l = list(labels[0,:].numpy())
            o = list(torch.softmax(outputs, dim=2).max(2)[1].numpy())
        print("Sample Output: ", " ".join(vocab_decoder[oo] for oo in o))
        print("Sample Label: ", " ".join(vocab_decoder[ll] for ll in l))