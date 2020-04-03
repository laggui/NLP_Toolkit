import torch

from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def evaluate_results(net, data_loader, cuda, g_mask1, g_mask2, args, create_masks, create_trg_mask, ignore_idx2=7):
    acc = 0; acc2 = 0
    print("Evaluating...")
    out_labels = []; true_labels = []
    with torch.no_grad():
        net.eval()
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            if args.model_no == 0:
                src_input, trg_input, trg2_input = data[0], data[1][:, :-1], data[2][:, :-1]
                labels = data[1][:,1:].contiguous().view(-1)
                labels2 = data[2][:,1:].contiguous().view(-1)
                src_mask, trg_mask = create_masks(src_input, trg_input)
                trg2_mask = create_trg_mask(trg2_input, ignore_idx=ignore_idx2)
                if cuda:
                    src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda(); trg2_mask = trg2_mask.cuda()
                    trg2_input = trg2_input.cuda().long(); labels2 = labels2.cuda().long()
                outputs, outputs2 = net(src_input, trg_input, trg2_input, src_mask, trg_mask, trg2_mask)
                
            elif args.model_no == 1:
                src_input, trg_input, trg2_input = data[0], data[1][:, :-1], data[2][:, :-1]
                labels = data[1][:,1:].contiguous().view(-1)
                labels2 = data[2][:,1:].contiguous().view(-1)
                if cuda:
                    src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    trg2_input = trg2_input.cuda().long(); labels2 = labels2.cuda().long()
                outputs, outputs2 = net(src_input, trg_input, trg2_input)
                
            outputs = outputs.view(-1, outputs.size(-1))
            outputs2 = outputs2.view(-1, outputs2.size(-1))
            acc += evaluate(outputs, labels, ignore_idx=1)[0]
            cal_acc, (o, l) = evaluate(outputs2, labels2, ignore_idx=ignore_idx2)
            out_labels.append([str(i) for i in o]); true_labels.append([str(i) for i in l])
            acc2 += cal_acc
    accuracy = (acc/(i + 1) + acc2/(i + 1))/2
    results = {
        "accuracy": accuracy,
        "precision": precision_score(true_labels, out_labels),
        "recall": recall_score(true_labels, out_labels),
        "f1": f1_score(true_labels, out_labels)
    }

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    return accuracy

def evaluate(output, labels, ignore_idx):
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != ignore_idx).nonzero().squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]; #print(output.shape, o_labels.shape)
    l = labels[idxs]; o = o_labels[idxs]
    
    if len(idxs) > 1:
        acc = (l == o).sum().item()/len(idxs)
    else:
        acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()
    return acc, (o, l)