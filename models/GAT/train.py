"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from sklearn.metrics import *
from gat import GAT
from utils import EarlyStopping
from dgl.data.utils import load_graphs
import torch.nn as nn

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        temp = logits
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels),temp

def load_data():
    graph_path = "../../data/others/company_graph.bin"
    graph = load_graphs(graph_path)
    return graph[0]

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(args.seed)

    # load graph and features
    data = load_data()
    hg = data[0]
    if args.gpu < 0:
        labels = hg.nodes['company'].data['label']
    else:
        labels = hg.nodes['company'].data['label'].to(args.gpu)

    features = nn.Parameter(hg.nodes['company'].data['feat'], requires_grad = False)
    hg.nodes['company'].data['feat'] = features

    p_id_feature = torch.tensor(np.zeros((hg.nodes['person'].data['feat'].size()[0],
                                hg.nodes['company'].data['feat'].size()[1])),dtype=torch.float32)
    hg.nodes['person'].data['feat'] = p_id_feature

    idx_all = np.arange(len(labels))
    total_len = len(idx_all)
    np.random.shuffle(idx_all)


    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        hg = hg.int().to(args.gpu)

    g = dgl.to_homogeneous(hg,ndata=['feat'])
    if args.gpu < 0:
        train_mask = torch.LongTensor(idx_all[:int(total_len/10*7)])
        val_mask = torch.LongTensor(idx_all[int(total_len/10*4):int(total_len/10*7)])
        test_mask = torch.LongTensor(idx_all[int(total_len/10*7):])
        features = g.ndata['feat']
    else:
        train_mask = torch.cuda.LongTensor(idx_all[:int(total_len/10*7)])
        val_mask = torch.LongTensor(idx_all[int(total_len/10*4):int(total_len/10*7)])
        test_mask = torch.cuda.LongTensor(idx_all[int(total_len/10*7):])
        features = g.ndata['feat'].to(args.gpu)
    
    num_feats = features.shape[1]
    n_classes = 2

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create model
    heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc, _= evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} ".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc,)) 

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    acc,log = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))

    preds = log.max(1)[1]
    if cuda:
        l = labels[test_mask].cpu().detach().numpy()
        p = preds[test_mask].cpu().detach().numpy()
    else:
        l = labels[test_mask].detach().numpy()
        p = preds[test_mask].detach().numpy()
    acc = accuracy_score(l,p)
    recall = recall_score(l,p)
    f1 = f1_score(l,p)
    pre = f1 * recall /(2 * recall - f1)
    print()
    print("Test set results:")
    print("      Accuracy = {:.4f}".format(acc))
    print("     Precision = {:.4f}".format(pre))
    print("        Recall = {:.4f}".format(recall))
    print("      F1-score = {:.4f}".format(f1))
    print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=8,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)
