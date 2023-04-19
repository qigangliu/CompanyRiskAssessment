import numpy as np
from model import *
import torch
import torch.nn as nn
import argparse
from dgl.data.utils import load_graphs
from sklearn.metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_hid',   type=int, default=8)
parser.add_argument('-s', '--seed', type=int, default=42,help='Random seed')
parser.add_argument('--clip',    type=int, default=1) 
parser.add_argument('--max_lr',  type=float, default=5e-1) 
args = parser.parse_args()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(model, G):
    best_val_acc = torch.tensor(0).to(device)
    best_test_acc = torch.tensor(0).to(device)
    train_step = torch.tensor(0).to(device)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits = model(G, 'company')
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx]) #to(device)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        if epoch % 5 == 0:
            model.eval()
            logits = model(G, 'company')
            pred   = logits.argmax(1)
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
            val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
            test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print('Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                epoch,
                optimizer.param_groups[0]['lr'], 
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
            ))
    preds = logits.max(1)[1]
    if device != 'cpu':
        l = labels[test_idx].cpu().detach().numpy()
        p = preds[test_idx].cpu().detach().numpy()
    else:
        l = labels[test_idx].detach().numpy()
        p = preds[test_idx].detach().numpy()
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

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
graph_path = "../../data/others/company_graph.bin"
graph = load_graphs(graph_path)
G = graph[0][0].to(device)
labels = G.nodes['company'].data['label'].long().to(device)
num_classes = 2
total_len = len(labels)
idx_all = np.arange(total_len)

np.random.shuffle(idx_all)
train_idx = torch.tensor(idx_all[:int(total_len/10*7)]).long()
val_idx = torch.tensor(idx_all[int(total_len/10 *4):int(total_len/10 *7)]).long() 
test_idx = torch.tensor(idx_all[int(total_len/10*7):]).long()

train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    G.edges[etype].data['id'] = (torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]).to(device)

features = nn.Parameter(G.nodes['company'].data['feat'], requires_grad = False)
features = features.to(device)
G.nodes['company'].data['inp'] = features
emb = nn.Parameter(G.nodes['person'].data['feat'], requires_grad = False)
linear = torch.nn.Linear(emb.size()[1],features.size()[1]).to(device)
emb = linear(emb)
emb = emb.to(device)
G.nodes['person'].data['inp'] = emb

model = HGT(G,
            node_dict, edge_dict,
            n_inp=G.nodes['company'].data['inp'].size()[1],
            n_hid=args.n_hid,
            n_out=labels.max().item()+1,
            n_layers=2,
            n_heads=8,
            use_norm = True).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
print('Training HGT with #param: %d' % (get_n_params(model)))
train(model, G)