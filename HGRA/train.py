#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import math
import urllib.request

import numpy as np
import scipy.io
from model import *

import dgl


torch.manual_seed(0)

parser = argparse.ArgumentParser(
    description="Training HGRA on company graph data"
)


parser.add_argument("--n_epoch", type=int, default=400)
parser.add_argument("--n_hid", type=int, default=128)
parser.add_argument("--n_inp", type=int, default=100)
parser.add_argument("--clip", type=int, default=1.0)
parser.add_argument("--max_lr", type=float, default=1e-3)

args = parser.parse_args()


def get_n_params(model): # calculate the number of params
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def train(model, G):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits,node_h = model(G, "company") 
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device)) # compute loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        # scheduler.step(train_step) # lr tuning
        if epoch % 5 == 0:
            model.eval()
            logits,node_h = model(G, "company")
            pred = logits.argmax(1).cpu() 
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean() # accuracy
            val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
            test_acc = (pred[test_idx] == labels[test_idx]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print(
                "Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)"
                % (
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                )
            )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

truefeatures = np.load('.\data_processed\data_x.npy') # load node attributes
truelabels = np.load('.\data_processed\data_y.npy') # node lable
labels = torch.tensor(truelabels).long() # generate labels

# load graph stucture
G=dgl.load_graphs(".\data_processed\heteroGraph.bin")[0][0]
emb = nn.Parameter(
    torch.Tensor(G.num_nodes('person'), 100), requires_grad=False # create feature vector for person nodes
)
nn.init.xavier_uniform_(emb)

G.nodes['company'].data["inp"] = torch.Tensor(truefeatures) 
G.nodes['person'].data["inp"] = emb

# print(G)


seedcompanyid = np.load('.\data_processed\seedcompany.npy')
idx=np.load('.\data_processed\train_test_split.npz') # split of training set and test set

train_idx = torch.tensor(seedcompanyid[idx['train_idx']]).long()
val_idx = torch.tensor(seedcompanyid[idx['train_idx']]).long()
test_idx = torch.tensor(seedcompanyid[idx['test_idx']]).long()


node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict) # encode the node type to number
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict) # encode the relation type to number
    G.edges[etype].data["id"] = (
        torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype] 
    )


G = G.to(device)

model = HGRA(
    G,
    node_dict,
    edge_dict,
    n_inp=args.n_inp, # input dimension 256
    n_hid=args.n_hid, # hidden dimension
    n_out=2, # output classes
    n_layers=2,
    n_heads=4, # multi-head
    use_norm=True,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.07,lr=0.001)
print("Training HGRA with #param: %d" % (get_n_params(model)))
train(model, G)
torch.save(model.state_dict(),'HGRA_model.pth')
logits,node_h = model(G, "company")
logits=logits.cpu()
node_h=node_h.cpu().numpy()
# print(node_h[0])
np.savez('HGRA_result.npz',ypre=logits[test_idx].detach().numpy(),ytrue=labels[test_idx].detach().numpy())
np.save('node_h.npy',node_h)





