import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from BGNN import BGNNPredictor
from sklearn.metrics import *
from torch.nn import Dropout
import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch import GATConv as GATConvDGL
from dgl.nn.pytorch import GraphConv
from hgt_model import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--model_temp', type=str, default='hgt',
                    help='the model combined with gbdt.')
parser.add_argument('--met', type=str, default='accuracy',
                    help='training metric.')
parser.add_argument('--n_hid',   type=int, default=32)
parser.add_argument('-s', '--seed', type=int, default=42,
                    help='Random seed')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
args = parser.parse_args()
epo_num = args.epochs

class GNNModelDGL(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout=0.0,
        name='hgt',
        residual=True,
        use_mlp=False,
        join_with_mlp=False,
        G = None
    ):
        super(GNNModelDGL, self).__init__()
        self.name = name
        self.use_mlp = use_mlp
        self.join_with_mlp = join_with_mlp
        self.normalize_input_columns = True
        if name == "gat":
            self.l1 = GATConvDGL(
                in_dim,
                hidden_dim // 8,
                8,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=False,
                activation=F.elu,
            )
            self.l2 = GATConvDGL(
                hidden_dim,
                out_dim,
                1,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=residual,
                activation=None,
            )
        elif name == "gcn":
            self.l1 = GraphConv(in_dim, hidden_dim, activation=F.elu)
            self.l2 = GraphConv(hidden_dim, out_dim, activation=F.elu)
            self.drop = Dropout(p=dropout)
        elif name == 'hgt':
            self.l1 = HGT(G,
            node_dict, edge_dict,
            n_inp=in_dim,
            n_hid=args.n_hid,
            n_out= 2,
            n_layers=2,
            n_heads=8,
            use_norm = True)


    def forward(self, graph, features):
        h = features
        if self.use_mlp:
            if self.join_with_mlp:
                h = torch.cat((h, self.mlp(features)), 1)
            else:
                h = self.mlp(features)
        if self.name == "gat":
            h = self.l1(graph, h).flatten(1)
            logits = self.l2(graph, h).mean(1)
        elif self.name == "gcn":
            h = self.drop(h)
            h = self.l1(graph, h)
            logits = self.l2(graph, h)
        elif self.name == 'hgt':
            logits = self.l1(graph, h)

        return logits


def load_data():
    graph_ori = load_graphs(graph_path)
    g = graph_ori[0][0]
    feature_company = g.nodes['company'].data['feat']
    feature_person = g.nodes['person'].data['feat']
    labels = g.nodes['company'].data['label'].tolist()

    # convert the graph to homogeneous graph
    g = dgl.to_homogeneous(g)
    g = dgl.add_self_loop(g)

    # generate features of DataFrame format
    p_id_feature = np.zeros(len(feature_person),len(feature_company[0]))
    label_add = [0 for _ in range(len(feature_person))]
    labels.extend(label_add)
    df_label = pd.DataFrame({'class':labels})
    features = np.concatenate([feature_company,p_id_feature],axis=0)
    df_features = pd.DataFrame(features)

    # split data
    total_len = len(labels)
    idx_all = np.arange(total_len)
    np.random.seed(args.seed)
    np.random.shuffle(idx_all)
    train_idx = idx_all[:int(total_len/10 * 7)]
    val_idx = idx_all[int(total_len/10 * 4):int(total_len/10 * 7)]
    test_idx = idx_all[int(total_len/10 * 7):]

    return g,df_features, df_label, train_idx, val_idx, test_idx


def loda_data_hgt():
    graph_ori = load_graphs(graph_path)
    graph_hgt = graph_ori[0][0].to(device)
    labels = graph_hgt.nodes['company'].data['label'].long().to(device)
    
    # companies' features
    features = nn.Parameter(graph_hgt.nodes['company'].data['feat'], requires_grad = False)
    features = features.to(device)
    graph_hgt.nodes['company'].data['inp'] = features
    feature_company = features.cpu().detach().numpy()

    # persons' features
    emb = nn.Parameter(graph_hgt.nodes['person'].data['feat'], requires_grad = False)
    linear = torch.nn.Linear(emb.size()[1],features.size()[1]).to(device)
    emb = linear(emb)
    emb = emb.to(device)
    graph_hgt.nodes['person'].data['inp'] = emb

    # from dgl to dataframe and dict
    total_len = len(labels)
    idx_all = np.arange(total_len)
    X = pd.DataFrame(feature_company)
    y = pd.DataFrame({'class':labels.cpu().detach().numpy()})
    node_dict = {}
    edge_dict = {}
    for ntype in graph_hgt.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in graph_hgt.etypes:
        edge_dict[etype] = len(edge_dict)
    graph_hgt.edges[etype].data['id'] = (torch.ones(graph_hgt.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]).to(device)
    
    # split data
    np.random.seed(args.seed)
    np.random.shuffle(idx_all)
    train_mask = idx_all[:int(total_len/10*4)]
    val_mask = idx_all[int(total_len/10 *4):int(total_len/10 *7)] 
    test_mask = idx_all[int(total_len/10*7):]
    return graph_hgt, X, y, train_mask, val_mask, test_mask,node_dict,edge_dict

if __name__ == "__main__":
    # Different data for HGT or GCN/GAT
    graph_path = "../../data/others/company_graph.bin"
    if args.model_temp == 'hgt':
        graph, X, y, train_mask, val_mask, test_mask,node_dict,edge_dict = loda_data_hgt()
    else:
        graph, X, y, train_mask, val_mask, test_mask = load_data()
    encoded_X = X.copy()
    cat_features = None

    # specify parameters
    task = "classification"
    hidden_dim = 32
    ep = 5
    trees_per_epoch = ep  # 5-10 are good values to try
    backprop_per_epoch = 10  # 5-10 are good values to try
    lr = 0.01  # 0.01-0.1 are good values to try
    append_gbdt_pred = (
        True  # this can be important for performance (try True and False)
    )
    train_input_features = False
    gbdt_depth = 10
    gbdt_lr = 0.1

    out_dim = 2

    in_dim = out_dim + X.shape[1] if append_gbdt_pred else out_dim
    # specify GNN model
    if args.model_temp == 'hgt':
        gnn_model = GNNModelDGL(in_dim, hidden_dim, out_dim,name='hgt',G=graph)
    else:
        gnn_model = GNNModelDGL(in_dim, hidden_dim, out_dim,name=args.model_temp)
        
    # initialize BGNN model
    bgnn = BGNNPredictor(
        gnn_model,
        task=task,
        loss_fn=None,
        trees_per_epoch=trees_per_epoch,
        backprop_per_epoch=backprop_per_epoch,
        lr=lr,
        append_gbdt_pred=append_gbdt_pred,
        train_input_features=train_input_features,
        gbdt_depth=gbdt_depth,
        gbdt_lr=gbdt_lr,
        random_seed=args.seed
    )

    # train
    metrics = bgnn.fit(
        graph,
        encoded_X,
        y,
        train_mask,
        val_mask,
        test_mask,
        original_X=X,
        cat_features=cat_features,
        num_epochs=epo_num,
        patience=100,
        metric_name=args.met,
    )


    bgnn.plot_interactive(
        metrics,
        legend=["train", "valid", "test"],
        title="company dataset",
        metric_name="accuracy",
    )

    pred = bgnn.predict(
        graph,
        encoded_X,
        test_mask,        
    )

    l = y['class'][test_mask]
    if device == "cuda:0":
        p = pred.cpu().detach().numpy()
    else:
        p = pred.detach().numpy()
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