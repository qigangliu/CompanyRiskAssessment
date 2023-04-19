
import numpy as np
import scipy.sparse as sp
import torch
import json
import pandas as pd

def load_json(data_path):
    f = open(data_path,'rb')
    data = json.load(f)
    f.close()
    return data

def encode_onehot(labels_temp):
    classes = ['low','high']
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels_temp)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data():      
    """Load company dataset"""
    feature_company = pd.read_csv("../../data/GCN&RGCN/company_feature.csv")
    feature_person = pd.read_csv("../../data/GCN&RGCN/person_feature.csv")
    path_relation = "../../data/GCN&RGCN/relation_list_reindex.json"
    edges = np.array(load_json(path_relation))
    shape_adj = len(feature_company) + len(feature_person)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(shape_adj,shape_adj), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # construct label and features
    labels = encode_onehot(list(feature_company['label']))
    total_len = len(labels)
    labels = torch.LongTensor(np.where(labels)[1])
    del feature_company['label']
    c_id_feature = np.array(feature_company.values,dtype=np.float64)
    p_id_feature = np.zeros((len(feature_person),len(c_id_feature[0])))
    features =sp.csr_matrix(np.concatenate([c_id_feature,p_id_feature],axis=0))

    # split the dataset
    idx_all = list(range(len(feature_company)))
    np.random.shuffle(idx_all)
    idx_train = idx_all[:int(total_len/10*7)]
    idx_val = idx_all[int(total_len/10*4):int(total_len/10*7)]
    idx_test = idx_all[int(total_len/10*7):]    

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)