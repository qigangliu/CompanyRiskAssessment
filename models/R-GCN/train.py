import time
import json
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from models import RelationalGraphConvModel
from utils import accuracy, get_splits,EarlyStopping
from params import args

from sklearn.metrics import *
from loss import CELoss

warnings.filterwarnings("ignore")


class Train:
    def __init__(self, args):
        self.args = args
        self.best_val = 0

        # Load data
        self.A, self.y, self.train_idx, self.test_idx,self.features= self.load_data()
        self.X = self.features
        self.num_nodes = len(self.features[0])
        self.num_rel = len(self.A)    
        self.labels = torch.LongTensor(np.array(np.argmax(self.y, axis=-1)).squeeze())

        # Get dataset splits
        (
            self.y_train,
            self.y_val,
            self.y_test,
            self.idx_train,
            self.idx_val,
            self.idx_test,
        ) = get_splits(self.y, self.train_idx, self.test_idx, self.args.validation)

        # Create Model
        self.model = RelationalGraphConvModel(
            input_size=self.num_nodes,
            hidden_size=self.args.hidden,
            output_size=self.y_train.shape[1],
            num_bases=self.args.bases,
            num_rel=self.num_rel,
            num_layer=2,
            dropout=self.args.drop,
            featureless=False,
            cuda=self.args.using_cuda,
        )
        print(
            "Loaded %s dataset with %d entities, %d relations and %d classes"
            % (self.args.data, len(self.features), self.num_rel, self.y_train.shape[1])
        )

        # Loss and optimizer
        self.criterion = CELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2
        )

        # initialize the early_stopping object
        if self.args.validation:
            self.early_stopping = EarlyStopping(patience=10, verbose=True)

        if self.args.using_cuda:
            print("Using the GPU")
            self.model.cuda()
            self.labels = self.labels.cuda()

    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx    


    def encode_onehot(self,labels_temp):
        classes = ['low','high']
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels_temp)),
                                dtype=np.int32)
        return labels_onehot


    def load_json(self,data_path):
        f = open(data_path,'rb')
        data = json.load(f)
        return data

    def load_data(self):
        '''Load company dataset'''
        # 设定方案
        feature_company = pd.read_csv("../../data/GCN&RGCN/company_feature.csv")
        feature_person = pd.read_csv("../../data/GCN&RGCN/person_feature.csv")
        path_relation = "../../data/GCN&RGCN/relation_list_reindex_rgcn.json"

        adj = [[] for _ in range(6)]                                                                                                                                                                                             
        shape_adj = len(feature_company) + len(feature_person)
        edges_list = self.load_json(path_relation)
        
        for i in range(len(adj)):
            edges = np.array(edges_list[i]) 
            adj[i] = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(shape_adj, shape_adj))
            adj[i] = adj[i] + adj[i].T.multiply(adj[i].T > adj[i]) - adj[i].multiply(adj[i].T > adj[i])
            adj[i] = self.normalize((adj[i] + sp.eye(adj[i].shape[0])))
            adj[i] = sp.csr_matrix(adj[i])

        # construct label and feautres
        labels = self.encode_onehot(list(feature_company['label']))
        labels = torch.LongTensor(labels)
        total_len = len(labels)
        del feature_company['label']
        c_id_feature = np.array(feature_company.values,dtype=np.float64)
        p_id_feature = np.zeros((len(feature_person),len(c_id_feature[0])))
        features =torch.FloatTensor(np.concatenate([c_id_feature,p_id_feature],axis=0))

        # split the dataset
        idx_all = list(range(len(feature_company)))
        np.random.shuffle(idx_all)
        idx_train = idx_all[:int(total_len/10*7)]
        idx_val = idx_all[int(total_len/10*4):int(total_len/10*7)]
        idx_test = idx_all[int(total_len/10*7):]    
        
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        data = {
                    "A": adj,
                    "y": labels,
                    "train_idx": idx_train,
                    "test_idx": idx_test,
                    "features": features
                }
        return data["A"], data["y"], data["train_idx"], data["test_idx"],data['features']

    def train(self, epoch):
        t = time.time()

        # Start training
        self.model.train()
        emb_train = self.model(A=self.A, X=self.X)
        loss = self.criterion(emb_train[self.idx_train], self.labels[self.idx_train])
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(
            "Epoch: {epoch}, Training Loss on {num} training data: {loss}".format(
                epoch=epoch, num=len(self.idx_train), loss=str(loss.item())
            )
        )

        if self.args.validation:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            with torch.no_grad():
                self.model.eval()
                emb_valid = self.model(A=self.A, X=None)
                loss_val = self.criterion(
                    emb_valid[self.idx_val], self.labels[self.idx_val]
                )
                acc_val = accuracy(emb_valid[self.idx_val], self.labels[self.idx_val])
                if acc_val >= self.best_val:
                    self.best_val = acc_val
                    self.model_state = {
                        "state_dict": self.model.state_dict(),
                        "best_val": acc_val,
                        "best_epoch": epoch,
                        "optimizer": self.optimizer.state_dict(),
                    }
                print(
                    "loss_val: {:.4f}".format(loss_val.item()),
                    "acc_val: {:.4f}".format(acc_val.item()),
                    "time: {:.4f}s".format(time.time() - t),
                )
                print("\n")

                self.early_stopping(loss_val, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    self.model_state = {
                        "state_dict": self.model.state_dict(),
                        "best_val": acc_val,
                        "best_epoch": epoch,
                        "optimizer": self.optimizer.state_dict(),
                    }
                    return False
        return True

    def test(self):
        with torch.no_grad():
            self.model.eval()
            emb_test = self.model(A=self.A, X=self.X)
            preds = emb_test.max(1)[1].type_as(self.labels)

            if self.args.using_cuda:
                l = self.labels[self.idx_test].cpu().detach().numpy()
                p = preds[self.idx_test].cpu().detach().numpy()
            else:
                l = self.labels[self.idx_test].detach().numpy()
                p = preds[self.idx_test].detach().numpy()

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

    def save_checkpoint(self, filename="./.checkpoints/" + args.name):
        print("Save model...")
        if not os.path.exists(".checkpoints"):
            os.makedirs(".checkpoints")
        torch.save(self.model_state, filename)
        print("Successfully saved model\n...")

    def load_checkpoint(self, filename="./.checkpoints/" + args.name, ts="teacher"):
        print("Load model...")
        load_state = torch.load(filename)
        self.model.load_state_dict(load_state["state_dict"])
        self.optimizer.load_state_dict(load_state["optimizer"])
        print("Successfully Loaded model\n...")
        print("Best Epoch:", load_state["best_epoch"])
        print("Best acc_val:", load_state["best_val"].item())


if __name__ == "__main__":
    train = Train(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.using_cuda:
        torch.cuda.manual_seed(args.seed)
    for epoch in range(args.epochs):
        if train.train(epoch) is False:
            break
    if args.validation:
        train.save_checkpoint()
        train.load_checkpoint()
    train.test()


