import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np


class RelationalGraphConvLayer(Module):
    def __init__(
        self, input_size, output_size, num_bases, num_rel, bias=False, cuda=False
    ):
        super(RelationalGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.cuda = cuda

        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(
                torch.FloatTensor(self.num_bases, self.input_size, self.output_size)
            )
            self.w_rel = Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        else:
            self.w = Parameter(
                torch.FloatTensor(self.num_rel, self.input_size, self.output_size)
            )
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.w.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)

    def forward(self, A, X):
        X = X.cuda() if X is not None and self.cuda else X
        self.w = (
            torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
            if self.num_bases > 0
            else self.w
        )
        weights = self.w.view(
            self.w.shape[0] * self.w.shape[1], self.w.shape[2]
        )  # shape(r*input_size, output_size)
        # Each relations * Weight
        supports = []
        for i in range(self.num_rel):
            if X is not None:
                supports.append(torch.sparse.mm(csr2tensor(A[i], self.cuda), X))
            else:
                supports.append(csr2tensor(A[i], self.cuda))

        tmp = torch.cat(supports, dim=1)
        out = torch.mm(tmp.float(), weights)  # shape(#node, output_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out


def csr2tensor(A, cuda):
    coo = A.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    if cuda:
        out = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()
    else:
        out = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return out
