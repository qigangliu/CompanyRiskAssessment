import math

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax

class HGRALayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads,
        dropout=0.1,
        use_norm=False,
    ):
        super(HGRALayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types): # transformation of each type of node
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim)) # FC
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads) 
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k) 
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k) 
        )
        self.skip = nn.Parameter(torch.ones(self.num_types)) 
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att) 
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h): # G: graph structure, h: feature vector of node
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes: # triplet: e.g.{company BRANCH company}
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]  
                v_linear = self.v_linears[node_dict[srctype]]  
                q_linear = self.q_linears[node_dict[dsttype]]  
                 
                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)  
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)  
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)  
                print(k.size(),'k.size') # [17975, 4, 32]

                e_id = self.edge_dict[etype]  
     
                relation_att = self.relation_att[e_id]  
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]
                # print(relation_att.size(),'relation_att') # [4, 32, 32]
            
                k = torch.einsum("bij,ijk->bik", k, relation_att) 
                v = torch.einsum("bij,ijk->bik", v, relation_msg)  
                 
                sub_graph.srcdata["k"] = k  
                sub_graph.dstdata["q"] = q  
                sub_graph.srcdata["v_%d" % e_id] = v # [17975, 4, 32]

                # 转换之后计算相关性
                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t")) #  [1656, 4, 1]
                # print(relation_pri,'relation_pri') #  (4,)
                attn_score = (  
                    sub_graph.edata.pop("t").squeeze(-1) * relation_pri / self.sqrt_dk 
                ) 
                 
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst") 
                sub_graph.edata["t"] = attn_score.unsqueeze(-1) 

            # information aggregation
            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),  
                        fn.sum("m", "t"),  
                    )
                    for etype, e_id in edge_dict.items() # within each neighbor group
                },
                cross_reducer="mean", # cross different group
            )

            new_h = {}
            for ntype in G.ntypes:  

                n_id = node_dict[ntype] 
                alpha = torch.sigmoid(self.skip[n_id]) # residual connection
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)  
                trans_out = self.drop(self.a_linears[n_id](t))  
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)  
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)  
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGRA(nn.Module):
    def __init__(
        self,
        G,
        node_dict,
        edge_dict,
        n_inp,
        n_hid,
        n_out,
        n_layers,
        n_heads,
        use_norm=True,
    ):
        super(HGRA, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):  
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(
                HGRALayer(
                    n_hid, #
                    n_hid,
                    node_dict,
                    edge_dict,
                    n_heads,
                    use_norm=use_norm,
                )
            )
        self.out = nn.Linear(n_hid, n_out)  
        self.drop = nn.Dropout(0.1)

    def forward(self, G, out_key): # out_key：company
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]  
            h[ntype] = F.gelu(self.drop(self.adapt_ws[n_id](G.nodes[ntype].data["inp"])))  # [17975, 128]
        for i in range(self.n_layers):
            h = self.gcs[i](G, h) 
        return F.softmax(self.out(h[out_key]),dim=-1),h[out_key] 

