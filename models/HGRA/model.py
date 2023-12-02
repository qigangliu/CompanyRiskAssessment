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
        self.sqrt_dk = math.sqrt(self.d_k) # 开方
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types): # 每种节点设置一个转换矩阵
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim)) # 最后的全连接
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads) # 初始化全1，每种关系独立
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k) # 映射，每种关系、多头
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k) # 每种关系一个矩阵，多头，消息
        )
        self.skip = nn.Parameter(torch.ones(self.num_types)) # 初始化为全1，每种节点一个激活值
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att) # 初始化参数值
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h): # h 节点特征
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes: # 取出图中的关系 company BRANCH company
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]] # 取出对应类型的映射矩阵，头节点类型
                v_linear = self.v_linears[node_dict[srctype]] # 第二种转换
                q_linear = self.q_linears[node_dict[dsttype]] # 尾节点类型
                # 与节点类型有关的转换，全连接
                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k) # 对头节点做映射，转换为多个单头，对接后面的操作
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k) # 对头节点做另一种映射
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k) # 对尾节点做映射，并变换维度，根据多头的设置
                print(k.size(),'k.size') # [17975, 4, 32]

                e_id = self.edge_dict[etype] # 取出边类型
                # 多头注意力参数，每种边独立
                relation_att = self.relation_att[e_id] # 取出类型对应的参数矩阵 维度：n_heads, d_k, d_k  （d_k单头维度）
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]
                print(relation_att.size(),'relation_att') # [4, 32, 32]
                # 应用多头注意力参数转换
                k = torch.einsum("bij,ijk->bik", k, relation_att) # 用一个d_k*d_k的参数矩阵 对头结做特征变换，变换之后维度不变：sample_num,n_heads,d_k
                v = torch.einsum("bij,ijk->bik", v, relation_msg) # 用另一个d_k*d_k的参数矩阵 对头结点做特征变换
                # 注意力转换结果
                sub_graph.srcdata["k"] = k # 头结点向量
                sub_graph.dstdata["q"] = q # 尾结点向量
                sub_graph.srcdata["v_%d" % e_id] = v # 头结点的另一种表示，与边建立关联  [17975, 4, 32]

                # 转换之后计算相关性
                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t")) # 头尾相关性计算，结果放到边t中，维度为 [1656, 4, 1]
                print(relation_pri,'relation_pri') # 维度 (4,)
                attn_score = ( # relation_pri：与特定关系对应的转换向量，sqrt_dk：特征缩放
                    sub_graph.edata.pop("t").squeeze(-1) * relation_pri / self.sqrt_dk # * relation_pri的作用是特征变换/ self.sqrt_dk 按多头进行缩放？ squeeze
                ) # sum(-1)的作用？
                # 注意力值 归一化
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst") # 计算注意力 按入边计算注意力权重，attn_score：应该是边上数据
                sub_graph.edata["t"] = attn_score.unsqueeze(-1) # 权重值

            # 在每种关系上基于权重 进行信息聚合
            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"), # t是权重值，v_%d是特征表示，获取环境向量，放入消息m
                        fn.sum("m", "t"), # 聚合方式：相加
                    )
                    for etype, e_id in edge_dict.items() # 每种关系分开聚合信息
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes: # 不同类型的节点单独处理

                n_id = node_dict[ntype] # n_id：节点类型
                alpha = torch.sigmoid(self.skip[n_id]) # 残差连接，每种类型产生一个激活值
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim) # 取出使用注意力后的特征向量 [17975, 128] out_dim 128
                trans_out = self.drop(self.a_linears[n_id](t)) # 全连接
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha) # h前一步学习的特征，(1 - alpha) 保留部分原理的特征
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out) # 不同类型的节点使用不同的规范化
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
        for t in range(len(node_dict)): # 每种类型的节点构建一个映射层
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
        self.out = nn.Linear(n_hid, n_out) # 输出层
        self.drop = nn.Dropout(0.1)

    def forward(self, G, out_key): # out_key：paper 返回哪种类型的节点
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype] # 节点类型编号
            h[ntype] = F.gelu(self.drop(self.adapt_ws[n_id](G.nodes[ntype].data["inp"]))) # 提取对应类型的节点特征数据，进行转换 h[ntype] [17975, 128]
            # h[ntype] = F.gelu(G.nodes[ntype].data["inp"])  # 提取对应类型的节点特征数据，进行转换
        for i in range(self.n_layers):
            h = self.gcs[i](G, h) # 使用注意力层
        return F.softmax(self.out(h[out_key]),dim=-1),h[out_key] # 产生分类结果，返回节点的隐藏层表示

