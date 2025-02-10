import torch.nn.init as init
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv




class MyHANAtt3(nn.Module):
    def __init__(
            self, meta_paths_herb, meta_paths_symptom, meta_paths_attribute,in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(MyHANAtt3, self).__init__()
        self.Embedding_layer = torch.nn.Embedding(1236, hidden_size)
        self.H_HAN_layer = HAN(meta_paths_herb, hidden_size, hidden_size, hidden_size, num_heads, dropout)
        self.H_mlp_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.H_bn_1 = torch.nn.BatchNorm1d(hidden_size)
        self.H_tanh_1 = torch.nn.Tanh()
        self.H_mlp_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.H_bn_2 = torch.nn.BatchNorm1d(hidden_size)
        self.H_tanh_2 = torch.nn.Tanh()
        self.H_mlp_3 = torch.nn.Linear(hidden_size, hidden_size)
        self.H_bn_3 = torch.nn.BatchNorm1d(hidden_size)
        self.H_tanh_3 = torch.nn.Tanh()

        self.S_HAN_layer = HAN(meta_paths_symptom, hidden_size, hidden_size, hidden_size, num_heads, dropout)
        self.S_mlp = torch.nn.Linear(hidden_size, hidden_size)
        self.S_bn_1 = torch.nn.BatchNorm1d(hidden_size)
        self.S_tanh_1 = torch.nn.Tanh()

        self.A_HAN_layer = HAN(meta_paths_attribute, hidden_size, hidden_size, hidden_size, num_heads, dropout)
        self.A_mlp = torch.nn.Linear(hidden_size, hidden_size)
        self.A_bn_1 = torch.nn.BatchNorm1d(hidden_size)
        self.A_tanh_1 = torch.nn.Tanh()

        self.ss_GAN_layer = GAT_SS_HH(hidden_size,hidden_size,num_heads[0],dropout)
        self.hh_GAN_layer = GAT_SS_HH(hidden_size,hidden_size,num_heads[0],dropout)

        self.A_mlp = torch.nn.Linear(hidden_size, hidden_size)
        self.A_bn_1 = torch.nn.BatchNorm1d(hidden_size)
        self.A_tanh_1 = torch.nn.Tanh()

        self.syndnorm_mlp = torch.nn.Linear(hidden_size, hidden_size)
        self.syndnorm_bn = torch.nn.BatchNorm1d(hidden_size)
        self.syndnorm_relu = torch.nn.ReLU()
        self.predict = torch.nn.Linear(811, 811)
        self.A_attention_sc1 = torch.nn.Linear(2*hidden_size, hidden_size)
        self.A_attention_sc2 = torch.nn.Linear(hidden_size, 1)
        init.kaiming_normal_(self.A_attention_sc1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.A_attention_sc2.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.A_attention_sc1.bias, 0)
        init.constant_(self.A_attention_sc2.bias, 0)

        self.S_attention_sc1 = torch.nn.Linear(2*hidden_size, hidden_size)
        self.S_attention_sc2 = torch.nn.Linear(hidden_size, 1)
        init.kaiming_normal_(self.S_attention_sc1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.S_attention_sc2.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.S_attention_sc1.bias, 0)
        init.constant_(self.S_attention_sc2.bias, 0)
        self.dropout = nn.Dropout(p=0.5)

        self.combineLayer = AdaptiveWeightLayerTwoFusion()



    def forward(self, shaGraph,hhGraph,ssGraph,kgOneHot, feature,sids):
        all_features = self.Embedding_layer(feature).squeeze(dim=1)
        herb1 = all_features[390:1201]
        symptom1 = all_features[0:390]
        attribute1 = all_features[1201:]
        herb2 = self.H_HAN_layer(shaGraph, herb1)
        symptom2 = self.S_HAN_layer(shaGraph, symptom1)

        kgOneHot1 = torch.mm(kgOneHot.float(),attribute1)
        herbInput = (herb1+kgOneHot1)/2

        herb3 = self.hh_GAN_layer(hhGraph,herbInput)
        herb3 = torch.mean(herb3,dim=1)
        symptom3 = self.ss_GAN_layer(ssGraph,symptom1)
        symptom3 = torch.mean(symptom3,dim=1)

        herb = (herb1+herb2+herb3)/3
        herb = self.H_mlp_1(herb)
        herb = self.H_bn_1(herb)
        herb = self.H_tanh_1(herb)
        symptom = (symptom1+symptom2+symptom3)/3
        symptom = self.S_mlp(symptom)
        symptom = self.S_bn_1(symptom)
        symptom = self.S_tanh_1(symptom)

        attribute1 = self.A_mlp(attribute1)
        attribute1 = self.A_bn_1(attribute1)
        attribute1 = self.A_tanh_1(attribute1)

        attribute_list = []
        a_List = []
        for i in range(kgOneHot.size(0)):
            row_vector = kgOneHot[i]
            ac = torch.nonzero(row_vector).squeeze()

            a_ac = attribute1[ac]
            if a_ac.dim() == 1:
                a_ac = a_ac.unsqueeze(0)
            tempHerb = herb[i]
            a_h_all = torch.softmax(torch.matmul(a_ac, tempHerb), dim=0)
            a_List.append(a_h_all)
            if a_h_all.dim() == 1:
                a_h_all = a_h_all.unsqueeze(0)
            a_h_all = torch.matmul(a_h_all, a_ac)

            attribute_list.append(a_h_all)
        attri = torch.stack(attribute_list).squeeze(dim=1)

        e_syndrome_list = []
        h_List = []
        for i in range(sids.size(0)):
            row_vector = sids[i]
            indices = torch.nonzero(row_vector).squeeze()

            e_sc = symptom[indices]
            if e_sc.dim() == 1:
                e_sc = e_sc.unsqueeze(0)
            a_h_all = torch.softmax(torch.matmul(e_sc, herb.T), dim=0)

            if a_h_all.dim() == 1:
                a_h_all = a_h_all.unsqueeze(0)
            h_List.append(a_h_all)
            e_syndrome_all = torch.matmul(a_h_all.T, e_sc)
            e_syndrome_list.append(e_syndrome_all)

        e_syndrome = torch.stack(e_syndrome_list)
        e_syndrome = torch.sum(e_syndrome, dim=1)

        herb = (herb+attri)/2
        herb = self.H_mlp_2(herb)
        herb = self.H_bn_2(herb)
        herb = self.H_tanh_2(herb)
        e_synd_norm = self.syndnorm_mlp(e_syndrome)
        e_synd_norm = self.syndnorm_bn(e_synd_norm)
        e_synd_norm = self.syndnorm_relu(e_synd_norm)
        mm = torch.mm(e_synd_norm, herb.t())
        return  mm

class HAN(nn.Module):
    def __init__(
            self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, feature):
        for gnn in self.layers:
            feature = gnn(g, feature)
        return self.predict(feature)

class GAT_SS_HH(nn.Module):
    def __init__(self, in_feats, out_size, num_heads,dropout):
        super(GAT_SS_HH, self).__init__()
        self.gat1 = GATConv(
                    in_feats,
                    out_size,
                    num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )

    def forward(self, g, features):
        features = features.to(torch.float32)
        h = torch.tanh(self.gat1(g, features))
        return h

class AdaptiveWeightLayerTwoFusion(nn.Module):
    def __init__(self):
        super(AdaptiveWeightLayerTwoFusion, self).__init__()
        self.herb_weight = nn.Parameter(torch.tensor([0.9]))
        self.symptom_weight = nn.Parameter(torch.tensor([0.1]))

    def forward(self, herb_vec, symptom_vec):
        combined_vec = self.herb_weight * herb_vec + self.symptom_weight * symptom_vec
        return combined_vec

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, hidden_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    hidden_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=hidden_size * layer_num_heads,
            hidden_size=hidden_size
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, feature):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, feature).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )
        return self.semantic_attention(semantic_embeddings)

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)
