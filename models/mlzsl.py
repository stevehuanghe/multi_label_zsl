import math
import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torchvision_utils import initialize_backbone, load_resnet
from models.mlp import MLP
from models.pos_estimator import PosVAE


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = nn.Linear(in_features, out_features, bias=bias)
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLRGCNPosVAE(nn.Module):
    def __init__(self,word_dim=300, d_dim=5, hid_dim=512, backbone="resnet101",
                 F_in_layers="512", F_out_layers="512", F_rel_layers="256", t_max=5, gcn_layers="512 512", acti="relu",
                 use_attn=True, enc_layers="256", dec_layers="256", tune_pos=False, pos_fuse="add", pos_bias=True,
                 topK=-1, gnn="gcn", use_pool=False, self_cyc=False, dropout=0.0, normalize=False):

        super(MLRGCNPosVAE, self).__init__()
        self.word_dim = word_dim
        self.d_dim = d_dim
        self.hid_dim = hid_dim
        self.t_max = t_max
        self.topK = topK
        self.use_pool = use_pool
        self.feat_net, self.imgnet_clf, self.feat_dim = load_resnet(backbone, fmap_tensor=True)
        self.F_in = MLP(self.feat_dim+word_dim, d_dim, layers=F_in_layers, dropout=dropout)
        self.F_rel = MLP(2*word_dim, 1, layers=F_rel_layers, dropout=dropout)
        self.self_cyc = self_cyc
        self.use_attn = use_attn
        self.normalize = normalize
        if use_attn:
            self.key_net = nn.Linear(self.feat_dim, hid_dim)
            self.query_net = nn.Linear(word_dim, hid_dim)

        self.gnn = gnn
        if gnn == "ggnn":
            self.gru_cell = nn.GRUCell(d_dim, d_dim)
            self.F_out = nn.Linear(d_dim, 1, bias=False)
        else:
            self.gcn = []
            self.gcn_params = []
            layers = gcn_layers.split()
            dim_in = d_dim
            dim_out = d_dim
            for i, layer in enumerate(layers):
                dim_out = int(layer)
                gcn = GraphConvolution(dim_in, dim_out)
                self.gcn.append(gcn)
                self.gcn_params += list(gcn.parameters())
                dim_in = dim_out

            self.gcn = nn.ModuleList(self.gcn)

            self.F_out = MLP(dim_out, 1, layers=F_out_layers, dropout=dropout)

        if acti == "sigmoid":
            self.activation = nn.Sigmoid()
        elif acti == "leaky":
            self.activation = nn.LeakyReLU()
        elif acti == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.pos_VAE = PosVAE(word_dim, d_dim, enc_layers, dec_layers, tune_pos, pos_fuse, pos_bias)

    def forward(self, images, embed, edges, n_unseen=None, n_coco=64, imgnet_idx=None, edge_weights=None):
        batch = images.size(0)
        n_class, word_dim = embed.size()
        feats = self.feat_net(images)
        imgnet_scores = self.imgnet_clf(feats)
        imgnet_scores = F.softmax(imgnet_scores[:,imgnet_idx], dim=1)
        feats = feats.view(batch, -1, self.feat_dim)

        embed_coco = embed[:n_coco]
        embed_imgnet = embed[n_coco:]

        if self.use_attn:
            feats_key = feats.view(-1, self.feat_dim)  # [batch*49, 2048]
            feats_key = self.key_net(feats_key).view(batch, -1, self.hid_dim).unsqueeze(1)  # [batch, 1, 49, hid_dim]

            embed_query = self.query_net(embed_coco).repeat(batch, 1).view(batch, n_coco, self.hid_dim, 1)  # [batch, n_class, hid_dim, 1]

            attn = F.softmax(torch.matmul(feats_key, embed_query), dim=2)  # [batch, n_class, 49, 1]

            graph_input = torch.sum(feats.unsqueeze(1) * attn, dim=2)  # [batch, class, feat_dim]
        else:
            graph_input = torch.mean(feats, dim=1, keepdim=True).repeat(1, n_coco, 1)  # [batch, class, feat_dim]
            attn = torch.zeros([batch, n_class, feats.size(1)])


        graph_input = torch.cat([graph_input, embed_coco.unsqueeze(0).repeat(batch, 1, 1)], dim=2)  # [batch, class, feat_dim+word_dim]
        graph_input = graph_input.view(batch*n_coco, -1)  # [batch*class, feat_dim]

        hc = self.F_in(graph_input).view(batch, n_coco, -1)  # [batch, class_coco, d_dim]

        X, Xp, mu, log_sigma, hp = self.pos_VAE(imgnet_scores, embed_imgnet)  # [batch, class_imgnet, d_dim]
        hp = hp.view(batch, embed_imgnet.size(0), -1)  # [batch, class_coco, d_dim]

        h0 = torch.cat([hc, hp], dim=1)  # [batch, class, d_dim]

        embed_tuple = torch.cat([embed[edges[0]], embed[edges[1]]], dim=1)
        values = self.F_rel(embed_tuple).view(-1)   # [n_edges]
        # values = torch.sigmoid(values)

        prop_mat = torch.sparse_coo_tensor(edges, values, torch.Size([n_class, n_class]), requires_grad=True).to(images.device).to_dense()
        if edge_weights is not None:
            prop_mat = prop_mat * edge_weights

        prop_mat[n_coco:] = 0  # don't let message pass back to imagenet classes

        if self.topK > 0:
            idx = torch.argsort(imgnet_scores, dim=1)[:, :self.topK]
            masks = torch.zeros_like(imgnet_scores).scatter_(1, idx, 1).unsqueeze(1)  # [batch, 1, n_imgnet]
            prop_mat = prop_mat.unsqueeze(0).repeat(batch, 1, 1)
            prop_mat[:, n_coco, n_coco:] *= masks

        prop_mat = torch.softmax(prop_mat, dim=1)

        if self.self_cyc:
            eye = torch.eye(n_class, requires_grad=False).float().to(images.device)
            prop_mat = (1.0 - eye) * prop_mat + eye

        if self.normalize:
            prop_mat = self.normalize_A(prop_mat)

        # if n_unseen is not None:
        #     prop_mat = self.adjust_unseen_mask(prop_mat, n_unseen)

        if self.gnn == "ggnn":
            # prop_mat = torch.tanh(prop_mat)
            hidden = h0.view(-1, self.d_dim)
            for t in range(self.t_max):
                msg = torch.matmul(prop_mat, hidden.view(batch, n_class, self.d_dim)).view(-1, self.d_dim)  # [batch*n_class, hid_dim]
                hidden = self.gru_cell(msg, hidden)
        else:
            hidden = h0
            for i, gcn in enumerate(self.gcn):
                if i < len(self.gcn) - 1:
                    hidden = self.activation(gcn(hidden, prop_mat))
                else:
                    hidden = gcn(hidden, prop_mat)

        logits = self.F_out(hidden).view(batch, n_class)

        return {
            "logits": logits,
            "attention": attn,
            "imgnet_scores": imgnet_scores.detach(),
            "VAE": (X, Xp, mu, log_sigma)
        }

    @staticmethod
    def normalize_A(mx):
        """Row-normalize sparse matrix"""
        rowsum = mx.sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.matmul(r_mat_inv, mx)
        return mx

    @staticmethod
    def adjust_unseen_mask(mask, n_unseen):
        mask[:, n_unseen] = 0.0
        return mask



class ZSGGNN(nn.Module):
    def __init__(self,word_dim=300, d_dim=5, hid_dim=512, backbone="resnet101",
                 F_in_layers="512", F_out_layers="512", F_rel_layers="256", t_max=5,
                 gcn_layers="512 512", acti="relu", use_attn=True):

        super(ZSGGNN, self).__init__()
        self.word_dim = word_dim
        self.d_dim = d_dim
        self.hid_dim = hid_dim
        self.t_max = t_max
        self.use_attn = use_attn
        self.feat_net, self.imgnet_clf, self.feat_dim = load_resnet(backbone, fmap_tensor=True)
        self.F_in = MLP(self.feat_dim, d_dim, layers=F_in_layers)
        self.F_rel = MLP(2*word_dim, 1, layers=F_rel_layers)
        self.F_out = MLP(d_dim, 1, layers=F_out_layers)

        self.key_net = nn.Linear(self.feat_dim, hid_dim)
        self.query_net = nn.Linear(word_dim, hid_dim)
        self.gru_cell = nn.GRUCell(d_dim, d_dim)


    def forward(self, images, embed, edges, *args, **kwargs):
        batch = images.size(0)
        n_class, word_dim = embed.size()
        feats = self.feat_net(images)
        imgnet_scores = F.softmax(self.imgnet_clf(feats), dim=1)
        feats = feats.view(batch, -1, self.feat_dim)

        if self.use_attn:
            feats_key = feats.view(-1, self.feat_dim)  # [batch*49, 2048]
            feats_key = self.key_net(feats_key).view(batch, -1, self.hid_dim).unsqueeze(1)  # [batch, 1, 49, hid_dim]

            embed_query = self.query_net(embed).repeat(batch, 1).view(batch, n_class, self.hid_dim, 1)  # [batch, n_class, hid_dim, 1]

            attn = F.softmax(torch.matmul(feats_key, embed_query), dim=2)  # [batch, n_class, 49, 1]

            graph_input = torch.sum(feats.unsqueeze(1) * attn, dim=2)  # [batch, class, feat_dim]
        else:
            graph_input = torch.mean(feats, dim=1, keepdim=True).repeat(1, n_class, 1)  # [batch, class, feat_dim]
            attn = torch.zeros([batch, n_class, feats.size(1)])

        graph_input = graph_input.view(-1, self.feat_dim)
        h0 = self.F_in(graph_input)  # [batch*class, hid_dim]

        # embed_tuple = torch.cat([embed.repeat(n_class, 1), embed.repeat(1, n_class).view(-1, self.word_dim)], dim=1)
        # prop_mat = self.F_rel(embed_tuple)   # [n_class*n_class, 1]
        # prop_mat = prop_mat * adj
        # prop_mat = prop_mat.view(1, n_class, n_class)
        # prop_mat = torch.tanh(prop_mat) # [1, n_class, n_class]

        embed_tuple = torch.cat([embed[edges[0]], embed[edges[1]]], dim=1)
        values = self.F_rel(embed_tuple).view(-1)   # [n_edges]
        values = torch.tanh(values)

        prop_mat = torch.sparse_coo_tensor(edges, values, torch.Size([n_class, n_class]), requires_grad=True).to(images.device).to_dense()


        hidden = h0
        for t in range(self.t_max):
            msg = torch.matmul(prop_mat, hidden.view(batch, n_class, self.d_dim)).view(-1, self.d_dim)  # [batch*n_class, hid_dim]
            msg = torch.tanh(msg)
            hidden = self.gru_cell(msg, hidden)

        logits = self.F_out(hidden).view(batch, n_class)

        return {
            "logits": logits,
            "attention": attn,
            "imgnet_scores": imgnet_scores
        }




