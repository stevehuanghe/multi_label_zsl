import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable, Function
from torch.nn.modules.rnn import RNNCellBase
import numpy as np
from models.torchvision_utils import load_resnet

class Relnet(nn.Module):
    def __init__(self, emb_dim, node_dim):
        super(Relnet, self).__init__()

        self.mat = nn.Parameter(torch.randn(emb_dim, emb_dim * (node_dim ** 2)))
        self.node_dim = node_dim
        self.emb_dim = emb_dim


    def forward(self, x, y):
        tmp = torch.mm(x, self.mat).view(-1, self.emb_dim, self.node_dim ** 2)
        res = torch.bmm(y.unsqueeze(1), tmp)
        # res = nn.functional.tanh(res)
        return res


class Inputnet(nn.Module):
    def __init__(self, wv_dim, feat_dim, node_dim, dropout=0.5):
        super(Inputnet, self).__init__()

        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim + wv_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, node_dim),
        )


    def forward(self, x, wv):
        # x: (batch, feat_dim)
        # wv: (n_nodes, wv_dim)
        x = x.unsqueeze(1).expand(x.size(0), wv.size(0), self.feat_dim)
        wv = wv.unsqueeze(0).expand(x.size(0), wv.size(0), wv.size(1))

        f = torch.cat([x, wv], 2)
        f = f.view(-1, wv.size(-1) + self.feat_dim)

        return self.net(f).view(x.size(0), -1)


class PropMat(Function):
    def __init__(self):
        super(PropMat, self).__init__()

    @staticmethod
    def forward(ctx, u_ids, v_ids, weights, zero_mat):
        ctx.save_for_backward(u_ids, v_ids)
        my_zero_mat = zero_mat.clone()

        my_zero_mat[u_ids, v_ids] = weights

        return my_zero_mat

    @staticmethod
    def backward(ctx, grad_mat):
        u_ids, v_ids = ctx.saved_tensors

        return None, None, grad_mat[u_ids.data, v_ids.data], None


class FCNet(nn.Module):
    def __init__(self, config, dropout=0., output_act=None):
        assert(len(config) > 1)
        super(FCNet, self).__init__()

        n_hidden = len(config) - 2

        self.net = nn.Sequential()
        self.net.add_module('Linear_0', nn.Linear(config[0], config[1]))
        for i in range(1, n_hidden + 1):
            self.net.add_module(f'ReLU_{i - 1}', nn.ReLU())
            self.net.add_module(f'Dropout_{i}', nn.Dropout(dropout))
            self.net.add_module(f'Linear_{i}', nn.Linear(config[i], config[i + 1]))

        if output_act is not None:
            self.net.add_module(f'{output_act}_{n_hidden + 1}', 
                                getattr(nn, output_act)())
    

    def forward(self, x):
        return self.net(x)


class PropNet(nn.Module):
    def __init__(self, node_hidden, n_steps, dropout=0.):
        super().__init__()

        self.node_hidden = node_hidden
        self.n_steps = n_steps
        self.cell = nn.GRUCell(input_size=node_hidden, hidden_size=node_hidden)


    def forward(self, hidden, prop_mat):
        hiddens = [hidden.unsqueeze(1)]

        for step in range(self.n_steps):
            x = hidden @ prop_mat.t()
            x = torch.tanh(x)
            hidden = self.cell(
                x.view(-1, self.node_hidden),
                hidden.view(-1, self.node_hidden)
            ).view(x.size(0), -1)

            hiddens.append(hidden.unsqueeze(1))
                
        return torch.cat(hiddens, 1)


class PropModel_Multi(nn.Module):
    def __init__(self, n_nodes, edges, mat_ids, wordvecs,
                 feat_dim, node_dim, n_steps, dropout):
        super().__init__()

        self.feat_dim = feat_dim
        self.node_dim = node_dim
        self.set_graph_config(n_nodes, edges, mat_ids, wordvecs)

        self.inputnet = Inputnet(self.wordvecs.size(1), feat_dim, self.node_dim, dropout) 
        self.embnet = FCNet([self.wordvecs.size(1), 50])
        self.relnets = nn.ModuleList([Relnet(50, node_dim) for _ in range(self.edge_types)])
        self.propnet = PropNet(node_dim, n_steps, dropout=0.)
        self.outnet = FCNet([self.node_dim, 20, 1], dropout=0., output_act=None)
        self.resnet, _, _ = load_resnet("resnet101")

    def set_graph_config(self, n_nodes, edges, mat_ids, wordvecs):
        self.n_nodes = n_nodes
        self.edges = edges
        self.edge_types = len(edges)
        self.mat_ids = mat_ids
        self.wordvecs = wordvecs

        mat = torch.zeros(n_nodes * self.node_dim, n_nodes * self.node_dim)
        self.zero_prop_mat_t = mat.requires_grad_(True)
        self.zero_prop_mat_e = mat.requires_grad_(False)
        if self.wordvecs.is_cuda:
            self.zero_prop_mat_t = self.zero_prop_mat_t.cuda()
            self.zero_prop_mat_e = self.zero_prop_mat_e.cuda()


    def get_prop_mat(self):
        weights = []
        all_embs = self.embnet(self.wordvecs)

        for edge_type, (u_ids, v_ids) in enumerate(self.edges):
            u_embs = all_embs[u_ids]
            v_embs = all_embs[v_ids]
            edge_w = self.relnets[edge_type](u_embs, v_embs)
            weights.append(edge_w)

        weights = torch.cat(weights, 0).squeeze().view(-1)
        prop_mat = PropMat.apply
        if self.training:
            return prop_mat(self.mat_ids[0], self.mat_ids[1], weights, self.zero_prop_mat_t)
        else:
            return prop_mat(self.mat_ids[0], self.mat_ids[1], weights, self.zero_prop_mat_e)

    def forward(self, inputs):
        inputs = self.resnet(inputs).view(-1, self.feat_dim)  # (batch, feat_dim)

        init_states = self.inputnet(inputs, self.wordvecs)

        prop_mat = self.get_prop_mat()
        prop_mat_norm = prop_mat.norm(p=2, dim=1)
        prop_mat_norm = prop_mat_norm + (prop_mat_norm == 0).float()
        prop_mat = prop_mat / prop_mat_norm

        # print(prop_mat.shape)
        # import ipdb
        # ipdb.set_trace()

        states = self.propnet(init_states, prop_mat) # states: (batch, n_steps, node_dim)

        new_logits = self.outnet(states.view(-1, self.node_dim)).view(-1, states.size(1), self.n_nodes)
        new_logits = new_logits.transpose(1, 2).contiguous()

        return new_logits

