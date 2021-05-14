from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim=2048, out_dim=300, layers='600', activation="leaky", alpha=0.1, dropout=0.0):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        layers = layers.split()
        fcn_layers = []

        if activation == "leaky":
            activation = nn.LeakyReLU(alpha)
        elif activation == "sigmoid":
            activation = nn.Sigmoid()
        elif activation == "tanh":
            activation = nn.Tanh()
        else:
            activation = nn.ReLU()

        for i in range(len(layers)):
            pre_hidden = int(layers[i - 1])
            num_hidden = int(layers[i])
            if i == 0:
                fcn_layers.append(nn.Linear(in_dim, num_hidden))
                fcn_layers.append(nn.Dropout(dropout))
                fcn_layers.append(activation)
            else:
                fcn_layers.append(nn.Linear(pre_hidden, num_hidden))
                fcn_layers.append(nn.Dropout(dropout))
                fcn_layers.append(activation)

            if i == len(layers) - 1:
                fcn_layers.append(nn.Linear(num_hidden, out_dim))

        if len(layers) == 0:
            self.FCN = nn.Linear(in_dim, out_dim)
            torch.nn.init.xavier_uniform_(self.FCN.weight)
        else:
            for net in fcn_layers:
                if isinstance(net, nn.Linear):
                    torch.nn.init.xavier_uniform_(net.weight)
            self.FCN = nn.Sequential(*fcn_layers)



    def forward(self, inputs):
        return self.FCN(inputs)


