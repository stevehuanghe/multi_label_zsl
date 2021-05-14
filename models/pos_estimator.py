import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
from models.util import *


class PosVAE(nn.Module):
    def __init__(self, word_dim, out_dim=256, enc_layers="256", dec_layers="256", tune_pos=False, fuse="add", bias=True):
        super(PosVAE, self).__init__()
        self.W = nn.Parameter(torch.rand([word_dim]), requires_grad=tune_pos)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.rand([word_dim]), requires_grad=tune_pos)
        self.fuse = fuse
        if fuse == "cat":
            p_dim = word_dim
        else:
            p_dim = 0

        input_dim = word_dim + p_dim

        enc_layers = enc_layers.split()
        encoder = []
        for i in range(len(enc_layers)):
            num_hidden = int(enc_layers[i])
            pre_hidden = int(enc_layers[i - 1])
            if i == 0:
                encoder.append(nn.Linear(input_dim, num_hidden))
                encoder.append(nn.ReLU())
            else:
                encoder.append(nn.Dropout(p=0.3))
                encoder.append(nn.Linear(pre_hidden, num_hidden))
                encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        last_hidden = int(enc_layers[-1])
        self.mu_net = nn.Sequential(
            nn.Linear(last_hidden, out_dim)
        )

        self.sig_net = nn.Sequential(
            nn.Linear(last_hidden, out_dim)
        )

        dec_layers = dec_layers.split()
        decoder = []
        for i in range(len(dec_layers)):
            num_hidden = int(dec_layers[i])
            pre_hidden = int(dec_layers[i - 1])
            if i == 0:
                decoder.append(nn.Linear(out_dim + word_dim, num_hidden))
                decoder.append(nn.ReLU())
            else:
                decoder.append(nn.Linear(pre_hidden, num_hidden))
                decoder.append(nn.ReLU())
            if i == len(dec_layers) - 1:
                decoder.append(nn.Linear(num_hidden, word_dim))

        self.decoder = nn.Sequential(*decoder)

    def forward(self, probs, embed):
        batch, n_class = probs.size()
        device = probs.device
        probs = probs.view(-1)

        logits = torch.ger(probs, self.W)  # [batch*class, word_dim]

        if self.bias:
            logits = logits + self.b.view(1, -1)

        embed_stack = embed.repeat(batch, 1)  # [batch*class, word_dim]
        if self.fuse == "cat":
            # embed_stack = embed.repeat(batch, 1)  # [batch*class, word_dim]
            inputs = torch.cat([logits, embed_stack])  # [batch*class, 2*word_dim]
        elif self.fuse == "mul":
            inputs = logits.view(batch, n_class, -1) * embed.unsqueeze(0)
        else:
            inputs = logits.view(batch, n_class, -1) + embed.unsqueeze(0)

        inputs = inputs.view(batch*n_class, -1)
        hidden = self.encoder(inputs)
        mu = self.mu_net(hidden)
        log_sigma = self.sig_net(hidden)
        eps = torch.rand(mu.size())
        eps = eps.to(device)
        Z = mu + torch.exp(log_sigma / 2) * eps

        ZS = torch.cat([embed_stack, Z], dim=1)
        Xp = self.decoder(ZS)
        X = logits
        return X, Xp, mu, log_sigma, Z

    def loss_VAE(self, X, Xp, mu, log_sigma, w_KL=1.0):
        reconstruct_loss = 0.5 * torch.sum(torch.pow(X-Xp, 2), 1)
        KL_divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1 - log_sigma, 1)
        return torch.mean(reconstruct_loss + w_KL * KL_divergence)
