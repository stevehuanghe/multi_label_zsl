from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidMLBCE(nn.Module):
    def __init__(self, label_embed):
        super(SigmoidMLBCE, self).__init__()
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.num_class, self.embed_dim = label_embed.size()
        self.embeddings_t = torch.t(label_embed)

    def forward(self, inputs, labels):
        logits = torch.matmul(inputs, self.embeddings_t)
        loss = self.criterion(logits, labels)
        return loss     


class TripletSigmoidRank(nn.Module):
    def __init__(self):
        super(TripletSigmoidRank, self).__init__()

    def forward(self, logits, gt_labels, ng_labels):
        batch = logits.size(0)

        pos_logits = logits * gt_labels.float()
        neg_logits = logits * ng_labels.float()

        gt_mask = pos_logits != 0.0
        ng_mask = neg_logits != 0.0

        mask = torch.bmm(ng_mask.unsqueeze(2).float(), gt_mask.unsqueeze(1).float())   # [batch, n_class, n_class]

        loss = mask * torch.log(1 + torch.exp(neg_logits.unsqueeze(2) - pos_logits.unsqueeze(1)))  # [batch, n_class, n_class]

        loss = loss.view(batch, -1)  # [batch, n_class*n_class]

        loss = loss.mean(1)  # [batch, 1]

        return loss.mean()



