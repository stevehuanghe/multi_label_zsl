import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn

from .torchvision_utils import initialize_backbone

class Fast0Tag(nn.Module):

    def __init__(self, out_dim=300, backbone_name="resnet101", use_dict=False):
        super(Fast0Tag, self).__init__()
        self.out_dim = out_dim
        self.use_dict = use_dict
        self.backbone, _, feat_dim = initialize_backbone(backbone_name)

        self.fcn = nn.Sequential(
            nn.Linear(feat_dim, 8096),
            nn.ReLU(),
            nn.Linear(8096, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, out_dim)
        )

    def forward(self, images, embed, *args, **kwargs):
        feats = self.backbone(images)
        out_feats = self.fcn(feats)
        logits = torch.matmul(out_feats, embed.t())
        if self.use_dict:
            return {
                "logits": logits,
                "feats": out_feats
            }
        return out_feats, logits

