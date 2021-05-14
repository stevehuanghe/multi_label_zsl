import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn

from .torchvision_utils import initialize_backbone

class BaseModel(nn.Module):

    def __init__(self, out_dim, hid_dim=512, backbone_name="resnet152"):
        super(BaseModel, self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim

        self.backbone, _, feat_dim = initialize_backbone(backbone_name)

        self.fcn = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, images, embed):
        feats = self.backbone(images)
        out_feats = self.fcn(feats)
        logits = torch.matmul(out_feats, embed.t())
        return out_feats, logits


