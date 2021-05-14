import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import torch.nn as nn
from utils.pytorch_misc import to_device, optimistic_restore, print_param, sample_negative_labels

from models.fast_zero_tag import Fast0Tag
from models.loss_func import TripletSigmoidRank
from utils.engine import Engine

class Main(Engine):
    def setup_model(self):
        if self.args.loss == "bce":
            self.criterion = nn.MultiLabelSoftMarginLoss()
        else:
            self.criterion = TripletSigmoidRank()

        self.model = Fast0Tag(backbone_name=self.args.backbone, out_dim=self.args.word_dim, use_dict=True)

        self.model.to(self.device)

    def train_batch(self, data, step):
        images, labels = data
        result = self.model(images, self.train_cats_embed)

        logits = result["logits"]

        loss_terms = {}
        if self.args.loss == 'bce':
            loss_terms["loss_data"] = self.criterion(logits[:, :self.n_cats_train], labels)
        else:
            ng_labels = sample_negative_labels(labels, self.args.n_neg)
            loss_terms["loss_data"] = self.criterion(logits[:, :self.n_cats_train], labels, ng_labels)

        loss = sum(loss_terms.values())

        self.optimizer.zero_grad()
        loss.backward()

        if self.args.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()

        res = {}
        for k,v in loss_terms.items():
            res[k] = v.item()

        return res

    # def pos_train_hook(self, *args, **kwargs):
    #     self.eval_on_train()


if __name__ == "__main__":
    experiment = Main(exp_name=None)
    experiment.start()
