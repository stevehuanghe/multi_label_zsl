import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import torch.nn as nn
from models.mlzsl import ZSGGNN
from models.loss_func import TripletSigmoidRank
from utils.pytorch_misc import to_device, optimistic_restore, print_param, sample_negative_labels


from utils.engine import Engine

class Main(Engine):
    def setup_model(self):
        if self.args.loss == "bce":
            self.criterion = nn.MultiLabelSoftMarginLoss()
        else:
            self.criterion = TripletSigmoidRank()
        self.imgnet_loss = nn.MSELoss()

        self.model = ZSGGNN(self.args.word_dim, self.args.d_dim, self.args.h_dim, self.args.backbone,
                           self.args.fin_layers, self.args.fout_layers, self.args.frel_layers, self.args.t_max,
                           self.args.gcn_layers, self.args.acti, self.args.use_attn)

        self.model.to(self.device)

    def train_batch(self, data, step):
        images, labels = data
        result = self.model(images, self.train_cats_embed, self.edges_train)
        logits = result["logits"]
        attn = result["attention"]

        loss_terms = {}
        if self.args.loss == 'bce':
            loss_terms["loss_coco"] = self.criterion(logits[:, :self.n_cats_train], labels)
        else:
            ng_labels = sample_negative_labels(labels, self.args.n_neg)
            loss_terms["loss_coco"] = self.criterion(logits[:, :self.n_cats_train], labels, ng_labels)

        if self.args.use_imgnet:
            imgnet_scores = torch.softmax(result["imgnet_scores"][:, self.imgnet_idx], dim=1)
            imgnet_logits = torch.softmax(logits[:, self.n_cats_seen:], dim=1)
            loss_terms["loss_imgnet"] = self.args.gamma * self.imgnet_loss(imgnet_logits, imgnet_scores)

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


if __name__ == "__main__":
    experiment = Main(exp_name=None)
    experiment.start()
