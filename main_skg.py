import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch.nn as nn
from utils.pytorch_misc import sample_negative_labels

from models.skg_official import PropModel_Multi
from models.loss_func import TripletSigmoidRank
from utils.engine import Engine


class Main(Engine):
    def setup_model(self):
        if self.args.loss == "bce":
            self.criterion = nn.MultiLabelSoftMarginLoss()
        else:
            self.criterion = TripletSigmoidRank()

        self.model = PropModel_Multi(
            n_nodes=self.n_cats_seen,
            edges=self.edges_train,
            mat_ids=self.mat_ids_train,
            wordvecs=self.all_cats_embed[self.cats_split['seen']].to(self.device),
            feat_dim=2048,
            node_dim=self.args.d_dim,
            n_steps=self.args.t_max,
            dropout=self.args.dropout,
        )

        self.model.to(self.device)

    def set_multilabel(self):
        self.model.set_graph_config(
            n_nodes=self.n_cats_seen,
            edges=self.edges_train,
            mat_ids=self.mat_ids_train,
            wordvecs=self.all_cats_embed[self.cats_split['seen']],
        )

    def set_zeroshot(self):
        self.model.set_graph_config(
            n_nodes=self.n_cats_all,
            edges=self.edges_test,
            mat_ids=self.mat_ids_test,
            wordvecs=self.all_cats_embed,
        )

    @staticmethod
    def weighted_bce_loss(logits, targets):
        loss = 0
        n_steps = logits.size(-1)
        for i in range(n_steps):
            loss += (1 / (n_steps - i)) * nn.functional.binary_cross_entropy_with_logits(logits[:, :, i], targets)

        return loss

    def train_batch(self, data, step):
        images, labels = data
        logits = self.model(images)[:,:,-1]

        loss_terms = {}
        if self.args.loss == "skg":
            loss_terms["loss_data"] = self.weighted_bce_loss(logits, labels.float())
        elif self.args.loss == 'bce':
            loss_terms["loss_coco"] = self.criterion(logits, labels)
        else:
            ng_labels = sample_negative_labels(labels, self.args.n_neg)
            loss_terms["loss_coco"] = self.criterion(logits, labels, ng_labels)

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

    def eval_batch(self, data):
        logits = self.model(data[0])[:, :, -1]
        return logits

    def pre_eval_hook(self, *args, **kwargs):
        self.set_zeroshot()

    def pos_eval_hook(self, *args, **kwargs):
        self.set_multilabel()

    # def pos_train_hook(self, *args, **kwargs):
    #     self.eval_on_train()

if __name__ == "__main__":
    experiment = Main(exp_name=None)
    experiment.start()
