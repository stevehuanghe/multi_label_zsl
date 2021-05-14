import os
import torch
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from models.loss_func import TripletSigmoidRank
from utils.pytorch_misc import to_device, optimistic_restore, print_param, sample_negative_labels


from utils.engine import Engine
from models.mlzsl import MLRGCNPosVAE


class Main(Engine):
    def setup_model(self):
        logger = self.log_master.get_logger("setup")
        logger.info("Setting up model...")

        if self.args.loss == "bce":
            self.criterion = nn.MultiLabelSoftMarginLoss()
        else:
            self.criterion = TripletSigmoidRank()

        self.imgnet_loss = nn.MSELoss()

        self.model = MLRGCNPosVAE(self.args.word_dim, self.args.d_dim, self.args.h_dim, self.args.backbone,
                                  self.args.fin_layers, self.args.fout_layers, self.args.frel_layers, self.args.t_max,
                                  self.args.gcn_layers, self.args.acti, self.args.use_attn,
                                  enc_layers=self.args.enc_layers, dec_layers=self.args.dec_layers,
                                  tune_pos=self.args.tune_pos, pos_fuse=self.args.pos_fuse,
                                  pos_bias=self.args.pos_bias, gnn=self.args.model, topK=self.args.topK,
                                  self_cyc=self.args.self_cyc, normalize=self.args.normalize)

    @staticmethod
    def loss_VAE(X, Xp, mu, log_sigma, w_KL=1.0):
        reconstruct_loss = 0.5 * torch.sum(torch.pow(X - Xp, 2), 1)
        KL_divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1 - log_sigma, 1)
        return torch.mean(reconstruct_loss + w_KL * KL_divergence)

    def train_batch(self, data, step):
        images, labels = data
        result = self.model(images, self.train_cats_embed, self.edges_train, n_coco=self.n_cats_seen,
                            imgnet_idx=self.imgnet_idx)
        logits = result["logits"]
        # attn = result["attention"]
        VAE_data = result["VAE"]

        loss_terms = {}

        if self.args.loss == 'bce':
            loss_terms["loss_coco"] = self.criterion(logits[:, :self.n_cats_train], labels)
        else:
            ng_labels = sample_negative_labels(labels, self.args.n_neg)
            loss_terms["loss_coco"] = self.criterion(logits[:, :self.n_cats_train], labels, ng_labels)

        loss_terms["loss_VAE"] = self.args.wVAE * self.loss_VAE(*VAE_data)
        loss = sum(loss_terms.values())

        self.optimizer.zero_grad()
        loss.backward()

        if self.args.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()

        res = {}
        for k, v in loss_terms.items():
            res[k] = v.item()

        return res


if __name__ == "__main__":
    experiment = Main(exp_name=None)
    experiment.start()
