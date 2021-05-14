import os
import time
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import numpy as np
import mlflow

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.experiments import Experiment
from utils.eval_utils import AveragePrecisionMeter
from utils.data_utils import transform_fn, Vocabulary, batch_map
from utils.mscoco_dataset import CocoDataset
from utils.nuswide_dataset import NUSWideDataset81, NUSWideDataset
from utils.vg_dataset import VGDataset
from utils.pytorch_misc import to_device, optimistic_restore, print_param, sample_negative_labels
from utils.graph_utils import load_graph, get_edges
from utils.word_vectors import get_word_vectors
from models.loss_func import TripletSigmoidRank

from backup2.mlgcn import MLRGCNPOS


class Engine(Experiment):

    def run(self):
        pprint(vars(self.args))
        if torch.cuda.device_count() > 0:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger = self.log_master.get_logger('main')

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)

        if self.args.data == "coco":
            train_loader, test_loader = self.load_data_coco()
        elif self.args.data == "vg":
            train_loader, test_loader = self.load_data_vg()
        elif self.args.data == "nuswide-1k":
            train_loader, test_loader = self.load_data_nuswide(NUSWideDataset)
        else:
            train_loader, test_loader = self.load_data_nuswide(NUSWideDataset81)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.setup_model()
        self.model.to(self.device)
        self.setup_optim()
        self.setup_scheduler()

        epoch = self.load_model()
        logger.info("print params:")
        logger.info(print_param(self.model))

        if torch.cuda.device_count() > 1:
            logger.info("using multiple GPUs")
            self.model = nn.DataParallel(self.model)

        logger.info("Evaluating without training...")
        self.pre_eval_hook()
        self.eval_epoch(test_loader, epoch)
        self.pos_eval_hook()
        logger.info('Start training...')
        best = 0
        while epoch <= self.args.epoch:
            epoch += 1
            self.epoch = epoch
            self.pre_train_hook()
            loss = self.train_epoch(train_loader, epoch)
            self.pos_train_hook()

            self.pre_eval_hook()
            metric = self.eval_epoch(test_loader, epoch)
            self.pos_eval_hook()

            self.scheduler_step(metric, loss, epoch)

            if self.args.save_epoch > 0 and epoch % self.args.save_epoch == 0:
                self.save_model(epoch, metric, loss)

            if metric[-1] > best:
                self.save_model(epoch, metric, loss, best=True)
                best = metric[-1]

            logger.info(f"Best: unseen miAP {best:.2f}")

    def pre_train_hook(self, *args, **kwargs):
        pass

    def pos_train_hook(self, *args, **kwargs):
        pass

    def pre_eval_hook(self, *args, **kwargs):
        pass

    def pos_eval_hook(self, *args, **kwargs):
        pass

    def setup_model(self):
        """
        build model, optimizer and restore checkpoint
        """
        logger = self.log_master.get_logger("setup")
        logger.info("Setting up model")
        if self.args.loss == "bce":
            self.criterion = nn.MultiLabelSoftMarginLoss()
        else:
            self.criterion = TripletSigmoidRank()
        self.imgnet_loss = nn.MSELoss()

        self.model = MLRGCNPOS(self.args.word_dim, self.args.d_dim, self.args.h_dim, self.args.backbone,
                               self.args.fin_layers, self.args.fout_layers, self.args.frel_layers, self.args.t_max,
                               self.args.gcn_layers, self.args.acti, self.args.use_attn,
                               pos_layers=self.args.pos_layers, tune_pos=self.args.tune_pos,
                               pos_fuse=self.args.pos_fuse,
                               pos_bias=self.args.pos_bias)

        self.model.to(self.device)

    def setup_optim(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(params, lr=self.args.learning_rate, weight_decay=self.args.decay,
                                        betas=(self.args.beta1, self.args.beta2))
        elif self.args.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(params, lr=self.args.learning_rate, weight_decay=self.args.decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(params, lr=self.args.learning_rate, momentum=self.args.alpha,
                                       weight_decay=self.args.decay)
        else:
            raise ValueError(f"undefined optimizer: {self.args.optimizer}")

    def setup_scheduler(self):
        if self.args.decay_mode == 'loss':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True,
                                                                  factor=self.args.lr_decay, patience=self.args.patience)
        elif self.args.decay_mode == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.patience, gamma=self.args.lr_decay)
        elif self.args.decay_mode == 'miAP':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True,
                                                                  factor=self.args.lr_decay,
                                                                  mode="max", patience=self.args.patience)
        elif self.args.decay_mode == "multistep":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.patience, gamma=self.args.lr_decay)

    def scheduler_step(self, metric, loss, epoch):
        if self.args.decay_mode == "loss":
            self.scheduler.step(loss)
        elif self.args.decay_mode == 'step' or self.args.decay_mode == 'multistep':
            self.scheduler.step()
        elif self.args.decay_mode == 'miAP':
            self.scheduler.step(metric[-1])


    def load_model(self):
        logger = self.log_master.get_logger("setup")
        if self.args.checkpoint is not None:
            checkpoint = Path(self.args.checkpoint)
            if not checkpoint.is_file():
                logger.info(f"File not found: {str(checkpoint)}")
            else:
                states = torch.load(str(checkpoint))
                logger.info(f"Restoring checkpoint from file: {str(checkpoint)}")
                optimistic_restore(self.model, states["model_state"])
                self.optimizer.load_state_dict(states["optim_state"])
                epoch = states.get("epoch", 0)
                return epoch
        return 0

    def save_model(self, epoch, metric, loss, best=False):
        if best:
            ckpt_path = Path(self.paths["model_ckpts"]) / Path("model_best.pt")
        else:
            ckpt_path = Path(self.paths["model_ckpts"]) / Path("model_" + str(epoch) + ".pt")
        states = {
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "config": self.args,
            "metric": metric,
            "loss": loss,
            "epoch": epoch
        }
        torch.save(states, str(ckpt_path))

    def save_list(self, data, filename="unseen_classes.txt"):
        filename = self.paths['output'] / Path(filename)
        with open(str(filename), "w") as fout:
            for d in data:
                fout.write(d + '\n')
        mlflow.log_artifacts(str(self.paths['output']))

    def load_data_coco(self):
        logger = self.log_master.get_logger("data")
        logger.info("loading COCO data...")

        coco_root = Path(self.args.coco_root)
        anno_train = str(coco_root / Path('annotations/instances_train2014.json'))
        anno_test = str(coco_root / Path('annotations/instances_val2014.json'))
        img_train_dir = str(coco_root / Path('train2014'))
        img_test_dir = str(coco_root / Path('val2014'))
        # self.coco_common = [33, 22, 24, 63, 68, 64, 72, 30, 70, 25, 75, 50, 49, 46, 53, 41]
        # self.imgnet_common = [21, 340, 414, 620, 651, 673, 760, 795, 859, 879, 883, 937, 950, 954, 963, 968]

        vocab = Vocabulary(self.args.coco_unseen, anno_train, n_val=0, embed_file=self.args.coco_glove)
        all_cats_embed, seen_train_cats, seen_val_cats, unseen_cats, seen_cats, label_dict = vocab.get_data()

        cats_split = {
            'train': batch_map(seen_train_cats, label_dict),
            'val': batch_map(seen_val_cats, label_dict),
            'unseen': batch_map(unseen_cats, label_dict),
            'seen': batch_map(seen_cats, label_dict),
            "all": [i for i in range(all_cats_embed.shape[0])]
        }
        self.all_cats_embed = all_cats_embed
        self.cats_split = cats_split
        mlflow.log_param("unseen_cats", vocab.unseen_cats_names)
        mlflow.log_param("val_cats", vocab.seen_val_cats_names)
        self.save_list(vocab.unseen_cats_names)
        self.save_list(vocab.seen_val_cats_names, filename="seen_val_classes.txt")

        train_dataset = CocoDataset(img_train_dir, anno_train, transform=transform_fn, label_set=seen_cats,
                                    n_val=0, mode="train")
        logger.info(f"Found {len(train_dataset)} training images")

        test_dataset = CocoDataset(img_test_dir, anno_test, transform=transform_fn,
                                   label_set=None, mode="test")
        logger.info(f"Found {len(test_dataset)} testing images")

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch, num_workers=self.args.n_worker,
                                  shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=self.args.batch, num_workers=self.args.n_worker,
                                 shuffle=False)

        all_data_cats = vocab.all_class_names
        train_data_cats = vocab.seen_train_cats_names

        self.load_graph_data(all_data_cats, train_data_cats, "coco")

        return train_loader, test_loader

    def load_graph_data(self, all_data_cats, seen_train_cats, data_name):

        self.all_cats_embed = self.all_cats_embed.to(self.device)

        if self.args.model in ['fast0tag', 'logistic']:
            self.all_train_names = seen_train_cats
            self.all_test_names = all_data_cats

            self.train_cats_embed = get_word_vectors(self.all_train_names, sep="_").to(self.device)
            self.test_cats_embed = get_word_vectors(self.all_test_names, sep="_").to(self.device)

            self.n_train = len(self.all_train_names)
            self.n_test = len(self.all_test_names)

            self.n_cats_train = len(seen_train_cats)
            self.n_cats_all = len(all_data_cats)
            self.n_cats_seen = len(seen_train_cats)
            self.n_cats_unseen = self.n_cats_all - self.n_cats_seen

            self.imgnet_idx = None

            self.adj_matrix_train = None
            self.adj_matrix_test = None

            self.edges_train = None
            self.edges_test = None

            return

        logger = self.log_master.get_logger("data")
        logger.info("load and building graph...")

        if self.args.model == 'skg':
            self.edges_train, self.mat_ids_train = get_edges(seen_train_cats, data_name, neg=self.args.wup_neg, pos=self.args.wup_pos,
                                                             node_dim=self.args.d_dim)
            self.edges_train = [(u.cuda(), v.cuda()) for u, v in self.edges_train]
            self.mat_ids_train = [e.cuda() for e in self.mat_ids_train]

            self.edges_test, self.mat_ids_test = get_edges(all_data_cats, data_name, neg=self.args.wup_neg, pos=self.args.wup_pos,
                                                           node_dim=self.args.d_dim)
            self.edges_test = [(u.cuda(), v.cuda()) for u, v in self.edges_test]
            self.mat_ids_test = [e.cuda() for e in self.mat_ids_test]

            self.n_cats_all = len(all_data_cats)
            self.n_cats_seen = len(seen_train_cats)
            self.n_cats_unseen = self.n_cats_all - self.n_cats_seen
            self.imgnet_idx = None

            self.train_cats_embed = self.all_cats_embed[self.cats_split['seen']]
            self.test_cats_embed = self.all_cats_embed

            return

        if not self.args.use_imgnet:
            self.args.imgnet_meta = None

        is_skg = self.args.model == "skg-posvae"

        if self.args.model in ['ggnn', 'skg', "skg-posvae"]:
            node_dim = self.args.d_dim
        else:
            node_dim = 5

        train_graph_meta = load_graph(seen_train_cats, self.args.imgnet_meta, self.args.wup_neg, self.args.wup_pos,
                                      binary=False, to_dense=True, data_name=data_name, node_dim=node_dim, skg=is_skg)

        test_graph_meta = load_graph(all_data_cats, self.args.imgnet_meta, self.args.wup_neg, self.args.wup_pos,
                                     binary=False, to_dense=True, data_name=data_name, node_dim=node_dim, skg=is_skg)
        logger.info("done")

        self.all_train_names = train_graph_meta["names"]
        self.all_test_names = test_graph_meta["names"]

        self.imgnet_idx = test_graph_meta["imgnet_idx"].to(self.device)

        if is_skg:
            self.edges_train = [(u.cuda(), v.cuda()) for u, v in train_graph_meta["edges"]]
            self.mat_ids_train = [e.cuda() for e in train_graph_meta["mat"]]
            self.edges_test = [(u.cuda(), v.cuda()) for u, v in test_graph_meta["edges"]]
            self.mat_ids_test = [e.cuda() for e in test_graph_meta["mat"]]
        else:
            self.adj_matrix_train = train_graph_meta["adj_wup_pos"].to(self.device).view(-1, 1)
            self.adj_matrix_test = test_graph_meta["adj_wup_pos"].to(self.device).view(-1, 1)

            self.edges_train = train_graph_meta["edges"][0].to(self.device)
            self.edges_test = test_graph_meta["edges"][0].to(self.device)
            self.edges_w_train = train_graph_meta["adj_wup_pos"].to(self.device)
            self.edges_w_test = test_graph_meta["adj_wup_pos"].to(self.device)
            self.edge_vars_train = train_graph_meta["edges"]
            self.edge_vars_test = test_graph_meta["edges"]


        self.train_cats_embed = get_word_vectors(self.all_train_names, sep="_").to(self.device)
        self.test_cats_embed = get_word_vectors(self.all_test_names, sep="_").to(self.device)

        self.n_train = len(self.all_train_names)
        self.n_test = len(self.all_test_names)

        self.n_cats_train = len(seen_train_cats)
        self.n_cats_all = len(all_data_cats)
        self.n_cats_seen = len(seen_train_cats)
        self.n_cats_unseen = self.n_cats_all - self.n_cats_seen


    def load_data_nuswide(self, Dataset):
        logger = self.log_master.get_logger("data")
        logger.info('loading NUS-WIDE data...')

        img_dir = Path(self.args.nuswide_root) / Path("images")
        anno_dir = Path(self.args.nuswide_root) / Path("annotations")

        train_dataset = Dataset(img_dir, anno_dir, transform=transform_fn, mode="train", unseen_file=self.args.nuswide_unseen)

        logger.info(f"Found {len(train_dataset)} training images")

        eval_dataset = Dataset(img_dir, anno_dir, transform=transform_fn, mode="test", unseen_file=self.args.nuswide_unseen)
        logger.info(f"Found {len(eval_dataset)} validation images")

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch, num_workers=self.args.n_worker,
                                  shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self.args.batch, num_workers=self.args.n_worker,
                                 shuffle=False)

        all_cats_embed = get_word_vectors(train_dataset.all_cats, wv_dim=self.args.word_dim)
        all_cats_names = train_dataset.all_cats
        seen_cats_names = train_dataset.seen_cats
        unseen_cats_names = train_dataset.unseen_cats
        self.save_list(unseen_cats_names)
        mlflow.log_param("unseen", unseen_cats_names)

        self.all_cats_embed = all_cats_embed

        cats_split = {
            'train': train_dataset.train_idx,
            'val': train_dataset.val_idx,
            'unseen': train_dataset.unseen_idx,
            'seen': train_dataset.seen_idx
        }
        self.cats_split = cats_split

        self.load_graph_data(all_cats_names, seen_cats_names, "nuswide")

        return train_loader, eval_loader

    def load_data_vg(self):
        logger = self.log_master.get_logger("data")
        logger.info('loading Visual Genome data...')

        img_dir = str(Path(self.args.vg_root) / Path("VG_100K"))
        anno_file = str(self.args.vg_anno)

        train_dataset = VGDataset(img_dir, anno_file, transform=transform_fn, mode="train",
                                  unseen_file=getattr(self.args, 'vg_unseen'), K=self.args.vg_K)
        logger.info(f"Found {len(train_dataset)} training images")

        test_dataset = VGDataset(img_dir, anno_file, transform=transform_fn, mode="test",
                                   unseen_file=getattr(self.args, 'vg_unseen'), K=self.args.vg_K)
        logger.info(f"Found {len(test_dataset)} testing images")

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch, num_workers=self.args.n_worker, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=self.args.batch, num_workers=self.args.n_worker, shuffle=False)

        all_cats_embed = get_word_vectors(train_dataset.all_cats, wv_dim=self.args.word_dim)
        all_cats_names = train_dataset.all_cats
        seen_cats_names = train_dataset.seen_cats
        unseen_cats_names = train_dataset.unseen_cats
        self.save_list(unseen_cats_names)
        mlflow.log_param("unseen", unseen_cats_names)

        self.all_cats_embed = all_cats_embed

        cats_split = {
            'train': train_dataset.train_idx,
            'val': train_dataset.val_idx,
            'unseen': train_dataset.unseen_idx,
            'seen': train_dataset.seen_idx
        }
        self.cats_split = cats_split

        self.load_graph_data(all_cats_names, seen_cats_names, "coco")

        return train_loader, test_loader

    def train_batch(self, data, step):
        images, labels = data
        result = self.model(images, self.all_cats_embed[self.cats_split['train']], self.edges_train, n_coco=self.n_cats_train,
                            imgnet_idx=self.imgnet_idx)
        logits = result["logits"][:, :self.n_cats_train]

        loss_terms = {}
        if self.args.loss == 'bce':
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

        loss_res = {}
        for k, v in loss_terms.items():
            loss_res[k] = v.item()
        return loss_res

    def train_epoch(self, train_loader, epoch):
        logger = self.log_master.get_logger('train')
        logger.info(f'Epoch {epoch: 3}, training...')
        running_loss = 0.0
        steps = 0
        t1 = time.time()
        cnt = 1
        for batch, data in enumerate(tqdm(train_loader, leave=False, ncols=70, unit='b', total=len(train_loader))):
            data = to_device(data, self.device)
            loss = self.train_batch(data, steps)
            if batch > 0 and batch % self.args.log_step == 0:
                mlflow.log_metrics(loss, cnt)
                cnt += 1
            running_loss += sum(loss.values())
            steps += 1
        running_loss /= steps

        elapsed = (time.time() - t1) / 60
        mlflow.log_metric("loss_epoch", running_loss, epoch)
        logger.info(f'Epoch: {epoch:3}, time elapsed: {elapsed:3.1f} mins, running loss: {running_loss: .5f}')

        return running_loss

    def eval_batch(self, data):
        return self.model(*data)["logits"]

    def eval_epoch(self, val_loader, epoch, tag=''):
        logger = self.log_master.get_logger("eval")
        logger.info(f'Epoch {epoch: 3}, evaluating...')
        metric_seen = AveragePrecisionMeter()
        metric_unseen = AveragePrecisionMeter()
        metric_all = AveragePrecisionMeter()

        if self.model.training:
            is_training = True
        else:
            is_training = False

        embed = self.test_cats_embed
        edges = self.edges_test
        seen_idx = self.cats_split["seen"]
        unseen_idx = self.cats_split["unseen"]
        n_cats = self.n_cats_all

        self.model.eval()
        for batch, data in enumerate(tqdm(val_loader, leave=False, ncols=70, unit='b', total=len(val_loader))):
            images, labels = to_device(data, self.device)

            labels_seen = labels[:, seen_idx]
            labels_unseen = labels[:, unseen_idx]

            batch_data = (images, embed, edges, unseen_idx, n_cats, self.imgnet_idx)
            result = self.eval_batch(batch_data).detach()

            logits = result[:, :n_cats]

            logits_seen = logits[:, seen_idx]
            logits_unseen = logits[:, unseen_idx]

            metric_seen.add(logits_seen, labels_seen)
            metric_unseen.add(logits_unseen, labels_unseen)
            metric_all.add(logits, labels)

        metrics_s = metric_seen.overall(tag=tag+"seen_")
        metrics_s_k = metric_seen.overall_topk(3, tag=tag+"seen_top3_")
        mAP_s = np.mean(metric_seen.get_mAP())
        metrics_s[tag+'seen_mAP'] = mAP_s
        miAP_s = metric_seen.get_miAP()
        metrics_s[tag+'seen_miAP'] = miAP_s

        mlflow.log_metrics(metrics_s, epoch)
        mlflow.log_metrics(metrics_s_k, epoch)

        scores = ', '.join([f"{k}:{v:3.3f}" for k, v in metrics_s.items()])
        logger.info(scores)
        scores = ', '.join([f"{k}:{v:3.3f}" for k, v in metrics_s_k.items()])
        logger.info(scores)

        metrics_u = metric_unseen.overall(tag=tag+"unseen_")
        metrics_u_k = metric_unseen.overall_topk(3, tag=tag+"unseen_top3_")
        mAP_u = np.mean(metric_unseen.get_mAP())
        metrics_u[tag+'unseen_mAP'] = mAP_u
        miAP_u = metric_unseen.get_miAP()
        metrics_u[tag+'unseen_miAP'] = miAP_u

        mlflow.log_metrics(metrics_u, epoch)
        mlflow.log_metrics(metrics_u_k, epoch)

        scores = ', '.join([f"{k}:{v:3.3f}" for k, v in metrics_u.items()])
        logger.info(scores)
        scores = ', '.join([f"{k}:{v:3.3f}" for k, v in metrics_u_k.items()])
        logger.info(scores)

        if getattr(self.args, "eval_all", False):
            miAP_all = metric_all.get_miAP()
            mAP_all = np.mean(metric_all.get_mAP())
            mlflow.log_metric("all_miAP", miAP_all, epoch)
            mlflow.log_metric("all_mAP", mAP_all, epoch)
            logger.info(f"all_mAP:{mAP_all:3.3f}")
            logger.info(f"all_miAP:{miAP_all:3.3f}")

        if is_training:
            self.model.train()

        return mAP_s, mAP_u, miAP_s, miAP_u

    def eval_on_train(self):
        logger = self.log_master.get_logger("eval")
        logger.info(f'Epoch {self.epoch: 3}, evaluating on training set...')

        metric_all = AveragePrecisionMeter()

        if self.model.training:
            is_training = True
        else:
            is_training = False

        embed = self.train_cats_embed
        edges = self.edges_train
        seen_idx = self.cats_split["train"]
        unseen_idx = None
        n_cats = self.n_cats_seen

        self.model.eval()
        for batch, data in enumerate(
                tqdm(self.train_loader, leave=False, ncols=70, unit='b', total=len(self.train_loader))):
            images, labels = to_device(data, self.device)

            batch_data = (images, embed, edges, unseen_idx, n_cats, self.imgnet_idx)
            result = self.eval_batch(batch_data).detach()

            logits = result[:, :n_cats]

            metric_all.add(logits, labels)

        metrics_s = metric_all.overall(tag="train_")
        metrics_s_k = metric_all.overall_topk(3, tag="train_top3_")
        mAP_s = np.mean(metric_all.get_mAP())
        metrics_s['train_mAP'] = mAP_s
        miAP_s = metric_all.get_miAP()
        metrics_s['train_miAP'] = miAP_s

        mlflow.log_metrics(metrics_s, self.epoch)
        mlflow.log_metrics(metrics_s_k, self.epoch)
        scores = ', '.join([f"{k}:{v:3.3f}" for k, v in metrics_s.items()])
        logger.info(scores)
        scores = ', '.join([f"{k}:{v:3.3f}" for k, v in metrics_s_k.items()])
        logger.info(scores)

        if is_training:
            self.model.train()

