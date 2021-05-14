import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from utils.word_vectors import get_word_vectors

class Vocabulary(object):

    def __init__(self, unseen_file, anno_json, n_val=0, wv_type='glove.840B', wv_dir='./data/glove', wv_dim=300,
                 embed_file=None, n_unseen=16, val_file=None):
        self.unseen_file = unseen_file
        self.anno_json = anno_json
        self.n_val = n_val

        self.load_class_split(unseen_file, anno_json, n_val, n_unseen, val_file)

        if embed_file is None:
            self.label_embed = get_word_vectors(self.all_class_names, wv_type, wv_dir, wv_dim)
        else:
            with open(embed_file, 'rb') as fin:
                word_embed = pickle.load(fin)
                self.label_embed = torch.from_numpy(word_embed)

    def load_class_split(self, unseen_file, anno_json, n_val=0, n_unseen=20, val_file=None):
        """
        load unseen classes, and return the IDs/names of seen_train, seen_val and unseen classes.
        param:
            unseen_file: path for txt that stores names of unseen classes, one per line
            anno_json: path for coco annotation files, either train or val is ok, since we only need the class IDs
            n_val: number of classes for validation
        return type: tuple of lists
        """

        unseen_names = []

        if unseen_file is None:
            unseen_names = sample_coco_class(n_unseen)
        else:
            with Path(unseen_file).open('r') as fin:
                lines = fin.readlines()
                for line in lines:
                    label = line.split('\n')[0]
                    unseen_names.append(label)

        self.coco = COCO(anno_json)

        self.unseen_cat_ids = list(self.coco.getCatIds(catNms=unseen_names))
        seen_cat_ids = list(set(self.coco.getCatIds()) - set(self.unseen_cat_ids))

        if n_val > 0:
            seen_val_names = []
            if val_file is not None:
                with Path(val_file).open('r') as fin:
                    lines = fin.readlines()
                    for line in lines:
                        label = line.split('\n')[0]
                        seen_val_names.append(label)
            else:
                seen_val_names = sample_coco_class(n_val, unseen_names)
            self.seen_val_cat_ids = sorted(list(self.coco.getCatIds(catNms=seen_val_names)))
            train_cat_ids = list(set(seen_cat_ids) - set(self.seen_val_cat_ids))
            self.seen_train_cat_ids = sorted(train_cat_ids)
        else:
            self.seen_val_cat_ids = sorted(self.unseen_cat_ids)
            self.seen_train_cat_ids = sorted(seen_cat_ids)

        self.all_class_names = []
        self.label_dict = {}
        idx = 0
        for cid in sorted(self.coco.getCatIds()):
            name = self.coco.loadCats(cid)[0]['name']
            self.all_class_names.append(name)
            self.label_dict[cid] = idx
            idx += 1
        
        # make class index start from zero
        self.seen_cat_ids = sorted(seen_cat_ids)
        self.seen_train_cat_ids = sorted(self.seen_train_cat_ids)
        self.seen_val_cat_ids = sorted(self.seen_val_cat_ids)
        self.unseen_cat_ids = sorted(self.unseen_cat_ids)

        self.seen_cats_names = [c["name"] for c in self.coco.loadCats(self.seen_cat_ids)]
        self.seen_train_cats_names = [c["name"] for c in self.coco.loadCats(self.seen_train_cat_ids)]
        self.seen_val_cats_names = [c["name"] for c in self.coco.loadCats(self.seen_val_cat_ids)]
        self.unseen_cats_names = [c["name"] for c in self.coco.loadCats(self.unseen_cat_ids)]

    def get_data(self):
        return self.label_embed, self.seen_train_cat_ids, self.seen_val_cat_ids, self.unseen_cat_ids, self.seen_cat_ids, self.label_dict


def transform_fn(image):
    transform = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    
    return transform(image)


def add_one(data_list):
    return [d + 1 for d in data_list]


def minus_one(data_list):
    return [d - 1 for d in data_list]


def batch_map(query, dictionary):
    return [dictionary[q] for q in query]


def sample_coco_class(num_unseen, exclude=None):
    coco_cats = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    common = ['kite', 'zebra', 'backpack', 'laptop', 'microwave', 'mouse', 'refrigerator', 'skis', 'toaster', 'umbrella',
              'vase', 'broccoli', 'orange', 'banana', 'pizza', 'cup']
    common = set(common)

    if exclude is not None:
        coco_cats = list(set(coco_cats) - set(exclude))

    np.random.shuffle(coco_cats)

    unseen_cats = sorted(coco_cats[:num_unseen])

    while set(unseen_cats).intersection(common) != set():
        np.random.shuffle(coco_cats)
        unseen_cats = sorted(coco_cats[:num_unseen])

    seen_cats = coco_cats[num_unseen:]

    return unseen_cats
    