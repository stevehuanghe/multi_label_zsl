import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from pathlib import Path
import pickle
from copy import deepcopy
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_dir, anno_json, label_set=None, transform=None, n_val=0, mode="train", return_filename=False, val_ids=None, val_cats=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_dir: image directory.
            anno_json: coco annotation file path.
            label_set: list of labels, IDs or names.
            transform: image transformation function, callable.
        """
        assert n_val >= 0

        self.coco = COCO(anno_json)
        self.image_dir = image_dir
        self.label_set = label_set
        self.return_filename = return_filename
        self.transform = transform

        if label_set is not None:
            if not isinstance(label_set, list):
                raise ValueError(f"label_set must be a list, but got {type(label_set)}")
            if isinstance(label_set[0], str):
                self.cat_ids = sorted(self.coco.getCatIds(catNms=label_set))
            else:
                self.cat_ids = sorted(label_set)
        else:
            self.cat_ids = sorted(self.coco.getCatIds())

        self.ids = list(sorted(self.coco.imgs.keys()))

        if mode == "train":
            if n_val > 0 and val_cats is not None:
                if isinstance(val_cats[0], str):
                    val_cat_ids = sorted(self.coco.getCatIds(catNms=val_cats))
                else:
                    val_cat_ids = val_cats

                val_ids = deepcopy(self.ids)
                val_ids = self.filter_image_list(val_ids, val_cat_ids)
                self.val_ids = sorted(val_ids[:n_val])
                self.ids = [x for x in self.ids if x not in self.val_ids]
            self.ids = self.filter_image_list(self.ids, self.cat_ids)

        elif mode == "val":
            assert val_ids is not None
            self.ids = val_ids
        else:        # otherwise it is in test mode, and we use all images
            self.ids = self.filter_image_list(self.ids, self.cat_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """Returns one data pair (image and labels)."""
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
        annotation = coco.loadAnns(ann_ids)
        image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        labels = torch.zeros(len(self.cat_ids))
        for ann in annotation:
            cat = ann['category_id']
            idx = self.cat_ids.index(cat)
            labels[idx] = 1

        if self.return_filename:
            return path, image, labels
        return image, labels

    def get_img_ids(self, catIds, union=True):
        ids = set()
        for i, catId in enumerate(catIds):
            if i == 0 and len(ids) == 0:
                ids = set(self.coco.catToImgs[catId])
            else:
                if union:
                    ids |= set(self.coco.catToImgs[catId])
                else:
                    ids &= set(self.coco.catToImgs[catId])
        return list(ids)

    def get_img_ids_tight(self, catIds):
        ids = []
        coco = self.coco
        all_imgs = self.coco.imgs.keys()
        for iid in all_imgs:
            ann_ids = coco.getAnnIds(imgIds=iid, iscrowd=None)
            annotation = coco.loadAnns(ann_ids)
            flag = True
            for ann in annotation:
                cat = ann['category_id']
                if cat not in catIds:
                    flag = False
            if flag:
                ids.append(iid)
        return ids


    def filter_image_list(self, ids, cat_ids):
        """
        filter out images with no labels
        :return:
        """
        valid_ids = []
        for i in ids:
            coco = self.coco
            ann_ids = coco.getAnnIds(imgIds=i, catIds=cat_ids, iscrowd=None)
            annotation = coco.loadAnns(ann_ids)
            labels = np.zeros(len(cat_ids))
            for ann in annotation:
                cat = ann['category_id']
                idx = cat_ids.index(cat)
                labels[idx] = 1
            if np.sum(labels) > 0:
                valid_ids.append(i)
        return valid_ids




if __name__ == '__main__':
    from torch.utils.data import DataLoader


    def transform_fn(image):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        return transform(image)
    dataset = CocoDataset('/media/hehuang/Data/coco/val2014', '/media/hehuang/Data/coco/annotations/instances_val2014.json',
                        transform=transform_fn, label_set=['person','dog','skateboard'])

    loader = DataLoader(dataset,
                         batch_size=10,
                         num_workers=2,
                         shuffle=False)


    for image, target in loader:
        print(image.size())
        print(target)
        break