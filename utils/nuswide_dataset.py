import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import csv
import copy


class NUSWideDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_dir, anno_dir, transform=None, n_val=0, mode="train", n_unseen=16, unseen_file=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_dir: image directory.
            anno_json: coco annotation file path.
            label_set: list of labels, IDs or names.
            transform: image transformation function, callable.
        """
        assert n_val >= 0
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.transform = transform
        self.mode = mode
        self.valid_ids = []
        common = ['plane', 'zebra', 'valley', 'tiger', 'castle']
        unseen_labels_file = Path(anno_dir) / Path("Concepts81.txt")
        seen_labels_file = Path(anno_dir) / Path("NUS_WID_Tags/TagList1k.txt")
        unseen_cats = self.load_label_set(unseen_labels_file)
        seen_cats = self.load_label_set(seen_labels_file)
        assert len(seen_cats) == 1000
        assert len(unseen_cats) == 81

        seen_cats_new = [x for x in seen_cats if x not in unseen_cats]
        seen_label_idx = [i for i, x in enumerate(seen_cats) if x not in unseen_cats]
        assert len(seen_cats_new) == 925
        self.seen_label_idx = torch.tensor(seen_label_idx).long()

        unseen_cats_new = [x for x in unseen_cats if x not in common]
        assert len(unseen_cats_new) == 76
        unseen_label_idx = [i for i, x in enumerate(unseen_cats) if x not in common]
        self.unseen_label_idx = torch.tensor(unseen_label_idx).long()

        self.seen_idx = torch.tensor([i for i in range(925)]).long()
        self.unseen_idx = torch.tensor([i+925 for i in range(len(unseen_cats_new))]).long()
        self.all_cats = seen_cats_new + unseen_cats_new
        self.seen_cats = seen_cats_new
        self.unseen_cats = unseen_cats_new

        self.train_idx = self.seen_idx
        self.val_idx = self.seen_idx

        train_seen_anno = Path(anno_dir) / Path("NUS_WID_Tags/Train_Tags1k.dat")
        test_unseen_anno = Path(anno_dir) / Path("NUS_WID_Tags/Test_Tags81.txt")
        test_seen_anno = Path(anno_dir) / Path("NUS_WID_Tags/Test_Tags1k.dat")

        train_image_file = Path(anno_dir) / Path("ImageList/TrainImagelist.txt")
        test_image_file = Path(anno_dir) / Path("ImageList/TestImagelist.txt")

        if mode == "train":
            self.img_list = self.load_image_list(train_image_file, image_dir)
            self.gt_labels = self.load_gt_labels(train_seen_anno)[:,self.seen_label_idx]
        else:
            self.img_list = self.load_image_list(test_image_file, image_dir)
            test_unseen_gt = self.load_gt_labels(test_unseen_anno)[:, self.unseen_label_idx]
            test_seen_gt = self.load_gt_labels(test_seen_anno)[:, self.seen_label_idx]
            self.gt_labels = torch.cat([test_seen_gt, test_unseen_gt], dim=1)

        assert len(self.img_list) == self.gt_labels.size(0)

    @staticmethod
    def load_label_set(label_file):
        if not os.path.isfile(label_file):
            raise FileNotFoundError(f"file not found: {label_file}")

        label_set = []

        with open(label_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                word = line.split('\n')[0]
                if word != '':
                    label_set.append(word)

        return label_set[:1000]

    def load_image_list(self, image_file, image_dir):
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"file not found: {image_file}")

        image_list = []
        with open(image_file, "r") as fin:
            lines = fin.readlines()
            for idx, line in enumerate(lines):
                filename = line.split()[0]
                filename = os.path.join(image_dir, filename.split('_')[-1])
                if os.path.isfile(filename):
                    image_list.append(filename)
                    self.valid_ids.append(idx)

        return image_list


    def load_gt_labels(self, anno_file):
        if not os.path.isfile(anno_file):
            raise FileNotFoundError(f"file not found: {anno_file}")

        gt_labels = []

        with open(anno_file, "r") as fin:
            reader = fin.readlines()
            for line in reader:
                line = line.split()
                labels = torch.from_numpy(np.array(line) == '1').long()
                gt_labels.append(labels.view(1, -1))

        assert len(self.valid_ids) > 0
        gt_labels = torch.cat(gt_labels, dim=0)[self.valid_ids]
        return gt_labels


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        labels = self.gt_labels[index]
        image = Image.open(os.path.join(self.image_dir, self.img_list[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, labels


class NUSWideDataset81(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_dir, anno_dir, transform=None, n_val=0, mode="train", n_unseen=16, unseen_file=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_dir: image directory.
            anno_json: coco annotation file path.
            label_set: list of labels, IDs or names.
            transform: image transformation function, callable.
        """
        assert n_val >= 0
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.transform = transform
        self.mode = mode
        self.valid_ids = []
        common = ['plane', 'zebra', 'valley', 'tiger', 'castle']
        labels_file = Path(anno_dir) / Path("Concepts81.txt")
        all_cats = self.load_label_set(labels_file)

        unseen_names = []

        if unseen_file is not None:
            with Path(unseen_file).open('r') as fin:
                lines = fin.readlines()
                for line in lines:
                    label = line.split('\n')[0]
                    unseen_names.append(label)

        elif n_unseen > 0:
            all_cats_copy = copy.deepcopy(all_cats)
            while True:
                np.random.shuffle(all_cats_copy)
                unseen_names = all_cats_copy[:n_unseen]
                if set(unseen_names).intersection(set(common)) == set():
                    break
        else:
            unseen_names = all_cats

        self.n_unseen = len(unseen_names)
        self.n_seen = len(all_cats) - self.n_unseen
        self.n_all = len(all_cats)
        seen_cats = []
        unseen_cats = []
        seen_idx = []
        unseen_idx = []
        for i, cat in enumerate(all_cats):
            if cat not in unseen_names:
                seen_idx.append(i)
                seen_cats.append(cat)
            else:
                unseen_idx.append(i)
                unseen_cats.append(cat)

        if len(seen_cats) == 0:
            self.n_seen = self.n_all
            seen_cats = unseen_cats
            seen_idx = unseen_idx

        self.seen_idx = torch.tensor(seen_idx).long()
        self.unseen_idx = torch.tensor(unseen_idx).long()
        self.all_cats = all_cats
        self.seen_cats = seen_cats
        self.unseen_cats = unseen_cats

        # TODO:
        self.train_idx = self.seen_idx
        self.val_idx = self.seen_idx

        train_anno = Path(anno_dir) / Path("NUS_WID_Tags/Train_Tags81.txt")
        test_anno = Path(anno_dir) / Path("NUS_WID_Tags/Test_Tags81.txt")

        train_image_file = Path(anno_dir) / Path("ImageList/TrainImagelist.txt")
        test_image_file = Path(anno_dir) / Path("ImageList/TestImagelist.txt")

        if mode == "train":
            self.img_list = self.load_image_list(train_image_file, image_dir)
            self.gt_labels = self.load_gt_labels(train_anno)[:, self.seen_idx]
        else:
            self.img_list = self.load_image_list(test_image_file, image_dir)
            self.gt_labels = self.load_gt_labels(test_anno)

        nonempty_idx = []
        for i in range(self.gt_labels.size(0)):
            if self.gt_labels[i].sum() > 0:
                nonempty_idx.append(i)

        self.img_list = [x for i, x in enumerate(self.img_list) if i in nonempty_idx]
        self.gt_labels = self.gt_labels[nonempty_idx, :]
        assert len(self.img_list) == self.gt_labels.size(0)

    @staticmethod
    def load_label_set(label_file, n_max=1000):
        if not os.path.isfile(label_file):
            raise FileNotFoundError(f"file not found: {label_file}")

        label_set = []

        with open(label_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                word = line.split('\n')[0]
                if word != '':
                    label_set.append(word)

        return label_set[:n_max]

    def load_image_list(self, image_file, image_dir):
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"file not found: {image_file}")

        image_list = []
        with open(image_file, "r") as fin:
            lines = fin.readlines()
            for idx, line in enumerate(lines):
                filename = line.split()[0]
                filename = os.path.join(image_dir, filename.split('_')[-1])
                if os.path.isfile(filename):
                    image_list.append(filename)
                    self.valid_ids.append(idx)

        return image_list


    def load_gt_labels(self, anno_file):
        if not os.path.isfile(anno_file):
            raise FileNotFoundError(f"file not found: {anno_file}")

        gt_labels = []

        with open(anno_file, "r") as fin:
            reader = fin.readlines()
            for line in reader:
                line = line.split()
                labels = torch.from_numpy(np.array(line) == '1').long()
                gt_labels.append(labels.view(1, -1))

        assert len(self.valid_ids) > 0
        gt_labels = torch.cat(gt_labels, dim=0)[self.valid_ids]
        return gt_labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        labels = self.gt_labels[index]
        image = Image.open(os.path.join(self.image_dir, self.img_list[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, labels


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


    nus_img_dir = '/media/hehuang/Data/nus_wide/images'
    nus_anno_dir = '/media/hehuang/Data/nus_wide/annotations'

    dataset = NUSWideDataset(nus_img_dir, nus_anno_dir, transform=transform_fn, mode="train")

    loader = DataLoader(dataset,
                        batch_size=10,
                        num_workers=2,
                        shuffle=False)
    print(len(dataset))
    for image, target in loader:
        print(image.size())
        print(target.size())
        break

