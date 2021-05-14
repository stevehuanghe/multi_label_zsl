import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
import json
from collections import defaultdict, Counter
import copy


class VGDataset(data.Dataset):
    def __init__(self, image_dir, image_anno, transform=None, mode="train", n_val_class=0, n_val_img=0,
                 n_unseen=50, unseen_file=None, K=150, n_test=30000, seen_val_file=None):

        # there are 495 common objects between ImageNet and VG,
        # note that VG has 4700+ objects, we take the top-K most frequent
        self.common = ['fountain_pen', 'clog', 'microwave', 'brass', 'table_lamp', 'cellular_telephone', 'model_t',
                       'barber_chair', 'fire_engine', 'basset', 'mask', 'poncho', 'giant_panda', 'ice_lolly', 'apron',
                       'lens_cap', 'barrow', 'dogsled', 'microphone', 'pole', 'bullet_train', 'binoculars', 'zucchini',
                       'valley', 'goblet', 'ear', 'paper_towel', 'cardigan', 'cup', 'collie', 'dalmatian', 'lakeside',
                       'lawn_mower', 'coffeepot', 'pitcher', 'ice_cream', 'pop_bottle', 'ski', 'snail', 'sunscreen',
                       'frying_pan', 'kimono', 'aircraft_carrier', 'teddy', 'otter', 'cock', 'bannister', 'pickup',
                       'mosque', 'strainer', 'remote_control', 'book_jacket', 'plastic_bag', 'bath_towel', 'cd_player',
                       'sea_lion', 'space_heater', 'beer_glass', 'bathing_cap', 'soccer_ball', 'pineapple', 'zebra', 'suit',
                       'mosquito_net', 'jinrikisha', 'hammer', 'water_buffalo', 'wool', 'cocktail_shaker', 'dishrag',
                       'gasmask', 'cliff', 'library', 'printer', 'prison', 'restaurant', 'orangutan', 'rotisserie',
                       'dam', 'monastery', 'ladybug', 'drake', 'pool_table', 'baseball', 'birdhouse', 'hamster', 'crash_helmet', 'bow_tie', 'lampshade', 'barbershop', 'cassette', 'grasshopper', 'hook', 'stingray', 'shopping_basket', 'hip', 'padlock', 'broccoli', 'vulture', 'cash_machine', 'lipstick', 'beach_wagon', 'tricycle', 'safe', 'cheetah', 'tripod', 'jeep', 'geyser', 'water_jug', 'screen', 'tape_player', 'broom', 'hand_blower', 'pencil_box', 'cheeseburger', 'band_aid', 'freight_car', 'african_elephant', 'coral_reef', 'carousel', 'cauliflower', 'bagel', 'marimba', 'honeycomb', 'cougar', 'street_sign', 'rifle', 'rule', 'butternut_squash', 'jersey', 'eel', 'tub', 'wing', 'cleaver', 'ashcan', 'breastplate', 'tractor', 'binder', 'bottlecap', 'web_site', 'desktop_computer', 'gazelle', 'hare', 'soup_bowl', 'vacuum', 'revolver', 'trench_coat', 'brassiere', 'cricket', 'pinwheel', 'toaster', 'slot', 'bee', 'trombone', 'lemon', 'computer_keyboard', 'catamaran', 'ant', 'wok', 'dough', 'accordion', 'water_bottle', 'envelope', 'pretzel', 'carton', 'limousine', 'bookshop', 'plow', 'koala', 'teapot', 'mountain_bike', 'toucan', 'promontory', 'park_bench', 'beer_bottle', 'mailbox', 'scale', 'reel', 'red_wine', 'traffic_light', 'border_collie', 'syringe', 'gong', 'greenhouse', 'espresso', 'radiator', 'dome', 'basketball', 'hog', 'banjo', 'beacon', 'seashore', 'caldron', 'piggy_bank', 'cab', 'motor_scooter', 'totem_pole', 'obelisk', 'tank', 'monitor', 'corn', 'grocery_store', 'pick', 'sweatshirt', 'ambulance', 'passenger_car', 'bonnet', 'snowmobile', 'necklace', 'bathtub', 'whistle', 'cinema', 'loudspeaker', 'mushroom', 'liner', 'iron', 'matchstick', 'knee_pad', 'sandal', 'volleyball', 'forklift', 'hay', 'nail', 'stopwatch', 'espresso_maker', 'sandbar', 'bow', 'crock_pot', 'llama', 'patio', 'maypole', 'oscilloscope', 'football_helmet', 'jaguar', 'space_bar', 'tile_roof', 'daisy', 'eskimo_dog', 'chocolate_sauce', 'robin', 'stethoscope', 'maze', 'knot', 'mitten', 'saltshaker', 'wine_bottle', 'leopard', 'hen', 'ski_mask', 'hotdog', 'swimming_trunks', 'hot_pot', 'parking_meter', 'file', 'warthog', 'organ', 'yurt', 'wardrobe', 'racket', 'crayfish', 'handkerchief', 'flamingo', 'cowboy_hat', 'nipple', 'stupa', 'picket_fence', 'cello', 'cocker_spaniel', 'diaper', 'breakwater', 'modem', 'quilt', 'harmonica', 'lifeboat', 'spatula', 'digital_clock', 'beaker', 'ballpoint', 'police_van', 'mortar', 'refrigerator', 'wig', 'partridge', 'pier', 'tick', 'pug', 'swing', 'stove', 'home_theater', 'radio', 'tabby', 'crutch', 'quill', 'pillow', 'golfcart', 'scoreboard', 'cannon', 'bucket', 'cornet', 'ballplayer', 'harvester', 'confectionery', 'torch', 'peacock', 'parachute', 'hourglass', 'drumstick', 'hyena', 'ox', 'crate', 'washer', 'bikini', 'ipod', 'crossword_puzzle', 'shopping_cart', 'washbasin', 'combination_lock', 'holster', 'banana', 'goldfish', 'barbell', 'sarong', 'chihuahua', 'palace', 'pedestal', 'sock', 'macaw', 'ladle', 'jigsaw_puzzle', 'umbrella', 'boston_bull', 'balloon', 'barometer', 'bib', 'coyote', 'wombat', 'acorn_squash', 'pizza', 'dining_table', 'vase', 'minibus', 'television', 'waffle_iron', 'strawberry', 'screwdriver', 'wild_boar', 'water_tower', 'shower_cap', 'ostrich', 'missile', 'swab', 'goose', 'wallet', 'flute', 'church', 'feather_boa', 'snorkel', 'gown', 'barrel', 'snowplow', 'stone_wall', 'streetcar', 'cowboy_boot', 'space_shuttle', 'meat_loaf', 'artichoke', 'toy_poodle', 'candle', 'chime', 'plate', 'dishwasher', 'jean', 'moped', 'lab_coat', 'miniskirt', 'switch', 'entertainment_center', 'notebook', 'sports_car', 'fly', 'shovel', 'violin', 'tiger', 'kite', 'pajama', 'trailer_truck', 'harp', 'soap_dispenser', 'dock', 'menu', 'acorn', 'meerkat', 'upright', 'granny_smith', 'gorilla', 'projector', 'beagle', 'wall_clock', 'toilet_tissue', 'bison', 'car_wheel', 'boxer', 'hummingbird', 'sunglass', 'unicycle', 'pot', 'desk', 'stretcher', 'barn', 'bakery', 'lion', 'thimble', 'fur_coat', 'crane', 'coil', 'cucumber', 'laptop', 'drum', 'medicine_chest', 'paddlewheel', 'hippopotamus', 'sombrero', 'burrito', 'golf_ball', 'spindle', 'horse_cart', 'manhole_cover', 'racer', 'analog_clock', 'garbage_truck', 'bubble', 'impala', 'plane', 'shoe_shop', 'odometer', 'pomegranate', 'coffee_mug', 'jay', 'tennis_ball', 'window_shade', 'chow', 'dumbbell', 'cloak', 'airliner', 'crib', 'lotion', 'convertible', 'toilet_seat', 'paddle', 'backpack', 'fountain', 'disk_brake', 'recreational_vehicle', 'corkscrew', 'chain', 'groom', 'tray', 'tow_truck', 'vault', 'thatch', 'altar', 'guacamole', 'dragonfly', 'paintbrush', 'buckle', 'beaver', 'mouse', 'boathouse', 'bookcase', 'cradle', 'sundial', 'starfish', 'car_mirror', 'spider_web', 'bassinet', 'seat_belt', 'hair_spray', 'plate_rack', 'shower_curtain', 'submarine', 'sax', 'speedboat', 'chickadee', 'gas_pump', 'pelican', 'pencil_sharpener', 'school_bus', 'cassette_player', 'shield', 'throne', 'canoe', 'china_cabinet', 'pill_bottle', 'abacus', 'minivan']

        self.image_dir = image_dir
        self.image_anno = image_anno
        self.transform = transform

        self.all_cats, img_data = self.load_all_cats(image_anno, K)
        self.seen_cats, self.unseen_cats = self.get_cats_split(self.all_cats, unseen_file, n_unseen)

        self.seen_idx = [i for i, x in enumerate(self.all_cats) if x in self.seen_cats]
        self.unseen_idx = [i for i, x in enumerate(self.all_cats) if x in self.unseen_cats]

        if n_val_class > 0 or seen_val_file is not None:
            self.seen_train_cats, self.seen_val_cats = self.get_cats_split(self.seen_cats, seen_val_file, n_val_class)
            self.train_idx = [i for i, x in enumerate(self.all_cats) if x in self.seen_train_cats]
            self.val_idx = [i for i, x in enumerate(self.all_cats) if x in self.seen_val_cats]
        else:
            self.train_idx = self.seen_idx
            self.val_idx = self.unseen_idx

        self.label2idx = self.get_label2idx(self.all_cats)

        test_data = img_data[:n_test]
        train_data = img_data[n_test:]
        if n_val_img > 0:
            val_data = train_data[:n_val_img]
            train_data = train_data[n_val_img:]
        else:
            val_data = train_data

        assert mode in ['train', 'val', 'test']

        if mode == 'train':
            img_data = train_data
            label_idx = self.train_idx
        elif mode == 'val':
            img_data = val_data
            label_idx = self.seen_idx
        else:
            img_data = test_data
            label_idx = None

        self.images, self.labels = self.parse_image_data(img_data, self.label2idx, image_dir, label_idx)

    @staticmethod
    def load_all_cats(image_anno, K):
        objects = defaultdict(lambda: 0)
        corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']

        with open(image_anno, 'r') as fin:
            obj_anno = json.load(fin)

        img_data = []
        for sample in obj_anno:
            basename = '{}.jpg'.format(sample['image_id'])
            if basename in corrupted_ims:
                continue

            img_data.append(sample)

            objs = sample['objects']
            for o in objs:
                if len(o['synsets']) == 0:
                    continue
                name = o['synsets'][0].split('.')[0]
                objects[name] += 1
        items = list(objects.items())
        items = sorted(items, key=lambda x: -x[1])
        topk_cats = [x[0] for x in items[:K]]
        return topk_cats, img_data

    def get_cats_split(self, all_cats, unseen_file, n_unseen):
        unseen_names = []
        if unseen_file is not None:
            with Path(unseen_file).open('r') as fin:
                lines = fin.readlines()
                for line in lines:
                    label = line.split('\n')[0]
                    unseen_names.append(label)
        else:
            all_cats_copy = copy.deepcopy(all_cats)
            while True:
                np.random.shuffle(all_cats_copy)
                unseen_names = all_cats_copy[:n_unseen]
                if set(unseen_names).intersection(set(self.common)) == set():
                    break
        seen_names = list(set(all_cats) - set(unseen_names))

        return seen_names, unseen_names

    @staticmethod
    def get_label2idx(label_set):
        lab2idx = {}
        for i, n in enumerate(label_set):
            lab2idx[n] = i
        return lab2idx

    @staticmethod
    def parse_image_data(data, label_map, image_dir, label_idx=None):
        image_paths = []
        gt_labels = []
        for d in data:
            labels = np.zeros(len(label_map.keys()))
            image_id = d['image_id']
            for o in d['objects']:
                if len(o['synsets']) == 0:
                    continue
                name = o['synsets'][0].split('.')[0]
                if name in label_map.keys():
                    idx = label_map[name]
                    labels[idx] = 1
            basename = '{}.jpg'.format(image_id)
            filename = os.path.join(image_dir, basename)
            if os.path.exists(filename):  # and np.sum(labels) > 0
                image_paths.append(filename)
                gt_labels.append(torch.tensor(labels).long().unsqueeze(0))

        gt_labels = torch.cat(gt_labels, dim=0)
        assert gt_labels.size(0) == len(image_paths)
        if label_idx is not None:
            gt_labels = gt_labels[:, label_idx]
        return image_paths, gt_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        labels = self.labels[index]
        image = Image.open(os.path.join(self.image_dir, self.images[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, labels
