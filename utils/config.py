import argparse
import yaml
from pprint import pprint

class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        ################################ Protected #####################################
        parser.add_argument('-cf', '--config', metavar='FILE', default=None,
                            help='path to config file')
        parser.add_argument('-r', '--result', metavar='DIR', default='./result',
                            help='path to result directory')
        parser.add_argument('--seed', metavar='INT', default=42, type=int,
                            help='random seed')
        parser.add_argument('-bs', '--batch', metavar='INT', default=512, type=int,
                            help='batch size')
        parser.add_argument('-ep', '--epoch', metavar='INT', default=60, type=int,
                            help='max epoch to train')
        parser.add_argument('--log_step', metavar='INT', default=5, type=int,
                            help='number of steps for logging training loss')
        parser.add_argument('--save_epoch', metavar='INT', default=0, type=int,
                            help='save model every number of epochs')
        parser.add_argument('--decay_epoch', metavar='INT', default=10, type=int,
                            help='decay learning rate every number of epochs')
        ############################## end of protected ################################

        # data
        parser.add_argument('--data', metavar='STR', default='coco',
                            help='use coco or nus_wide data')
                            
        parser.add_argument('--coco_root', metavar='STR', default='/media/hehuang/Data/coco',
                            help='root directory of coco data')
        parser.add_argument('--coco_unseen', metavar='STR', default='./data/mscoco/unseen_classes.txt',
                            help='path for file that stores the names for unseen classes')
        parser.add_argument('--coco_glove', metavar='STR', default='./data/mscoco/coco_glove_word2vec.pkl',
                            help='path for file that stores the glove embeddings for all classes')

        parser.add_argument('--nuswide_root', metavar='STR', default='/media/hehuang/Data/nus_wide',
                            help='root directory of NUS-WIDE data')
        parser.add_argument('--nuswide_unseen', metavar='STR', default='./data/nus_wide/unseen_classes.txt',
                            help='path for file that stores the names for unseen classes')

        parser.add_argument('--vg_root', metavar='STR', default='/media/hehuang/Data/visual_genome',
                            help='root directory of vg data')
        parser.add_argument('--vg_anno', metavar='STR', default='./data/visual_genome/objects.json',
                            help='path for file that stores the annotations')
        parser.add_argument('--vg_unseen', metavar='STR', default='./data/visual_genome/vg_unseen_classes.txt',
                            help='path for file that stores the names for unseen classes')
        parser.add_argument('--vg_K', metavar='INT', default=150, type=int,
                            help='number of top-K class')

        parser.add_argument('--imgnet_meta', metavar='STR', default='./data/meta_data.pkl',
                            help='root directory of NUS-WIDE data')

        parser.add_argument('--n_val_img', metavar='INT', default=0, type=int,
                            help='number of images for validation')
        parser.add_argument('--n_val_class', metavar='INT', default=0, type=int,
                            help='number of classes for validation')
        parser.add_argument('--word_dim', metavar='INT', default=300, type=int, choices=[50, 100, 200, 300],
                            help='word embedding dimensions, in [50, 100, 200, 300]')
        parser.add_argument('-ckpt', '--checkpoint', metavar='FILE',
                            default=None,
                            help='path to checkpoint file to restore')
        parser.add_argument('--setting', metavar='STR', default="trans_image", type=str,
                            choices=["inductive", "trans_all_image", "trans_image"],
                            help='zero-shot setting')

        # training
        parser.add_argument('--n_worker', metavar='INT', default=4, type=int,
                            help='number of wokers for dataloader')

        # optimization
        parser.add_argument('-lr', '--learning_rate', metavar='FLOAT', default=0.0001, type=float,
                            help='learning rate')
        parser.add_argument('-a', '--alpha', metavar='FLOAT', default=0.9, type=float,
                            help='hyper-param ')
        parser.add_argument('-b1', '--beta1', metavar='FLOAT', default=0.9, type=float,
                            help='hyper-param ')
        parser.add_argument('-b2', '--beta2', metavar='FLOAT', default=0.999, type=float,
                            help='hyper-param ')
        parser.add_argument('-dc', '--decay', metavar='FLOAT', default=0.0, type=float,
                            help='weight normalization term')
        parser.add_argument('-opt', '--optimizer', metavar='STR', default='adam', type=str,
                            help='which optimizer to use')
        parser.add_argument('--grad_clip', metavar='FLOAT', default=10.0, type=float,
                            help='clip grad norm')


        # model
        parser.add_argument('--model', metavar='STR', default='rgcn', type=str,
                            help='which model to use')
        parser.add_argument('-bb', '--backbone', metavar='STR', default='resnet101', type=str,
                            help='which backbone to use')
        parser.add_argument('--acti', metavar='STR', default='relu', type=str,
                            help='which activation function to use')
        parser.add_argument('--n_neg', metavar='INT', default=5, type=int,
                            help='number of negative labels')
        parser.add_argument('--dropout', metavar='FLOAT', default=0.0, type=float,
                            help='dropout')
        parser.add_argument('-ls', '--loss', metavar='STR', default='bce', type=str,
                            help='which loss to use')

        ## GAN related
        # parser.add_argument('--n_disc', metavar='INT', default=5, type=int,
        #                     help='number of iterations that D updates before G updates once')
        # parser.add_argument('--g_layers', metavar='STR', default='1024 2048',
        #                     help='hidden layers for generator')
        # parser.add_argument('--d_layers', metavar='STR', default='1024 256',
        #                     help='unshared hidden layers for discriminator')
        # parser.add_argument('--c_layers', metavar='STR', default='1024 1024',
        #                     help='unshared hidden layers for classifier')
        # parser.add_argument('--lambda_gp', metavar='FLOAT', default=10.0, type=float,
        #                     help='gradient penalty for WGAN-GP')
        # parser.add_argument('--gan', metavar='STR', default='wgan-gp', type=str, choices=["lsgan", "wgan-gp"],
        #                     help='which GAN to use')


        ## GCN related
        parser.add_argument('--wup_pos', metavar='FLOAT', default=0.5, type=float,
                            help='threshold for positive relations')
        parser.add_argument('--wup_neg', metavar='FLOAT', default=0.11, type=float,
                            help='threshold for negative relations')
        parser.add_argument('--sparse',  action='store_true')

        parser.add_argument('--fin_layers', metavar='STR', default='1024', help='hidden layers')
        parser.add_argument('--fout_layers', metavar='STR', default='64', help='hidden layers')
        parser.add_argument('--frel_layers', metavar='STR', default='128', help='hidden layers')

        parser.add_argument('--gcn_layers', metavar='STR', default='128 128', help='hidden layers')
        parser.add_argument('--d_dim', metavar='INT', default=300, type=int, help='dimension for GCN input')
        parser.add_argument('--h_dim', metavar='INT', default=64, type=int, help='dimension for attention input')
        parser.add_argument('--t_max', metavar='INT', default=5, type=int, help='dimension for GCN input')
        parser.add_argument('--gamma', metavar='FLOAT', default=0.01, type=float,
                            help='hyper-param to weight ImageNet loss')



        self.parser = parser

        args = self.parser.parse_args()
        if args.config is not None:
            with open(args.config, 'r') as fin:
                options_yaml = yaml.load(fin)
            self.update_values(options_yaml, vars(args))
        self.args = args

    def update_values(self, dict_from, dict_to):
        for key, value in dict_from.items():
            if isinstance(value, dict):
                self.update_values(dict_from[key], dict_to[key])
            # elif value is not None:
            else:
                dict_to[key] = dict_from[key]

    def get_args(self):
        return self.args

    def save_args(self, filename):
        with open(filename, 'w') as out:
            yaml.dump(vars(self.args), out)

if __name__ == "__main__":
    config = ArgParser()
    config.log_args('args.yaml')
