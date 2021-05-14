import os
import numpy as np
import argparse
import yaml
import subprocess
from pprint import pprint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class HyperOpt(object):
    def __init__(self, space, config, runner, gpus, gpu_per=1):
        """
        :param space: search space
        :param config: original config file
        :param runner: runner.py
        :param gpus: e.g., '0,1,2,3'
        """
        self.space = space
        self.runner = runner
        self.gpus = gpus.split(',')

        self.pool = {}

        for gid in gpus:
            self.pool[gid] = None

        with open(config, 'r') as fin:
            self.old_config = yaml.load(fin)

    @staticmethod
    def split_gpus(gpus, per):
        N = len(gpus) // per
        split = []
        for i in range(N):
            split.append(','.join(gpus[i*per:(i+1)*per]))
        return split

    def random_sample(self):
        result = {}
        for key in space.keys():
            candidates = self.space[key]
            if isinstance(candidates, list):
                idx = np.random.choice(len(candidates), 1)[0]
                val = candidates[idx]
            elif isinstance(candidates, tuple):
                bound = int((candidates[1] - candidates[0]) // candidates[2])
                cand = [candidates[0] + float(i) * candidates[2] for i in range(bound + 1)]
                val = float(np.random.choice(cand, 1)[0])
            else:
                val = None
                print(f'error -- arg not list or tuple for {key}: {candidates} {type(candidates)}')
            result[key] = val
        return result

    def update_values(self, *, dict_from, dict_to, keep_null=True):
        for key, value in dict_from.items():
            if isinstance(value, dict):
                self.update_values(dict_from=dict_from[key], dict_to=dict_to[key])
            elif keep_null or value is not None:
                dict_to[key] = dict_from[key]

    @staticmethod
    def dump_config(config):
        with open(".config.yaml", 'w') as out:
            yaml.dump(config, out)

    def run_next(self, gpu):
        opts = self.random_sample()
        new_config = self.update_values(dict_from=opts, dict_to=self.old_config)
        self.dump_config(new_config)
        command = f"CUDA_VISIBLE_DEVICES={gpu} python {self.runner} --config .config.yaml"
        print(command)
        p = subprocess.Popen(command)
        self.pool[gpu] = p

    def get_available_gpu(self):
        for k,v in self.pool:
            if v.poll() is not None:
                return k
        return None

    def run(self, N):
        cnt = 1
        while cnt <= N:
            gpu = self.get_available_gpu()
            if gpu is not None:
                print(f"Creating new experiment {cnt}/{N}")
                self.run_next(gpu)
                cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default='config.yaml',
                        help='default config of experiments')
    parser.add_argument('-r', '--runner', default='main.py',
                        help='runner.py of experiments')
    parser.add_argument('-i', '--iter', default=100, type=int, metavar='INT',
                        help='number of attempts for random search')
    parser.add_argument('-p', '--per', default=1, type=int, metavar='INT',
                        help='number of GPUs per process')
    parser.add_argument('-g', '--gpu', metavar='GPU', default='1',
                        help='which gpu to use')

    space = {
        'learning_rate': [1e-4, 1e-5, 1e-6],
        'decay': [0.0, 0.01, 0.001, 0.0001],
        'layers': ['512', '1024 512'],  #
        'acti': ['relu', 'leaky'],  # , 'tanh'
        'norm': ['none', 'batch'],  # , 'layer', 'instance'
        'alpha': (0.5, 1.5, 0.1),
        'beta': (0.5, 3.0, 0.2),
        'gamma': (1, 2, 1),
        'base': (0, 0.5, 0.1),
        'mode': ['add', 'mul'],
        'form': ['exp', 'logit']
    }

    args = parser.parse_args()

    print("default config:")
    pprint(vars(args))
    print("-------------------------------")
    print("search space:")
    pprint(space)

    worker = HyperOpt(space=space, config=args.config, runner=args.runner, gpus=args.gpu, gpu_per=args.per)
    worker.run(args.iter)
