import argparse
import os
from typing import Any, Mapping, Text
import torch
import yaml

def read_yaml_config(filename: Text) -> Mapping[Text, Any]:
    """Loads the YAML config file and returns the dictionary.
    Args:
        filename (Text): The path to the config to be loaded.
    Returns:
        Mapping[Text, Any]: The loaded configuration.
    Raises:
        ValueError: An error occurs if the file does not exist.
    """
    if not os.path.exists(filename):
        raise ValueError(f"Given file {filename} does not exist.")
    with open(filename, "r") as f:
        config = yaml.load(f)
    return config

class Parser():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Basic parameters
        parser.add_argument('--results_dir', '-o', type=str, default='results_directory', help='models are saved here')
        parser.add_argument('--name', type=str, default='name_exp', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--seed', type=int, default=10)
        parser.add_argument('--phase', type=str, default='train', help='train, val, etc')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1')
        parser.add_argument('--save_step', type=int, default=50)
        parser.add_argument('--checkpoint', type=str, default=None)
        parser.add_argument('--config', type=str, default='config/defaults.yaml', help='Path to the config file.')
        parser.add_argument('--isTrain', default=True, action='store_true')
        parser.add_argument('--resume', default=True, action='store_true')
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--num_workers', type=int, default=0)
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Get the basic options
        opt, _ = parser.parse_known_args()
        # Save and return the parser
        self.parser = parser
        return parser.parse_args()


    def print_options(self, opt):
        # Print and save options
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # Save to the disk
        prefix = opt.name
        out_dir = os.path.join(opt.results_dir, prefix)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        file_name = os.path.join(out_dir, 'log.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            opt_file.flush()
        return file_name


    def parse(self):
        opt = self.gather_options()
        file = self.print_options(opt)
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt, file

    def log(self, ms, log=None):
        print(ms)
        if log:
            log.write(ms + '\n')
            log.flush()
