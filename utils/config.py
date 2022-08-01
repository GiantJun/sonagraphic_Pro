import argparse
from numpy import int64
import yaml
import logging

def load_yaml(settings_path):
    args = {}
    with open(settings_path) as data_file:
        param = yaml.load(data_file, Loader=yaml.FullLoader)
    args.update(param['basic'])
    dataset = args['dataset']
    if 'options' in param:
        args.update(param['options'][dataset])
    if 'special' in param:
        args.update(param['special'])
    return args

class Config:

    def __init__(self):
        parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
        parser.add_argument('--config', type=str, default=None, help='yaml file of settings.')

        # basic config
        self.basic_config_names = ['device', 'seed', 'num_workers', 'dataset', 'split_for_valid', 'kfold',
                        'backbone', 'pretrained', 'freeze', 'select_list', 'save_models', 'save_name']
        self.special_config_names = ['base_backbone']

        parser.add_argument('--device', nargs='+', type=int, default='-1')
        parser.add_argument('--seed', nargs='+', type=int, default=0)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--dataset', type=str, default=None)
        parser.add_argument('--split_for_valid', type=bool, default=None)
        parser.add_argument('--kfold', type=int, default=1)
        parser.add_argument('--backbone', type=str, default=None)
        parser.add_argument('--method', type=str, default=None)
        parser.add_argument('--mlp_num', type=int, default=None)
        parser.add_argument('--pretrained', type=bool, default=False)
        parser.add_argument('--pretrain_path', type=str, default=None)
        parser.add_argument('--freeze', type=bool, default=False)
        parser.add_argument('--select_list', nargs='+', type=int, default=list(range(3)))
        parser.add_argument('--save_models', type=bool, default=False)
        parser.add_argument('--save_name', type=str, default=None)
        parser.add_argument('--get_roc_auc', type=bool, default=False)
        parser.add_argument('--get_mistake', type=bool, default=False)

        # special config
        parser.add_argument('--base_backbone', type=str, default=None) # multi_branch
        parser.add_argument('--T', type=float, default=None) # simclr and moco
        parser.add_argument('--K', type=int64, default=None) # moco
        parser.add_argument('--m', type=float, default=None) # moco

        # training config
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--lrate', type=float, default=None)
        parser.add_argument('--opt_type', type=str, default=None)
        parser.add_argument('--scheduler', type=str, default=None)
        parser.add_argument('--milestones', nargs='+', type=int, default=None)
        parser.add_argument('--lrate_decay', type=float, default=0.1)
        parser.add_argument('--criterion', type=str, default=None)

        for name, value in vars(parser.parse_args()).items():
            setattr(self, name, value)
        
        if self.config != None:
            init_config = load_yaml(self.config)
            for name, value in init_config.items():
                setattr(self, name, value)
            print('Loaded config file: {}'.format(self.config))
    
    def get_save_config(self) -> dict:
        result = {}
        for item in self.basic_config_names:
            result.update({item:getattr(self, item)})
        for item in self.special_config_names:
            result.update({item:getattr(self, item)})
        return result
    
    def load_basic_config(self, init_dict: dict) -> None:
        if 'state_dict' in init_dict:
            init_dict.pop('state_dict')
        for key, value in init_dict.items():
            setattr(self, key, value)
        self.kfold = 1
        self.pretrained = True
        self.split_for_valid = False

    def update(self, update_dict: dict) -> None:
        for key, value in update_dict.items():
            setattr(self, key, value)
    
    @property
    def is_two_stage_method(self) -> bool:
        return self.method in ['simclr', 'mocoV2', 'sup_simclr']
    
    def print_config(self) -> None:
        logging.info(30*"=")
        logging.info("log hyperparameters in seed {}".format(self.seed))
        logging.info(30*"-")
        for name, value in vars(self).items():
            if name != 'basic_config_names' and name != 'special_config_names':
                logging.info('{}: {}'.format(name, value))
        logging.info(30*"=")
        