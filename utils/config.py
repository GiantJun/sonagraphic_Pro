import argparse
from numpy import int64
import yaml
import logging

def load_yaml(settings_path):
    args = {}
    with open(settings_path) as data_file:
        param = yaml.load(data_file, Loader=yaml.FullLoader)
    args.update(param['basic'])
    if not ('test' in args['method'] or 'gen_grad_cam' in args['method']): # 测试不需要指定训练参数
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

        self.overwrite_names = []
        # basic config
        self.basic_config_names = ['device', 'seed', 'num_workers', 'dataset', 'split_for_valid', 'kfold',
                        'backbone', 'pretrained', 'freeze', 'select_list', 'save_models', 'save_name', 'logdir']
        self.special_config_names = ['base_backbone']

        parser.add_argument('--device', nargs='+', type=int, default=None, help='GPU ids, e.g. 0 (for single gpu) or 0 1 2 (for multi gpus)')
        parser.add_argument('--seed', nargs='+', type=int, default=None, help='random seed for the programe, 0 (for single seed) or 0 1 2 (run in seed 0 1 2 respectively)')
        parser.add_argument('--num_workers', type=int, default=None, help='CPUs for dataloader')
        parser.add_argument('--dataset', type=str, default=None, help='dataset to be used')
        parser.add_argument('--split_for_valid', type=bool, default=None, help='whether to split training set to true training set and valid set') # 赋初值为 None 相当于 False
        parser.add_argument('--kfold', type=int, default=None, help='for k-fold validation')
        parser.add_argument('--backbone', type=str, default=None, help='backbone to train')
        parser.add_argument('--method', type=str, default=None, help='methods to apply')
        parser.add_argument('--mlp_num', type=int, default=None, help='use mlp to replace linear classifier head')
        parser.add_argument('--pretrained', type=bool, default=None, help='whether use pretrained network weights to initial the network')
        parser.add_argument('--pretrain_path', type=str, default=None, help='prtrained network weights path to load')
        parser.add_argument('--pretrain_dir', type=str, default=None, help='(For emsemble test) pretrained network weights directory')
        parser.add_argument('--freeze', type=bool, default=None, help='freeze the feature extractor')
        parser.add_argument('--select_list', nargs='+', type=int, default=None, help='image channels selected to input the model')
        parser.add_argument('--save_models', type=bool, default=None, help='save trained models weights')
        parser.add_argument('--save_name', type=str, default=None, help='save dir name for the training result')
        parser.add_argument('--logdir', type=str, default=None, help='Do not consider as an parameters to be set by user')

        parser.add_argument('--get_roc_auc', type=bool, default=None, help='for 2 class dataset')
        parser.add_argument('--get_mistake', type=bool, default=None, help='store mistake predictions to the result directory')
        parser.add_argument('--gen_cam_img_path', type=str, default=None, help='Do not consider as an parameters to be set by user')

        # special config
        parser.add_argument('--base_backbone', type=str, default=None, help='for multi-branch network') # multi_branch
        parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax') # simclr and moco
        parser.add_argument('--K', type=int64, default=None, help='memory queue size for moco') # moco
        parser.add_argument('--m', type=float, default=None, help='momentume rate for moco') # moco

        # training config
        parser.add_argument('--epochs', type=int, default=None, help='training epochs')
        parser.add_argument('--batch_size', type=int, default=None, help='batch size for training')
        parser.add_argument('--lrate', type=float, default=None, help='training learning rate')
        parser.add_argument('--opt_type', type=str, default=None, help='optimizer for training')
        parser.add_argument('--scheduler', type=str, default=None, help='learning rate decay method')
        parser.add_argument('--milestones', nargs='+', type=int, default=None, help='for multi step learning rate decay scheduler')
        parser.add_argument('--lrate_decay', type=float, default=None, help='for multi step learning rate decay scheduler')
        parser.add_argument('--criterion', type=str, default=None, help='loss function, e.g. ce, focal')

        for name, value in vars(parser.parse_args()).items():
            setattr(self, name, value)
        
        if self.config != None:
            init_config = load_yaml(self.config)
            for name, value in init_config.items():
                if getattr(self, name) == None:
                    setattr(self, name, value)
                    self.overwrite_names.append(name)
            print('Loaded config file: {}'.format(self.config))
    
    def get_save_config(self) -> dict:
        result = {}
        for item in self.basic_config_names:
            result.update({item:getattr(self, item)})
        for item in self.special_config_names:
            result.update({item:getattr(self, item)})
        return result
    
    def load_saved_config(self, init_dict: dict) -> None:
        if 'state_dict' in init_dict:
            init_dict.pop('state_dict')
        for key, value in init_dict.items():
            if (not hasattr(self,key)) or getattr(self, key) == None:
                setattr(self, key, value)
                self.overwrite_names.append(key)
                
        if 'test' in self.method or 'gen_grad_cam' in self.method: # 包含 test emsemble_test_avg, emsemble_test_vote
            self.kfold = 1
            self.pretrained = True
            self.split_for_valid = False
        elif self.method == 'retrain':
            self.save_name = 'retrain_' + self.save_name

    def update(self, update_dict: dict) -> None:
        for key, value in update_dict.items():
            setattr(self, key, value)
    
    @property
    def is_two_stage_method(self) -> bool:
        return self.method in ['simclr', 'mocoV2', 'sup_simclr', 'bal_sup_moco']
    
    def print_config(self) -> None:
        logging.info(30*"=")
        logging.info("log hyperparameters in seed {}".format(self.seed))
        logging.info(30*"-")
        for name, value in vars(self).items():
            if not name in ['basic_config_names', 'special_config_names', 'overwrite_names']:
                logging.info('{}: {}'.format(name, value))
        logging.info(30*"=")
        logging.info('overwrite configs : {}'.format(self.overwrite_names))
        