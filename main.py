import argparse
from utils.toolkit import set_logger
import yaml
from dataset.data_manager import get_dataloader
import torch
from utils.factory import get_trainer
import os
import logging

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='',
                        help='yaml file of settings.')
    return parser

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

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args, seed):
    # log hyperparameter
    logging.info(30*"=")
    logging.info("log hyperparameters in seed {}".format(seed))
    logging.info(30*"-")
    # args = dict(sorted(args.items()))
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
    logging.info(30*"=")

if __name__ == '__main__':
    args = setup_parser().parse_args()
    param = load_yaml(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    if not 'pretrain_path' in args:
        args['pretrain_path'] = None

    os.environ['CUDA_VISIBLE_DEVICES']=args['device']

    tblog = set_logger(args)
    try:
        if 'test' in args['method']:
            saved_dict = torch.load(args['pretrain_path'])
            args.update({'kfold': 1,
                    'img_size':saved_dict['img_size'], 
                    'select_list':saved_dict['select_list'],
                    'backbone': saved_dict['backbone'],
                    'seed': saved_dict['seed'],
                    'base_backbone': saved_dict['base_backbone'],
                    'split_for_valid': False}) # 为了同时测试 内部测试集 和 外部测试集
            data_loaders, class_num, class_names = get_dataloader(args)
            test_dataloaders = {'valid':data_loaders['valid'][0], 'test':data_loaders['test'][0]}
            args.update({'class_num':class_num, 'class_names':class_names})
            seed = saved_dict['seed']
            print_args(args, seed)
            set_random(seed)
            trainer = get_trainer(0, args, seed)
            trainer.train_model(test_dataloaders, tblog)
            trainer.after_train(test_dataloaders, tblog)

        else:
            # 准备数据集
            data_loaders, class_num, class_names = get_dataloader(args)
            args.update({'class_num':class_num, 'class_names':class_names})
            
            for seed in args['seed']:
                print_args(args, seed)
                set_random(seed)
                for idx in range(args['kfold']):
                    logging.info('='*10+' fold {} '.format(idx)+'='*10)
                    fold_dataloaders = {'train':data_loaders['train'][idx], 'valid':data_loaders['valid'][idx], 'test':data_loaders['test'][idx]}
                    trainer = get_trainer(idx, args, seed)
                    trainer.train_model(fold_dataloaders, tblog)
                    trainer.after_train(fold_dataloaders, tblog)
    except Exception as e:
        logging.error(e, exc_info=True, stack_info=True)



    
    


    