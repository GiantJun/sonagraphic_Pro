from os.path import dirname
from utils.toolkit import set_logger
from dataset.data_manager import get_dataloader
import torch
from utils.factory import get_trainer
import os
import logging
import re
from utils.config import Config
import copy

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    config = Config()

    os.environ['CUDA_VISIBLE_DEVICES']=config.device

    try:
        if config.method in ['test', 'retrain', 'gen_grad_cam']:
            saved_dict = torch.load(config.pretrain_path)
            config.load_saved_config(saved_dict)
            config.logdir = dirname(config.pretrain_path)
            
            if config.method == 'retrain':
                tblog = set_logger(config, ret_tblog=True, reuse=True)
            else:
                tblog = set_logger(config, ret_tblog=False, reuse=True) # 复用文件夹
            
            data_loaders, class_num, class_names, img_size = get_dataloader(config)
            data_loaders = {'train':data_loaders['train'][0], 'valid':data_loaders['valid'][0], 'test':data_loaders['test'][0]}
            config.update({'class_num':class_num, 'class_names':class_names, 'img_size':img_size})
            seed = saved_dict['seed']
            config.print_config()
            set_random(seed)

            trainer_id = re.search('\d+', os.path.basename(config.pretrain_path)).group()
            trainer = get_trainer(trainer_id, config, seed)
            trainer.train_model(data_loaders, tblog)
            trainer.after_train(data_loaders, tblog)

        elif 'emsemble_test' in config.method:
            config.update({'save_name': os.path.basename(config.pretrain_dir)})
            config.logdir=config.pretrain_dir
            tblog = set_logger(config, ret_tblog=False, reuse=True) # 复用文件夹
            seed = 0
            set_random(seed)
            trainer = get_trainer(0, config, seed)
            trainer.train_model(None, tblog)
            trainer.after_train(None, tblog)

        else: # 训练模型
            tblog = set_logger(config, ret_tblog=True, rename=True)
            # 准备数据集
            data_loaders, class_num, class_names, img_size = get_dataloader(config)
            config.update({'class_num':class_num, 'class_names':class_names, 'img_size':img_size})
            
            for seed in config.seed:
                temp_config = copy.deepcopy(config)
                temp_config.update({'seed':seed})
                temp_config.print_config()
                set_random(seed)
                for idx in range(temp_config.kfold):
                    logging.info('='*10+' fold {} '.format(idx)+'='*10)
                    fold_dataloaders = {'train':data_loaders['train'][idx], 'valid':data_loaders['valid'][idx], 'test':data_loaders['test'][idx]}
                    trainer = get_trainer(idx, temp_config, seed)
                    trainer.train_model(fold_dataloaders, tblog)
                    trainer.after_train(fold_dataloaders, tblog)

    except Exception as e:
        logging.error(e, exc_info=True, stack_info=True)



    
    


    