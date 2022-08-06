import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import torch.optim as optim
import logging
from utils.config import Config
from utils.toolkit import count_parameters
from os.path import join
import copy
from torch import nn
from utils.losses import FocalLoss
import copy

class Base(object):
    def __init__(self, trainer_id:int, config:Config, seed:int):
        self.config = copy.deepcopy(config)
        self.trainer_id = trainer_id
        # basic config
        self.multiple_gpus = list(range(len(config.device.split(','))))
        self.method = config.method
        self.save_models = config.save_models
        self.save_dir = config.logdir
        self.seed = seed
        self.backbone = config.backbone
        self.freeze = config.freeze
        self.select_list = config.select_list
        self.get_roc_auc = config.get_roc_auc

        # training config
        self.class_names = config.class_names if hasattr(config, 'class_names') else None
        self.img_size = config.img_size if hasattr(config, 'img_size') else None
        self.epochs = config.epochs
        self.lrate = config.lrate
        
        self.opt_type = config.opt_type
        if self.opt_type == 'sgd':
            self.weight_decay = config.weight_decay
        
        self.scheduler = config.scheduler
        if self.scheduler == 'multi_step':
            self.milestones = config.milestones
            self.lrate_decay = config.lrate_decay
        
        if config.criterion != None:
            if config.criterion == 'ce':
                self.criterion = nn.CrossEntropyLoss()
            elif config.criterion == 'focal':
                self.criterion = FocalLoss(config.gamma,config.alpha)

        self.network = None


    def train_model(self, dataloaders, tblog, valid_epoch=1):

        logging.info('All params before training: {}'.format(count_parameters(self.network)))
        logging.info('Trainable params: {}'.format(count_parameters(self.network, True)))
        # for name, param in self.network.named_parameters():
        #     logging.info("{} require grad={}".format(name, param.requires_grad))

        # 根据 valid dataset 挑选模型
        best_valid = 0.
        best_train = 0.
        best_model_wts = copy.deepcopy(self.network.state_dict())
        best_epoch = 0

        # 设置优化器
        if self.opt_type == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.opt_type == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.lrate)
        else: 
            raise ValueError('No optimazer: {}'.format(self.opt_type))

        if self.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
        elif self.scheduler == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            
        for epoch in range(self.epochs):
            
            train_acc, train_loss = self.epoch_train(dataloaders['train'], optimizer, scheduler)

            # 将平均 loss 和 acc 结果输出到 tensorboard
            if not train_acc is None:
                tblog.add_scalar('trainer{}_seed{}_train/acc'.format(self.trainer_id, self.seed), train_acc, epoch)
            tblog.add_scalar('trainer{}_seed{}_train/loss'.format(self.trainer_id, self.seed), train_loss, epoch)

            if epoch % valid_epoch == 0 and dataloaders['valid'] != None and not self.config.is_two_stage_method:
                valid_acc = self.compute_accuracy(self.network, dataloaders['valid'])
                test_acc = self.compute_accuracy(self.network, dataloaders['test'])
                info = 'Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Valid_accy {:.2f}, Test_accy {:.2f}'.format(
                epoch+1, self.epochs, train_loss, train_acc, valid_acc, test_acc)
                if epoch > 35 and (valid_acc > best_valid or 
                    (best_valid == valid_acc and train_acc > best_train)):
                    best_model_wts = copy.deepcopy(self.network.state_dict())
                    best_valid = valid_acc
                    best_train = train_acc
                    best_epoch = epoch
            elif train_acc is None:
                # 仅训练特征提取器的方法, 隔步保存模型的方式ict
                info = 'Epoch {}/{} => Loss {:.3f}'.format(epoch+1, self.epochs, train_loss)
                if (epoch % 100 == 0 and epoch != 0) and epoch != self.epochs-1:
                    self.save_checkpoint('{}_trainer{}_model_dict_{}'.format(self.method, self.trainer_id, epoch+1), copy.deepcopy(self.network).cpu())
                    logging.info('save model dict from current model')
            else:
                test_acc = self.compute_accuracy(self.network, dataloaders['test'])
                info = 'Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                epoch+1, self.epochs, train_loss, train_acc, test_acc)
                
            logging.info(info)
        
        self.save_checkpoint('{}_trainer{}_model_dict_{}'.format(self.method, self.trainer_id, epoch+1), state_dict=best_model_wts)
        logging.info('save model dict from best valid model')
            
        if dataloaders['valid'] != None and not self.config.is_two_stage_method:
            logging.info('Best model was selected in epoch {} with valid acc={}'.format(best_epoch, best_valid))
            self.network.load_state_dict(best_model_wts)

        # 可视化网络结构
        # tblog.add_graph(self.network, inputs)
        
    def epoch_train(self, dataloader, optimizer, scheduler):
        self.network.train()
        losses = 0.
        correct, total = 0, 0
        with tqdm(total=len(dataloader), ncols=150) as prog_bar:
            for inputs, targets in dataloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                logits = self.network(inputs)

                loss = self.criterion(logits,targets) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += torch.sum(preds == targets).item()
                total += len(targets)

                prog_bar.update(1)

        scheduler.step()

        train_acc = round(correct*100 / total, 2)
        train_loss = losses/len(dataloader)
        return train_acc, train_loss
        

    def compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += torch.sum(predicts == targets).item()
            total += len(targets)

        return np.around(correct*100 / total, decimals=2)

    
    def after_train(self, dataloader, tblog=None):
        test_acc = self.compute_accuracy(self.network, dataloader['test'])
        logging.info('Evaluate test set result: acc={}'.format(test_acc))

        
    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = join(self.save_dir, filename+'.pkl')
        if state_dict != None:
            model_dict = state_dict
        else:
            model_dict = model.state_dict()
        save_dict = {'state_dict': model_dict}
        save_dict.update(self.config.get_save_config())
        torch.save(save_dict, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))

