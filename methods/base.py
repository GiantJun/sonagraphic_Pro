import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import torch.optim as optim
import logging
from utils.toolkit import count_parameters
from os.path import join
import copy
from torch import nn
from utils.losses import FocalLoss

class Base(object):
    def __init__(self, trainer_id, args, seed):
        self.trainer_id = trainer_id
        
        self.class_names = args['class_names'] if 'class_names' in args else None
        self.img_size = args['img_size'] if 'img_size' in args else None

        self.multiple_gpus = list(range(len(args['device'].split(','))))
        self.method = args['method']
        self.epochs = args['epochs'] if 'epochs' in args else None
        self.lrate = args['lrate'] if 'lrate' in args else None

        self.seed = seed
        self.backbone = args['backbone'] if 'backbone' in args else None

        self.opt_type = args['opt_type'] if 'opt_type' in args else None
        if self.opt_type == 'sgd':
            self.weight_decay = args['weight_decay']

        self.scheduler = args['scheduler'] if 'scheduler' in args else None
        if self.scheduler == 'multi_step':
            self.milestones = args['milestones']
            self.lrate_decay = args['lrate_decay']
        
        self.network = None
        self.save_models = args['save_models'] if 'scheduler' in args else None
        self.save_dir = args['logdir']

        if 'criterion' in args:
            if args['criterion'] == 'ce':
                self.criterion = nn.CrossEntropyLoss()
            elif args['criterion'] == 'focal':
                self.criterion = FocalLoss(args['gamma'],args['alpha'])

        self.freeze = False if not 'freeze' in args else args['freeze']
        

    def train_model(self, dataloaders, tblog, valid_epoch=1):

        logging.info('All params before training: {}'.format(count_parameters(self.network)))
        logging.info('Trainable params: {}'.format(count_parameters(self.network, True)))
        # for name, param in self.network.named_parameters():
        #     logging.info("{} require grad={}".format(name, param.requires_grad))

        # 根据 valid dataset 挑选模型
        best_valid = 0.
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

            if epoch % valid_epoch == 0 and dataloaders['valid'] != None and not ('mocov2' in self.method):
                valid_acc = self.compute_accuracy(self.network, dataloaders['valid'])
                test_acc = self.compute_accuracy(self.network, dataloaders['test'])
                info = 'Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Valid_accy {:.2f}, Test_accy {:.2f}'.format(
                epoch+1, self.epochs, train_loss, train_acc, valid_acc, test_acc)
                if epoch > 30 and valid_acc > best_valid:
                    best_model_wts = copy.deepcopy(self.network.state_dict())
                    best_valid = valid_acc
                    best_epoch = epoch
            elif train_acc is None:
                info = 'Epoch {}/{} => Loss {:.3f}'.format(
                epoch+1, self.epochs, train_loss)
            else:
                test_acc = self.compute_accuracy(self.network, dataloaders['test'])
                info = 'Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                epoch+1, self.epochs, train_loss, train_acc, test_acc)
                
            logging.info(info)

            # 不采用隔步保存模型的方式
            # if self.save_models and (epoch % 100 == 0 or epoch==self.epochs-1) and epoch > 0:
            #     if dataloaders['valid'] != None and not ('moco' in self.method):
            #         self.save_checkpoint('trainer{}_model_dict_{}'.format(self.trainer_id, epoch+1), state_dict=best_model_wts)
            #         logging.info('save model dict from best valid model')
            #     else:
            #         self.save_checkpoint('trainer{}_model_dict_{}'.format(self.trainer_id, epoch), copy.deepcopy(self.network).cpu())
            #         logging.info('save model dict from current model')
        
        self.save_checkpoint('trainer{}_model_dict'.format(self.trainer_id), state_dict=best_model_wts)
        logging.info('save model dict from best valid model')
            
        if dataloaders['valid'] != None and not ('mocov2' in self.method):
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
            save_dict = state_dict
        else:
            save_dict = model.state_dict()
        torch.save({
            'state_dict': save_dict,
            'backbone': self.backbone
            }, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))

