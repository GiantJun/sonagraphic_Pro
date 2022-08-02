from torch.nn import functional as F
from backbones.moco_net import MoCoNet
from backbones.simclr_net import SimCLRNet
from backbones.sup_simclr_net import SupSimCLRNet
from backbones.bal_sup_moco_netV2 import BalSupMoCoNet
from methods.base import Base
import torchvision.models as models
from torch import nn
from tqdm import tqdm
from os.path import join
import torch
import logging
from backbones.network import get_model

class Contrastive_Methods(Base):
    def __init__(self, trainer_id, config, seed):
        super().__init__(trainer_id, config, seed)
        network = get_model(config, class_num=128) # 使用moco, simclr等方法默认的特征向量维度128
        if config.method == 'mocoV2':
            self.network = MoCoNet(network, K=config.K, m=config.m, T=config.T)
        elif config.method == 'simclr':
            self.network = SimCLRNet(network, batch_size=self.config.batch_size, T=config.T)
        elif config.method == 'sup_simclr':
            self.network = SupSimCLRNet(network, batch_size=self.config.batch_size, T=config.T)
        elif config.method == 'bal_sup_mocoV2':
            self.network = BalSupMoCoNet(network, batch_size=self.config.batch_size, T=config.T)
        else:
            raise ValueError('Unknown method {}'.format(config.method))
        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

    def epoch_train(self, dataloader, optimizer, scheduler):
        self.network.train()
        losses = 0.
        with tqdm(total=len(dataloader), ncols=150) as prog_bar:
            for inputs, targets in dataloader:
                img_q, img_k, targets = inputs[0].cuda(), inputs[1].cuda(), targets.cuda()
                
                optimizer.zero_grad()
                if self.method in 'sup_simclr':
                    loss = self.network(img_q, img_k, targets)
                else:
                    logits, labels = self.network(img_q, img_k)
                    loss = F.cross_entropy(logits, labels)
                
                loss.backward()
                optimizer.step()
                losses += loss.item()
            
                prog_bar.update(1)

        scheduler.step()

        train_loss = losses/len(dataloader)
        return None, train_loss
    
    def compute_accuracy(self, model, loader):
        return None
    
    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = join(self.save_dir, filename+'.pkl')
        if state_dict != None:
            model_dict = state_dict
        else:
            model_dict = model.encoder_q.state_dict()
        save_dict = {'state_dict': model_dict}
        save_dict.update(self.config.get_save_config())
        torch.save(save_dict, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))


        
        
        
        

        

