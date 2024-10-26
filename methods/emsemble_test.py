from torch import nn
from methods.base import Base
import logging
from backbones.network import get_model
import torch
from utils.config import Config
from utils.toolkit import plot_confusion_matrix, plot_ROC_curve
from utils.toolkit import count_parameters
from os.path import join, basename
from os import listdir
import copy
from dataset.data_manager import get_dataloader
import csv

class Multi_Avg_Test(Base):
    def __init__(self, trainer_id:int, config:Config, seed:int):
        super().__init__(trainer_id, config, seed)
        file_names = listdir(config.pretrain_dir)
        file_names.sort()

        pretrain_paths = []
        self.networks = []

        self.batch_size = config.batch_size

        logging.info('=== emsemble models: ===')
        for item in file_names:
            if item.endswith('.pkl'):
                pretrain_paths.append(join(config.pretrain_dir, item))
                logging.info(item)
        logging.info('='*20)

        valid_dataloaders, test_dataloaders = [], []
        for pretrain_path in pretrain_paths:
            temp_config = copy.deepcopy(config)
            saved_dict = torch.load(pretrain_path)
            temp_config.load_saved_config(saved_dict)
            temp_config.pretrain_path = pretrain_path

            data_loaders, class_num, class_names, img_size = get_dataloader(temp_config)
            test_dataloaders.append(data_loaders['test'][0].__iter__())
            valid_dataloaders.append(data_loaders['valid'][0].__iter__())
            temp_config.update({'class_num':class_num, 'class_names':class_names, 'img_size':img_size})
            temp_config.print_config()

            network = get_model(temp_config).eval()
            for name, param in network.named_parameters():
                param.requires_grad = False
                # logging.info("{} require grad={}".format(name, param.requires_grad))
            network = network.cuda()
            if len(self.multiple_gpus) > 1:
                network = nn.DataParallel(network, self.multiple_gpus)
            self.networks.append(network)
        
        self.config.update({'class_num':temp_config.class_num, 'class_names':temp_config.class_names})
        self.dataloaders = {'valid':valid_dataloaders, 'test':test_dataloaders}
        self.class_names = class_names
        self.class_num = class_num

    def train_model(self, dataloaders, tblog, valid_epoch=1):
        pass
    
    def after_train(self, dataloaders, tblog=None):
        # valid
        logging.info('===== Evaluate valid set result ======')
        all_preds, all_labels, all_scores, all_paths = self.get_output(self.dataloaders['valid'])

        # 将 confusion matrix 和 ROC curve 输出到 tensorboard
        cm_name = "{}_valid_Confusion_Matrix".format(self.method)
        cm_figure, cm = plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_name)
        cm_figure.savefig(join(self.save_dir, cm_name+'.png'), bbox_inches='tight')

        acc = torch.sum(all_preds == all_labels).item() / len(all_labels)            
        if self.get_roc_auc:
            roc_name = "{}_valid_ROC_Curve".format(self.method)
            roc_auc, roc_figure, opt_threshold, opt_point = plot_ROC_curve(all_labels, all_scores, self.class_names, roc_name)
            roc_figure.savefig(join(self.save_dir, roc_name+'.png'), bbox_inches='tight')
            
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            specificity = tn / (tn + fp)
            logging.info('acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}, opt_threshold = {}, opt_point = {}'.format(
                acc, roc_auc, precision, recall, specificity, opt_threshold, opt_point))
        else:
            logging.info('acc = {:.4f}'.format(acc))
        if self.config.get_mistake:
            self.log_mistakes(all_preds, all_labels, all_paths, all_scores, 'valid')

        # test        
        logging.info('===== Evaluate test set result ======')
        all_preds, all_labels, all_scores, all_paths = self.get_output(self.dataloaders['test'])

        # 将 confusion matrix 和 ROC curve 输出到 tensorboard
        cm_name = "{}_test_Confusion_Matrix".format(self.method)
        cm_figure, cm = plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_name)
        cm_figure.savefig(join(self.save_dir, cm_name+'.png'), bbox_inches='tight')

        acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
        if self.get_roc_auc:
            roc_name = "{}_test_ROC_Curve".format(self.method)
            roc_auc, roc_figure, opt_threshold, opt_point = plot_ROC_curve(all_labels, all_scores, self.class_names, roc_name)
            roc_figure.savefig(join(self.save_dir, roc_name+'.png'), bbox_inches='tight')

            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            specificity = tn / (tn + fp)
            logging.info('acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}, opt_threshold = {}, opt_point = {}'.format(
                    acc, roc_auc, precision, recall, specificity, opt_threshold, opt_point))
        else:
            logging.info('acc = {:.4f}'.format(acc))
        
        if self.config.get_mistake:
            self.log_mistakes(all_preds, all_labels, all_paths, all_scores, 'test')

    def get_output(self, dataloader):
        all_preds = torch.tensor([], dtype=torch.long)
        all_labels = torch.tensor([], dtype=torch.long)
        all_scores = torch.tensor([])
        all_paths = []

        with torch.no_grad():
            while(1):
                try:
                    scores = torch.zeros((self.batch_size, self.class_num))
                    for idx in range(len(self.networks)):
                        item = next(dataloader[idx])
                        if len(item) == 2:
                            inputs, labels = item
                        elif len(item) == 3:
                            inputs, labels, paths = item
                        model = self.networks[idx]
                        inputs= inputs.cuda(non_blocking=True)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        if idx==0 :
                            all_labels = torch.cat((all_labels, labels.long()), 0)
                            if len(item) == 3:
                                all_paths.extend(paths)
                        
                        # logits 求平均
                        scores[0:outputs.shape[0]] += torch.softmax(outputs.detach().cpu(), 1)
                    
                    _, preds = torch.max(scores,1)
                    all_preds = torch.cat((all_preds, preds.long()), 0)
                    all_scores = torch.cat((all_scores, scores), 0)  
                except StopIteration as e:
                    logging.info('test set inference and evaluation finished')
                    # logging.error(e, exc_info=True, stack_info=True)
                    break

            # 不足batch size 的情况
            minus = labels.shape[0] - self.batch_size
            if minus < 0:
                all_preds = all_preds[0:minus]
                all_scores = all_scores[0:minus]
        
        all_scores /= len(self.networks)*1.0
        
        if len(all_paths) > 0:
            return all_preds, all_labels, all_scores, all_paths
        else:
            return all_preds, all_labels, all_scores, None
    
    def log_mistakes(self, predict, targets, paths, scores, phase):
        csv_name = join(self.config.logdir, self.method+'_'+phase+'_mistake.csv')
        with open(csv_name,'w') as error_csv:
            error_writer = csv.writer(error_csv)
            error_writer.writerow(['Name', 'Target', 'Predict', 'Error_Conf', 'Path'])
            for index in torch.where(predict != targets)[0].tolist():
                error_writer.writerow([self.config.class_names[targets[index]]+'_'+basename(paths[index]),
                    targets[index].item(), predict[index].item(), scores[index][predict[index]].item(), paths[index]])

class Multi_Vote_Test(Multi_Avg_Test):
    
    def after_train(self, dataloaders, tblog=None):
        # valid
        logging.info('===== Evaluate valid set result ======')
        all_preds, all_labels, all_scores, all_paths = self.get_output(self.dataloaders['valid'])

        # 将 confusion matrix 和 ROC curve 输出到 tensorboard
        cm_name = "{}_valid_Confusion_Matrix".format(self.method)
        cm_figure, cm = plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_name)
        cm_figure.savefig(join(self.save_dir, cm_name+'.png'), bbox_inches='tight')

        acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
        if self.get_roc_auc:
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            specificity = tn / (tn + fp)
            logging.info('acc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}'.format(acc, precision, recall, specificity))
        else:
            logging.info('acc = {:.4f}'.format(acc))

        if self.config.get_mistake:
            self.log_mistakes(all_preds, all_labels, all_paths, all_scores, 'valid')

        # test        
        logging.info('===== Evaluate test set result ======')
        all_preds, all_labels, all_scores, all_paths = self.get_output(self.dataloaders['test'])

        # 将 confusion matrix 和 ROC curve 输出到 tensorboard
        cm_name = "{}_test_Confusion_Matrix".format(self.method)
        cm_figure, cm = plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_name)
        cm_figure.savefig(join(self.save_dir, cm_name+'.png'), bbox_inches='tight')

        # 计算 precision 和 recall， 将 zero_division 置为0，使当 precision 为0时不出现warning
        acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
        if self.get_roc_auc:
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            specificity = tn / (tn + fp)
            logging.info('acc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}'.format(acc, precision, recall, specificity))
        else:
            logging.info('acc = {:.4f}'.format(acc))

        if self.config.get_mistake:
            self.log_mistakes(all_preds, all_labels, all_paths, all_scores, 'test')


    def get_output(self, dataloader):
        all_preds = torch.tensor([], dtype=torch.long)
        all_labels = torch.tensor([], dtype=torch.long)
        all_scores = torch.tensor([])
        all_paths = []

        with torch.no_grad():
            while(1):
                try:
                    scores = torch.zeros((self.batch_size, self.class_num))
                    for idx in range(len(self.networks)):
                        item = next(dataloader[idx])
                        if len(item) == 2:
                            inputs, labels = item
                        elif len(item) == 3:
                            inputs, labels, paths = item
                        model = self.networks[idx]
                        inputs= inputs.cuda(non_blocking=True)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        if idx==0 :
                            all_labels = torch.cat((all_labels, labels), 0)
                            if len(item) == 3:
                                all_paths.extend(paths)
                        # 多数投票
                        preds = preds.detach().cpu().tolist()
                        scores[list(range(len(preds))),preds] = scores[list(range(len(preds))),preds]+1
                    
                    _, preds = torch.max(scores,1)
                    all_preds = torch.cat((all_preds, preds.long()), 0)
                    all_scores = torch.cat((all_scores, scores), 0)  
                except StopIteration:
                    logging.debug('test set inference and evaluation finished')
                    break
                
            # 不足batch size 的情况
            minus = labels.shape[0] - self.batch_size
            if minus < 0:
                all_preds = all_preds[0:minus]
                all_scores = all_scores[0:minus]
        
        if len(all_paths) > 0:
            return all_preds, all_labels, all_scores, all_paths
        else:
            return all_preds, all_labels, all_scores, None
        

            
