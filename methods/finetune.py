from torch import nn
from methods.base import Base
import logging
from backbones.network import get_model
import torch
from tqdm import tqdm
from torch.nn.functional import softmax
from utils.toolkit import plot_confusion_matrix, plot_ROC_curve
from os.path import join

class Finetune(Base):
    def __init__(self, trainer_id, config, seed):
        super().__init__(trainer_id, config, seed)
        self.base_backbone = config.base_backbone
        self.network = get_model(config)

        if self.freeze:
            for name, param in self.network.named_parameters():
                if not ('fc' in name or 'classifier' in name or 'heads' in name):
                    param.requires_grad = False
                else:
                    logging.info("{} require grad = True !".format(name))

        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
            
    
    def after_train(self, dataloaders, tblog=None):
        # valid
        if not dataloaders['valid'] is None:
            logging.info('===== Evaluate valid set result ======')
            all_preds, all_labels, all_scores = self.get_output(dataloaders['valid'])

            # 将 confusion matrix 和 ROC curve 输出到 tensorboard
            cm_name = self.method+'_'+self.backbone+"_valid_Confusion_Matrix"
            cm_figure, cm = plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_name)
            if tblog is not None:
                tblog.add_figure(cm_name, cm_figure)
            
            acc = torch.sum(all_preds == all_labels).item() / len(all_labels)

            if self.get_roc_auc:
                roc_name = self.method+'_'+self.backbone+"_valid_ROC_Curve"
                roc_auc, roc_figure, opt_threshold, opt_point = plot_ROC_curve(all_labels, all_scores, self.class_names, roc_name)
                if tblog is not None:
                    tblog.add_figure(roc_name, roc_figure)
                tn, fp, fn, tp = cm.ravel()
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                specificity = tn / (tn + fp)
                logging.info('acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}, opt_threshold = {}, opt_point = {}'.format(
                    acc, roc_auc, precision, recall, specificity, opt_threshold, opt_point))
            else:
                logging.info('acc = {:.4f}'.format(acc))

        # test        
        logging.info('===== Evaluate test set result ======')
        all_preds, all_labels, all_scores = self.get_output(dataloaders['test'])

        # 将 confusion matrix 和 ROC curve 输出到 tensorboard
        cm_name = self.method+'_'+self.backbone+"_test_Confusion_Matrix"
        cm_figure, cm = plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_name)
        if tblog is not None:
            tblog.add_figure(cm_name, cm_figure)

        acc = torch.sum(all_preds == all_labels).item() / len(all_labels)

        if self.get_roc_auc:
            roc_name = self.method+'_'+self.backbone+"_test_ROC_Curve"
            roc_auc, roc_figure, opt_threshold, opt_point = plot_ROC_curve(all_labels, all_scores, self.class_names, roc_name)
            if tblog is not None:
                tblog.add_figure(roc_name, roc_figure)

            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            specificity = tn / (tn + fp)

            logging.info('acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f} , opt_threshold = {} , opt_point = {}'.format(
                    acc, roc_auc, precision, recall, specificity, opt_threshold, opt_point))
        else:
            logging.info('acc = {:.4f}'.format(acc))

    def get_output(self, dataloader):
        self.network.eval()

        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        all_scores = torch.tensor([])

        with torch.no_grad():
            with tqdm(total=len(dataloader), ncols=150) as _tqdm:   # 显示进度条
                for inputs, labels in dataloader:
                    inputs = inputs.cuda()
                    outputs = self.network(inputs)
                    _, preds = torch.max(outputs, 1)
                    scores = softmax(outputs.detach().cpu(),1)
                    preds = preds.detach().cpu()
                    all_preds = torch.cat((all_preds.long(), preds.long()), 0)
                    all_labels = torch.cat((all_labels.long(), labels), 0)
                    all_scores = torch.cat((all_scores,scores), 0)

                    _tqdm.update(1)
        
        return all_preds, all_labels, all_scores

        
        

            
