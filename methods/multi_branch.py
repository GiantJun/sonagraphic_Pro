from torch import nn
from methods.base import Base
import logging
from backbones.network import get_model
import torch
from tqdm import tqdm
from torch.nn.functional import softmax
from utils.toolkit import plot_confusion_matrix, plot_ROC_curve
from os.path import join

class Multi_Branch(Base):
    def __init__(self, trainer_id, args, seed):
        super().__init__(trainer_id, args, seed)
        self.backbone = args['backbone']
        self.network = get_model(args)
        self.select_list = args['select_list']

        if self.freeze:
            for name, param in self.network.named_parameters():
                if not 'fc' in name:
                    param.requires_grad = False
                logging.info("{} require grad={}".format(name, param.requires_grad))

        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
            
    
    def after_train(self, dataloaders, tblog=None):
        # valid
        if not dataloaders['valid'] is None:
            all_preds, all_labels, all_scores = self.get_output(dataloaders['valid'])

            # 将 confusion matrix 和 ROC curve 输出到 tensorboard
            cm_name = self.method+'_'+self.backbone+"_valid_Confusion_Matrix"
            cm_figure, tp, fp, fn, tn = plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_name)
            if tblog is not None:
                tblog.add_figure(cm_name, cm_figure)

            acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
            recall = tn / (tn + fp)
            precision = tn / (tn + fn)
            specificity = tp / (tp + fn)
            logging.info('===== Evaluate valid set result ======')
            logging.info('acc = {:.4f} precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}'.format(acc, precision, recall, specificity))

        # test        
        all_preds, all_labels, all_scores = self.get_output(dataloaders['test'])

        # 将 confusion matrix 和 ROC curve 输出到 tensorboard
        cm_name = self.method+'_'+self.backbone+"_test_Confusion_Matrix"
        cm_figure, tp, fp, fn, tn = plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_name)
        if tblog is not None:
            tblog.add_figure(cm_name, cm_figure)

        roc_name = self.method+'_'+self.backbone+"_test_ROC_Curve"
        roc_auc, roc_figure, opt_threshold, opt_point = plot_ROC_curve(all_labels, all_scores, self.class_names, roc_name)
        if tblog is not None:
            tblog.add_figure(roc_name, roc_figure)

        # 计算 precision 和 recall， 将 zero_division 置为0，使当 precision 为0时不出现warning
        acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
        recall = tn / (tn + fp)
        precision = tn / (tn + fn)
        specificity = tp / (tp + fn)

        logging.info('===== Evaluate test set result ======')
        logging.info('acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}'.format(acc, roc_auc, precision, recall, specificity))

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

    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = join(self.save_dir, filename+'.pkl')
        if state_dict != None:
            save_dict = state_dict
        else:
            save_dict = model.state_dict()
        torch.save({
            'state_dict': save_dict,
            'backbone': self.backbone,
            'select_list': self.select_list,
            'img_size': self.img_size,
            'seed': self.seed
            }, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))
        
        

            
