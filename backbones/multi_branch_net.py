import torch
import torch.nn as nn
import logging
import torchvision.models as torch_models
from utils.grad_cam import get_target_layers

def get_base_bacbone(base_backbone):
    name = base_backbone.lower()
    net = None
    if name in torch_models.__dict__.keys():
        net = torch_models.__dict__[name](pretrained=True)
        logging.info('created base_backbone: {} !'.format(name))
    else:
        raise ValueError('Unknown base_backbone type {}'.format(base_backbone))
    
    if 'resnet' in name:
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif 'efficientnet' in name:
        net.features[0][0] = nn.Conv2d(1, net.features[0][0].out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    else:
        raise ValueError('Could not change base_backbone {} input channel'.format(base_backbone))

    logging.info('Change network input channel from 3 to {}'.format(1))
    return net

class MultiBranch_cat(nn.Module):
    def __init__(self, base_backbone, branch_num, num_classes, mlp_num=None):
        super(MultiBranch_cat, self).__init__()
        self.branch_num = branch_num
        self.base_backbone = base_backbone

        model_list = []
        for i in range(branch_num):
            sub_model = get_base_bacbone(base_backbone)
            logging.info('created no pretrained sub-network {}'.format(i))
            model_list.append(sub_model)

        logging.info('subnet branches : {}'.format(self.branch_num))
        
        self.features = nn.ModuleList(model_list)
        
        if mlp_num != None and mlp_num > 0:
            self.classifier = nn.Sequential(*[
                nn.Linear(1000*self.branch_num, 1000),
                nn.ReLU(),
                nn.Linear(1000, num_classes)
            ])
            logging.info('Change network classifier head to {} MLP with output dim {}'.format(mlp_num, num_classes))
        else:
            self.classifier = nn.Linear(1000*self.branch_num, num_classes)
            logging.info('Change network classifier head with output dim {}'.format(num_classes))
        
    def forward(self, x):
        # assert x.shape[1] % self.subnetwork_num == 0, "picture channels do not match!"
        out = []
        for i in range(self.branch_num):
            temp = x[:,i,:,:].unsqueeze(1)
            temp = self.features[i](temp)   
            temp = temp.flatten(1)
            temp = temp.unsqueeze(1)    # b, 1, feature_dim
            out.append(temp)
        
        out = torch.cat(out, 2)
        out = torch.squeeze(out,1)

        return self.classifier(out)

class MultiBranch_sum(nn.Module):
    def __init__(self, base_backbone, branch_num, num_classes, mlp_num=None):
        super(MultiBranch_sum, self).__init__()
        self.branch_num = branch_num
        self.base_backbone = base_backbone

        model_list = []
        for i in range(branch_num):
            sub_model = get_base_bacbone(base_backbone)
            logging.info('created no pretrained sub-network {}'.format(i))
            model_list.append(sub_model)

        logging.info('subnet branches : {}'.format(self.branch_num))
        
        self.features = nn.ModuleList(model_list)

        self.weights = nn.Parameter(torch.ones((self.branch_num,1)))
        
        if mlp_num != None and mlp_num > 0:
            self.classifier = nn.Sequential(*[
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, num_classes)
            ])
            logging.info('Change network classifier head to {} MLP with output dim {}'.format(mlp_num, num_classes))
        else:
            self.classifier = nn.Linear(1000, num_classes)
            logging.info('Change network classifier head with output dim {}'.format(num_classes))
        
    def forward(self, x):
        # assert x.shape[1] % self.subnetwork_num == 0, "picture channels do not match!"
        out = []
        for i in range(self.branch_num):
            temp = x[:,i,:,:].unsqueeze(1)
            temp = self.features[i](temp)   
            temp = temp.flatten(1)
            temp = temp.unsqueeze(1)    # b, 1, feature_dim
            out.append(temp)
        
        out = torch.cat(out,1)
        out = (out * self.weights).sum(1)

        return self.classifier(out)