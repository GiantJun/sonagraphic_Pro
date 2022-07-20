import torch
import torch.nn as nn
import logging
from torchvision.models import resnet18, resnet34, resnet50, resnet101

def get_base_bacbone(base_backbone):
    name = base_backbone.lower()
    net = None
    if name == 'resnet18':
        logging.info('created resnet18!')
        net = resnet18(pretrained=True)
    elif name == 'resnet34':
        net = resnet34(pretrained=True)
        logging.info('created resnet34!')
    elif name == 'resnet50':
        net = resnet50(pretrained=True)
        logging.info('created resnet50!')
    elif name == 'resnet101':
        net = resnet101(pretrained=True)
        logging.info('created resnet101!')
    else:
        raise NotImplementedError('Unknown type {}'.format(base_backbone))
    
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    logging.info('Change network input channel from 3 to {}'.format(1))
    return net

class MultiBranch_cat(nn.Module):
    def __init__(self, base_backbone, branch_num, num_classes):
        super(MultiBranch_cat, self).__init__()
        self.branch_num = branch_num

        model_list = []
        for i in range(branch_num):
            sub_model = get_base_bacbone(base_backbone)
            logging.info('created no pretrained sub-network {}'.format(i))
            model_list.append(sub_model)

        logging.info('subnet branches : {}'.format(self.branch_num))
        
        self.features = nn.ModuleList(model_list)
        
        self.classifier = nn.Linear(1000*self.branch_num, num_classes)
        
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

    # def get_feature_layer(self):
    #     return self.features[-2][-1]

class MultiBranch_sum(nn.Module):
    def __init__(self, base_backbone, branch_num, num_classes):
        super(MultiBranch_sum, self).__init__()
        self.branch_num = branch_num

        model_list = []
        for i in range(branch_num):
            sub_model = get_base_bacbone(base_backbone)
            logging.info('created no pretrained sub-network {}'.format(i))
            model_list.append(sub_model)

        logging.info('subnet branches : {}'.format(self.branch_num))
        
        self.features = nn.ModuleList(model_list)

        self.weights = nn.Parameter(torch.ones((self.branch_num,1)))
        
        self.classifier = nn.Linear(1000, num_classes)
        
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

    # def get_feature_layer(self):
    #     return self.features[-2][-1]