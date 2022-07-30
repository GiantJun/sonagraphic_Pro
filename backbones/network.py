from torch import nn
from torch import load
import logging
import torchvision.models as torch_models
from backbones.multi_branch_net import MultiBranch_cat, MultiBranch_sum

def get_model(config):
    name = config.backbone.lower()
    net = None
    if name in torch_models.__dict__.keys():
        net = torch_models.__dict__[name](pretrained=config.pretrained)
        logging.info('created {} !'.format(name))
    elif name == 'multi_branch_sum':
        net = MultiBranch_sum(config.base_backbone, len(config.select_list), config.class_num)
        logging.info('created multi_branch_sum !')
    elif name == 'multi_branch_cat':
        net = MultiBranch_cat(config.base_backbone, len(config.select_list), config.class_num)
        logging.info('created multi_branch_cat !')
    else:
        raise NotImplementedError('Unknown type {}'.format(config.backbone))

    # 调整模型输入通道, 输出logits维度
    if 'resnet' in name:
        if len(config.select_list) != 3:
            net.conv1 = nn.Conv2d(len(config.select_list), 64, kernel_size=7, stride=2, padding=3, bias=False)
            logging.info('Change network input channel from 3 to {}'.format(len(config.select_list)))

        dim_mlp = net.fc.in_features
        classify_head = []
        if config.mlp_num != None and config.mlp_num > 0:
            for i in range(config.mlp_num):
                classify_head.append(nn.Linear(dim_mlp, dim_mlp))
                classify_head.append(nn.ReLU())
            logging.info('Change network classifier head to {} MLP'.format(config.mlp_num))
        classify_head.append(nn.Linear(dim_mlp, config.class_num))
        if len(classify_head) > 1:
            net.fc = nn.Sequential(*classify_head)
        else:
            net.fc = classify_head[0]
        logging.info('Change network output logits dimention from 1000 to {}'.format(config.class_num))

    elif 'efficientnet' in name:
        if len(config.select_list) != 3:
            net.features[0][0] = nn.Conv2d(len(config.select_list), 48, kernel_size=3, stride=2, padding=1, bias=False)
            logging.info('Change network input channel from 3 to {}'.format(len(config.select_list)))

        dim_mlp = net.classifier[1].in_features
        classify_head = []
        classify_head.append(nn.Dropout(p=0.2, inplace=True))
        if config.mlp_num != None and config.mlp_num > 0:
            for i in range(config.mlp_num):
                classify_head.append(nn.Linear(dim_mlp, dim_mlp))
                classify_head.append(nn.ReLU())
            logging.info('Change network classifier head to {} MLP'.format(config.mlp_num))
        classify_head.append(nn.Linear(dim_mlp, config.class_num))
        net.classifier = nn.Sequential(*classify_head)
        logging.info('Change network output logits dimention from 1000 to {}'.format(config.class_num))
    
    elif 'vit' in name:
        if len(config.select_list) != 3:
            patch_size = net.conv_proj.kernel_size[0]
            net.conv_proj = nn.Conv2d(len(config.select_list), 768, kernel_size=patch_size, stride=patch_size, bias=False)
            logging.info('Change network input channel from 3 to {}'.format(len(config.select_list)))

        dim_mlp = net.heads[0].in_features
        net.heads[0] = nn.Linear(in_features=dim_mlp, out_features=config.class_num, bias=True)
        logging.info('Change network output logits dimention from 1000 to {}'.format(config.class_num))
            
    # 载入自定义预训练模型
    if config.pretrain_path != None and config.pretrained:        
        pretrained_dict = load(config.pretrain_path)['state_dict']
        state_dict = net.state_dict()
        logging.info('special keys in load model state dict: {}'.format(pretrained_dict.keys()-state_dict.keys()))
        for key in (pretrained_dict.keys() & state_dict.keys()):
            state_dict[key] = pretrained_dict[key]
        net.load_state_dict(state_dict)

        logging.info("loaded pretrained_dict_name: {}".format(config.pretrain_path))
    
    return net