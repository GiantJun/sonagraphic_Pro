from torch import nn
from torch import load
import logging
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from backbones.multi_branch_net import MultiBranch_cat, MultiBranch_sum
from timm.models import swin_small_patch4_window7_224, vit_base_patch16_224, efficientnet_b5, tnt_b_patch16_224, tnt_s_patch16_224

def get_model(config):
    name = config.backbone.lower()
    net = None
    if name == 'resnet18':
        logging.info('created resnet18!')
        net = resnet18(pretrained=config.pretrained)
    elif name == 'resnet34':
        net = resnet34(pretrained=config.pretrained)
        logging.info('created resnet34!')
    elif name == 'resnet50':
        net = resnet50(pretrained=config.pretrained)
        logging.info('created resnet50!')
    elif name == 'resnet101':
        net = resnet101(pretrained=config.pretrained)
        logging.info('created resnet101!')
    elif name == 'resnet152':
        net = resnet152(pretrained=config.pretrained)
        logging.info('created resnet152!')
    elif name == 'multi_branch_sum':
        net = MultiBranch_sum(config.base_backbone, len(config.select_list), config.class_num)
    elif name == 'multi_branch_cat':
        net = MultiBranch_cat(config.base_backbone, len(config.select_list), config.class_num)
    else:
        raise NotImplementedError('Unknown type {}'.format(config.backbone))

    if 'resnet' in name:
        if len(config.select_list) != 3:
            net.conv1 = nn.Conv2d(len(config.select_list), 64, kernel_size=7, stride=2, padding=3, bias=False)
            logging.info('Change network input channel from 3 to {}'.format(len(config.select_list)))

        dim_mlp = net.fc.in_features
        net.fc = nn.Linear(dim_mlp, config.class_num)
        logging.info('Change network output logits dimention from 1000 to {}'.format(config.class_num))
        if hasattr(config,'nlp_num'):
            if config.nlp_num > 0:
                mlp_list = [nn.Linear(dim_mlp, dim_mlp), nn.ReLU()]
                for i in range(config.nlp_num-1):
                    mlp_list.append(nn.Linear(dim_mlp, dim_mlp))
                    mlp_list.append(nn.ReLU())
                mlp_list.append(net.fc)
                net.fc = nn.Sequential(*mlp_list)
            logging.info('Change network classifier head with {} MLP'.format(config.nlp_num))
    
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