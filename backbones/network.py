from torch import nn
from torch import load
import logging
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from backbones.multi_branch_net import MultiBranch_cat, MultiBranch_sum


def get_model(args):
    name = args['backbone'].lower()
    net = None
    if name == 'resnet18':
        logging.info('created resnet18!')
        net = resnet18(pretrained=args['pretrained'])
    elif name == 'resnet34':
        net = resnet34(pretrained=args['pretrained'])
        logging.info('created resnet34!')
    elif name == 'resnet50':
        net = resnet50(pretrained=args['pretrained'])
        logging.info('created resnet50!')
    elif name == 'resnet101':
        net = resnet101(pretrained=args['pretrained'])
        logging.info('created resnet101!')
    elif name == 'multi_branch_sum':
        net = MultiBranch_sum(args['base_backbone'], len(args['select_list']), args['class_num'])
    elif name == 'multi_branch_cat':
        net = MultiBranch_cat(args['base_backbone'], len(args['select_list']), args['class_num'])
    else:
        raise NotImplementedError('Unknown type {}'.format(args['backbone']))

    if not 'multi_branch' in name:
        if 'select_list' in args and len(args['select_list']) != 3:
            net.conv1 = nn.Conv2d(len(args['select_list']), 64, kernel_size=7, stride=2, padding=3, bias=False)
            logging.info('Change network input channel from 3 to {}'.format(len(args['select_list'])))

        dim_mlp = net.fc.in_features
        net.fc = nn.Linear(dim_mlp, args['class_num'])
        logging.info('Change network output from 1000 to {}'.format(args['class_num']))
        if 'nlp_num' in args:
            if args['nlp_num'] > 0:
                mlp_list = [nn.Linear(dim_mlp, dim_mlp), nn.ReLU()]
                for i in range(args['nlp_num']-1):
                    mlp_list.append(nn.Linear(dim_mlp, dim_mlp))
                    mlp_list.append(nn.ReLU())
                mlp_list.append(net.fc)
                net.fc = nn.Sequential(*mlp_list)
            logging.info('Change network classifier head with {} MLP'.format(args['nlp_num']))
    
    # 载入自定义预训练模型
    if args['pretrain_path'] != None and args['pretrained']:        
        pretrained_dict = load(args['pretrain_path'])['state_dict']
        state_dict = net.state_dict()
        logging.info('special keys in load model state dict: {}'.format(pretrained_dict.keys()-state_dict.keys()))
        for key in (pretrained_dict.keys() & state_dict.keys()):
            state_dict[key] = pretrained_dict[key]
        net.load_state_dict(state_dict)

        logging.info("loaded pretrained_dict_name: {}".format(args['pretrain_path']))
    
    return net