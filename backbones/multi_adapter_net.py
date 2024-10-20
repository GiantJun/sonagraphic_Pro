import torch
import torch.nn as nn
import logging
import torchvision.models as torch_models
from backbones.usfn import BEiTBackbone4Seg
from typing import Callable, Iterable

def get_base_bacbone(base_backbone):
    name = base_backbone.lower()
    net = None
    if name in torch_models.__dict__.keys():
        net = torch_models.__dict__[name](pretrained=True)
        logging.info('created base_backbone: {} !'.format(name))
    if name == 'usfn':
        net = BEiTBackbone4Seg(pretrained=True)
        net.fpn1 = nn.Identity()
        net.fpn2 = nn.Identity()
        net.fpn3 = nn.Identity()
        net.fpn4 = nn.Identity()
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

class Special_Adapter_v1(nn.Module):
    def __init__(self, in_planes:int, mid_planes:int, kernel_size:int, conv_group=1):
        super().__init__()
        self.in_planes = in_planes
        self.mid_planes = mid_planes
        self.conv = nn.Conv2d(in_planes, mid_planes, kernel_size=kernel_size, groups=conv_group)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.convTransposed = nn.ConvTranspose2d(mid_planes, in_planes, kernel_size=kernel_size, groups=conv_group)
        self.bn2 = nn.BatchNorm2d(in_planes)
        
        self.alpha = nn.Parameter(torch.ones(1)*0.02)
    
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        
        ### original: conv+bn+ReLU+convT+bn+ReLU ###
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.convTransposed(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = out * self.alpha

        return out

class Multi_Adapter_Net(nn.Module):
    def __init__(self, base_backbone, branch_num, num_classes, pretrained=True, pretrain_path=None, layer_names:Iterable[str]=[]):
        '''
        layers_name can be ['conv1','layer1','layer2','layer3','layer4'] for resnet18
        '''
        super(Multi_Adapter_Net, self).__init__()

        self._base_backbone = base_backbone
        self._branch_num = branch_num
        self._pretrained = pretrained
        self._pretrain_path = pretrain_path
        self._layer_names = layer_names

        self._training_mode = 'test_mode'
        self.forward_batch_size = None

        self._feature_extractor = get_base_bacbone(base_backbone)

        model_dict = dict([*self._feature_extractor.named_modules()]) 
        for layer_id in self._layer_names:
            adapter_id = layer_id.replace('.', '_')+'_adapters'
            self.register_module(adapter_id, nn.ModuleList([]))
            layer = model_dict[layer_id]
            layer.register_forward_pre_hook(self.apply_adapters(adapter_id, branch_num))
        
        self._fc = nn.Linear(1000*branch_num, num_classes)
    
    def train_adapter_mode(self):
        self._training_mode = 'train_adapters'

    def test_mode(self):
        self._training_mode = 'test_mode'
    
    def skip_adapters_mode(self):
        self._training_mode = 'skip_adapters_mode'

    def apply_adapters(self, adapter_id: str, branch_num: int) -> Callable:
        def hook(module, input):
            if isinstance(input, tuple):
                input = input[0]
            b, c, h, w = input.shape

            if len(getattr(self, adapter_id)) < branch_num:
                for branch_id in range(branch_num):
                    getattr(self, adapter_id).append(Special_Adapter_v1(c, c, 3).cuda())
             
            if self._training_mode == 'skip_adapters_mode':
                return (input,)
            else:
                adapter_features = []
                for i, adapter in enumerate(getattr(self, adapter_id)):
                    if b != self.forward_batch_size:
                        adapter_input = input[i*self.forward_batch_size : (i+1)*self.forward_batch_size]
                    else:
                        adapter_input = input
                    adapter_output = adapter(adapter_input)
                    adapter_features.append(adapter_output+adapter_input)
                
                return torch.cat(adapter_features, 0) # b * number of adapters, c, h, w
        return hook
    
    def forward(self, x):
        self.forward_batch_size = x.shape[0]

        features = self._feature_extractor(x)
        slice_feature_list = []
        for i in range(len(self.task_sizes)): # features: [b * number  of adapters, feature_dim]
            task_feature = features[i*self.forward_batch_size : (i+1)*self.forward_batch_size]
                
            slice_feature_list.append(task_feature) # features: [b, feature_dim]
        
        features = torch.cat(slice_feature_list, dim=1) # features: [b, feature_dim * number of adapters]

        
        return self._fc(features)
    
