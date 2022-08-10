from torch import nn
from methods.base import Base
import logging
from backbones.network import get_model
from utils.config import Config
from tqdm import tqdm
from os.path import join, basename, dirname, exists
from os import listdir
from utils.toolkit import count_parameters
import imageio
from utils.grad_cam import GradCAM, show_cam_on_image, get_target_layers
from utils.toolkit import check_makedirs
from shutil import rmtree
import cv2

class Gen_Grad_CAM(Base):
    def __init__(self, trainer_id:int, config:Config, seed:int):
        super().__init__(trainer_id, config, seed)
        self.grad_save_dir = join(config.logdir, 'grad_cam')
        check_makedirs(self.grad_save_dir)
        if exists(self.grad_save_dir):
            rmtree(self.grad_save_dir)

        self.network = get_model(config)

        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

    def train_model(self, dataloaders, tblog, valid_epoch=1):
        logging.info('All params before training: {}'.format(count_parameters(self.network)))
        logging.info('Trainable params: {}'.format(count_parameters(self.network, True)))
        pass
    
    def after_train(self, dataloaders, tblog=None):

        cam = GradCAM(model=self.network,
                    target_layers=get_target_layers(self.config.backbone, self.network, self.config.base_backbone),
                    use_cuda=True,
                    is_merge_layers_cam= False )
        
        if not self.config.gen_cam_img_path is None:
            img_names = listdir(self.config.gen_cam_img_path)
            for idx in range(len(img_names)):
                if not idx in self.config.select_list:
                    continue
                img = cv2.cvtColor(cv2.imread(join(self.config.gen_cam_img_path, img_names[idx])), cv2.COLOR_BGR2RGB)
                if 'multi_branch' in self.backbone:
                    result_img = show_cam_on_image(img/255., cam_img[idx], use_rgb=True)
                else:
                    result_img = show_cam_on_image(img/255., cam_img[0], use_rgb=True)
                cv2.imwrite(join(self.grad_save_dir, '{}.png'.format(idx)), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                
                cv2.putText(result_img, 'model {} picuture {}'.format(self.trainer_id, idx), (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, 
                        color=(123,222,238), thickness=1, lineType=cv2.LINE_AA)
            
                # 生成gif
                result_list.append(result_img)
            imageio.mimsave(join(self.grad_save_dir, 'output.gif'), result_list, fps=1)
            pass
        else:
            for phase in ['valid', 'test']:
                with tqdm(total=len(dataloaders[phase])) as _tqdm:   # 显示进度条
                    for input, labels, paths in dataloaders[phase]:
                        grayscale_cam = cam(input_tensor=input, target_category=labels)
                        for batch_idx in range(grayscale_cam.shape[0]):
                            cam_img = grayscale_cam[batch_idx]
                            origin_img_dir = paths[batch_idx]
                            img_save_dir = join(self.grad_save_dir, phase, basename(dirname(origin_img_dir)),
                                        basename(origin_img_dir))
                            check_makedirs(img_save_dir)
                            img_names = listdir(origin_img_dir)
                            img_names.sort()
                            result_list = []
                            for idx in range(len(img_names)):
                                if not idx in self.config.select_list:
                                    continue
                                img = cv2.cvtColor(cv2.imread(join(origin_img_dir, img_names[idx])), cv2.COLOR_BGR2RGB)
                                if 'multi_branch' in self.backbone:
                                    result_img = show_cam_on_image(img/255., cam_img[idx], use_rgb=True)
                                else:
                                    result_img = show_cam_on_image(img/255., cam_img[0], use_rgb=True)
                                cv2.imwrite(join(img_save_dir, '{}.png'.format(idx)), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                                
                                cv2.putText(result_img, 'model {} picuture {}'.format(self.trainer_id, idx), (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, 
                                        color=(123,222,238), thickness=1, lineType=cv2.LINE_AA)
                            
                                # 生成gif
                                result_list.append(result_img)
                            imageio.mimsave(join(img_save_dir, 'output.gif'), result_list, fps=1)
                    
                        _tqdm.update(1)
    
    
        

            
