from torch import nn
from methods.base import Base
import logging
from backbones.network import get_model
import torch
from utils.config import Config
from utils.toolkit import count_parameters
from os.path import join, basename, dirname, splitext
from os import listdir
import torchvision.transforms as transforms
import cv2
from PIL import Image
import torch
import imageio
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.toolkit import check_makedirs
from torchvision.models import ResNet
from dataset.ultra_dataset import UltrasoundDataset
import numpy as np

class Gen_Grad_CAM(Base):
    def __init__(self, trainer_id:int, config:Config, seed:int):
        super().__init__(trainer_id, config, seed)
        self.imput_base_dir = config.img_base_dir
        self.grad_save_dir = join(config.logdir, 'grad_cam')
        self.select_list = config.select_list
        self.img_size = config.img_size

        self.network = get_model(config)

        # for name, param in self.network.named_parameters():
        #     param.requires_grad = False
        #     logging.info("{} require grad={}".format(name, param.requires_grad))

        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
    
    def train_model(self, dataloaders, tblog, valid_epoch=1):
        logging.info('All params before training: {}'.format(count_parameters(self.network)))
        logging.info('Trainable params: {}'.format(count_parameters(self.network, True)))
        pass
    
    def after_train(self, dataloaders, tblog=None):
        # 处理输入图片目录
        input_image_dirs = []
        for item in listdir(self.imput_base_dir):
            class_dir = join(self.imput_base_dir, item) # 得到类似 seperate_dataset/nonstandard 形式的目录
            for image_no in listdir(class_dir):
                input_image_dirs.append(join(class_dir,image_no))

        for input_image_dir in input_image_dirs:
            image_no = basename(input_image_dir)
            image_class = basename(dirname(input_image_dir))
            save_dir = join(self.grad_save_dir,image_class, image_no)
            check_makedirs(save_dir)

            if isinstance(self.network, ResNet):
                target_layer = self.network.layer4[-1]
            else:
                target_layer = None

            self.gen_grad_cam(self.network, target_layer, input_image_dir, save_dir)
            logging.info('处理完成 model{} : {}'.format(self.trainer_id, input_image_dir))
        
        
    def gen_grad_cam(self, model, target_layer, img_dir, output_base_dir):
        """
        输入:model:需要生成 CAM 的模型
            img_dir:一张原图切割出的9张图片保存路径
            
        """
        tf = transforms.Compose([
                # transforms.Resize([224, 224]),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(UltrasoundDataset.mean_gray, UltrasoundDataset.std_gray)])

        cam = GradCAM(model=model,
                    target_layers=[target_layer],
                    use_cuda=True)
        tensor_list = []
        image_list = []
        result_list = []
        name_list = listdir(img_dir)
        name_list.sort()
        # output_dir = join(output_base_dir,'model_%d' % index)
        output_dir = output_base_dir
        check_makedirs(output_dir)
        for img_name in name_list:
            img_position, extension = splitext(img_name)
            if int(img_position) in self.select_list:
                img = Image.open(join(img_dir, img_name))
                image_list.append(img)
                tensor_list.append(tf(img))
        
        input_tensors = torch.cat(tensor_list, dim=0)
        input_tensors = torch.unsqueeze(input_tensors, dim=0)
        # grayscale_cam = cam(input_tensor=input_tensors, target_category=None)
        grayscale_cam = cam(input_tensor=input_tensors)
        
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0,:]

        for idx,img in enumerate(image_list):
            # img = img.resize(self.img_size)
            img = np.float32(img) / 255
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            # 首先保存图片
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(join(output_dir,str(idx)+'.png'), cam_image)
            
            # 接着生成视频
            cv2.putText(cam_image, 'model {} picuture {}'.format(self.trainer_id, idx), (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                    color=(123,222,238), thickness=1, lineType=cv2.LINE_AA)
            # 使用原图大小,即不进行resize
            # cv2.putText(cam_image, 'fold %d picuture %d' % (index,idx), (400, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
            #           color=(123,222,238), thickness=2, lineType=cv2.LINE_AA)
            
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
            result_list.append(cam_image)
            # plt.imshow(cam_image)
            # plt.show()
            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            
        imageio.mimsave(join(output_dir, 'output.gif'),result_list,fps=1)
            
