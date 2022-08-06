from torch.utils.data import Dataset
from os.path import join, splitext
from os import listdir
from torch import cat
from PIL import Image
import logging
from pytorch_grad_cam.utils.image import show_cam_on_image

class UltrasoundDataset(Dataset):
    """自定义 Dataset 类"""
    # 预先运行本文件计算好的 mean 和 std
    mean_RGB = (0.3215154, 0.32233492, 0.3232533)
    std_RGB = (0.1990278, 0.19854955, 0.19809964)
    mean_gray = 0.3505138
    std_gray = 0.21441448

    img_size = (175,300) # 读入数据时的图片尺寸
    
    def __init__(self, root, transform=None, select_list=list(range(0,9)),dataset_type='train', ret_path=False, ret2Views=False):
        """
        输入：root_dir：数据子集的根目录
        dataset_type如果为“test”则会获取测试集数据
        说明：此处的root_dir 需满足如下的格式：
        root_dir/
        |-- class0
        |   |--image_group0
        |   |   |--1.png
        |   |   |--2.png
        |   |   |--3.png
        |   |--image_group2
        |   |   |--1.png
        |   |   |--2.png
        |-- class1
        """
        self.transform = transform
        self.return2View = ret2Views
        
        self.data = []
        self.target = []

        self.select_list = select_list  # 选择9幅图中的几幅（从0开始计）
        
        self.ret_path = ret_path

        self.class_names = None
        self.class_num = None
        
        if dataset_type == 'all':
            root_dir = join(root,'train')
            # 训练集测试集下，各类别名默认为一致
            self.class_names = listdir(root_dir)
            self.class_names.sort(reverse=True) # 0-standard, 1-nonstandard
            self.class_num = len(self.class_names)
            for idx, item in enumerate(self.class_names):   # idx 相当于类别标号，此处0-nonstandard,1-standard
                for type in ['train', 'valid', 'test']:
                    root_dir = join(root,type)
                    # 训练集测试集下，各类别名默认为一致
                    class_dir = join(root_dir, item)
                    img_dirs = listdir(class_dir)
                    # img_dirs.sort()
                    self.data.extend([join(class_dir, image_dir) for image_dir in img_dirs])
                    self.target.extend([idx for i in range(len(img_dirs))])
        elif dataset_type in ['train', 'valid', 'test'] or 'test' in dataset_type:
            root_dir = join(root,dataset_type)
            self.class_names = listdir(root_dir)
            self.class_names.sort(reverse=True) # 0-standard, 1-nonstandard
            self.class_num = len(self.class_names)
            
            for idx, item in enumerate(self.class_names):   # idx 相当于类别标号，此处0-nonstandard,1-standard
                class_dir = join(root_dir, item)
                img_dirs = listdir(class_dir)
                # img_dirs.sort()
                self.data.extend([join(class_dir, image_dir) for image_dir in img_dirs])
                self.target.extend([idx for i in range(len(img_dirs))])
                logging.info('{}-{}-{} : {}'.format(dataset_type, idx, self.class_names[idx], len(img_dirs)))
        else:
            raise ValueError('No dataset type {} , dataset type must in [\'train\',\'test\']'.format(dataset_type))
        
        
        
    # def get_XY(self):
    #     x, y = zip(*(self.item_list))
    #     return x, y

    def __len__(self):
        """返回 dataset 大小"""
        return len(self.target)

    def set_transforms(self, tf):
        self.transform = tf

    def __getitem__(self, idx):
        """根据 idx 从 图片名字-类别 列表中获取相应的 image 和 label"""
        img_dir, label = self.data[idx], self.target[idx]
        name_list = listdir(img_dir)
        tensor_list = []
        tensor_list2 = []
        for item in name_list:
            img_position, extension = splitext(item)
            if int(img_position) in self.select_list:    # 只读入相应位置的图片
                img = Image.open(join(img_dir,item))
                tensor_list.append(self.transform(img))
                if self.return2View:
                    tensor_list2.append(self.transform(img))
        result = cat(tensor_list, dim=0)
        if self.return2View:
            result = [result, cat(tensor_list2, dim=0)]
        if self.ret_path:
            return result, label, img_dir
        else:
            return result, label