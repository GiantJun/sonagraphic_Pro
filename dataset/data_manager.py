import numpy as np
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset
from dataset import myTransform
import logging
from dataset.myTransform import TwoCropsTransform, GaussianBlur
from dataset.ultra_dataset import UltrasoundDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import train_test_split
import logging


img_size_global = (100,100)

def get_dataloader(args):
    global img_size_global
    img_size_global = args['img_size']

    dataset = get_data(args['dataset'])
    dataset.download_data(args['select_list'])
    
    target = dataset.train_dataset.target
    train_dataloaders, valid_dataloaders, test_dataloaders = [], [], []
    
    if args['kfold'] > 1: # kfold 默认在 kfold 中划分验证集
        kfold = StratifiedKFold(n_splits=args['kfold'], shuffle=True, random_state=0)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(np.arange(len(target)), target)):
            # 再将训练集划分为训练集和选择参数的验证集
            ids_labels = np.array([target[i] for i in train_ids])
            train_ids, valid_ids, _, _  = train_test_split(train_ids, ids_labels, test_size=0.1, random_state=0, stratify=ids_labels)

            logging.info('Fold {}: valid indexs {}'.format(fold, valid_ids))
            logging.info('Fold {}: test indexs {}'.format(fold, test_ids))
            logging.info('-' * 50)

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)

            # trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_dataloaders.append(DataLoader(dataset.train_dataset, batch_size=args['batch_size'], num_workers=args['num_workers']))
            valid_dataloaders.append(DataLoader(dataset.train_dataset, batch_size=args['batch_size'], num_workers=args['num_workers']))
            # 注意此处dataloader加载的是验证集数据
            test_dataloaders.append(DataLoader(dataset.test_dataset, batch_size=args['batch_size'], num_workers=args['num_workers']))
    else:
        if args['split_for_valid']:
            train_ids, valid_ids, _, _  = train_test_split(np.arange(len(target)), target, test_size=0.1, random_state=0, stratify=target)

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)

            # trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_dataloaders.append(DataLoader(dataset.train_dataset, batch_size=args['batch_size'], sampler=train_subsampler, num_workers=args['num_workers']))
            valid_dataloaders.append(DataLoader(dataset.train_dataset, batch_size=args['batch_size'], sampler=valid_subsampler, num_workers=args['num_workers']))
            # 注意此处dataloader加载的是验证集数据，作为内部测试集
            test_dataloaders.append(DataLoader(dataset.valid_dataset, batch_size=args['batch_size'], num_workers=args['num_workers']))
        else:
            train_dataloaders.append(DataLoader(dataset.train_dataset, batch_size=args['batch_size'], num_workers=args['num_workers']))
            valid_dataloaders.append(DataLoader(dataset.valid_dataset, batch_size=args['batch_size'], num_workers=args['num_workers']))
            test_dataloaders.append(DataLoader(dataset.test_dataset, batch_size=args['batch_size'], num_workers=args['num_workers']))

    return {'train':train_dataloaders, 'valid':valid_dataloaders, 'test':test_dataloaders}, dataset.class_num, dataset.class_names


def get_data(dataset_name):
    name = dataset_name.lower()
    if name == 'ultrasound_dataset':
        return Sonagraph()
    else:
        raise ValueError('Unknown dataset {}.'.format(dataset_name))

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

class iData(object):
    train_tf = None
    test_tf = None
    class_num = None

class Sonagraph(iData):
    # 训练数据集对象
    global img_size_global

    train_tf = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            myTransform.AddPepperNoise(0.95, p=0.5),
            transforms.Resize(img_size_global),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.3505138, 0.21441448)])  # 此处的 mean 和 std 由

    test_tf = transforms.Compose([
            transforms.Resize(img_size_global),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.3505138, 0.21441448)])

    def download_data(self, select_list):
        # or replay os.environ['xxx'] with './data/'
        self.train_dataset = UltrasoundDataset(os.environ['ALTRASOUND7_12'], transform=self.train_tf, select_list=select_list, dataset_type='train')
        self.valid_dataset = UltrasoundDataset(os.environ['ALTRASOUND7_12'], transform=self.test_tf, select_list=select_list, dataset_type='valid')
        self.test_dataset = UltrasoundDataset(os.environ['ALTRASOUND7_12'], transform=self.test_tf, select_list=select_list, dataset_type='test')

        self.class_num = self.train_dataset.class_num
        self.class_names = self.train_dataset.class_names
        
       

def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return np.array(img)