from os.path import join, basename, exists, dirname
import csv
from utils.toolkit import check_makedirs
from shutil import copy

csv_path = 'logs/test/efficientnet_b5_all_dataset2/valid_mistake.cvs'
origin_img_dir = '/home/2021/yujun/Storage/Data/肛提肌7月25日/外部验证—佛山'

save_dir = join(dirname(csv_path), 'error_img')
check_makedirs(join(save_dir, '非标准'))
check_makedirs(join(save_dir, '标准'))
with open(csv_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for line in reader:
        error_path = line['Path']
        img_id = basename(error_path)
        img_class = '标准' if basename(dirname(error_path))=='standard' else '非标准'
        
        if exists(join(origin_img_dir, img_class, img_id+'.bmp')):
            origin_img_path = join(origin_img_dir, img_class, img_id+'.bmp')
        elif exists(join(origin_img_dir, img_class, img_id+'.jpg')):
            origin_img_path = join(origin_img_dir, img_class, img_id+'.jpg')
        else:
            print('Could not find {}-{} in origin dir!'.format(img_class, img_id))
            continue
        
        copy(origin_img_path, join(save_dir, img_class))







