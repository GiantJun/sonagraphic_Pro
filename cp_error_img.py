from os.path import join, basename, exists, dirname
import csv
from utils.toolkit import check_makedirs
from shutil import copy, rmtree

# 每次只生成一个数据集的分类错误图片
######## 括约肌(dataset1) ########
# csv_path = 'XXXXX/valid_mistake.cvs'
# origin_img_dir = '/home/2021/yujun/Storage/Data/括约肌7月13日/内部验证集'

# csv_path = 'XXXXX/test_mistake.cvs'
# origin_img_dir = '/home/2021/yujun/Storage/Data/括约肌7月13日/外部验证—湖南妇幼'

######## 肛提肌(dataset2) ########
# csv_path = 'logs/test/multi_branch_cat_all_dataset2/valid_mistake.cvs'
# origin_img_dir = '/home/2021/yujun/Storage/Data/肛提肌7月25日/外部验证—佛山'

csv_path = 'logs/test/multi_branch_cat_all_dataset2/test_mistake.cvs'
origin_img_dir = '/home/2021/yujun/Storage/Data/肛提肌7月25日/外部验证—湖南妇幼'


save_dir = join(dirname(csv_path), 'error_img')
# 操作前先删除生成目录
if exists(save_dir):
    rmtree(save_dir)
    print('已清理生成目录!')

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







