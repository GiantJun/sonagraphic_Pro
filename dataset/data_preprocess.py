import numpy as np
import os
import cv2
import re
from random import sample
import random
import numpy as np
from tqdm import tqdm

# (520, 1000), (816, 1512), (528, 1000), (820, 1332), (712, 1340), (768, 1312) (1020, 1660)
# 输出的图片尺寸
img_w = 300
img_h = 175
output_shape = (img_w, img_h)
# 一下列表的顺序是从上到下，从左到右，对9张图进行编号排列的
# point表示子图片的左上角坐标,crop_size表示裁剪的尺寸

# 原始图片尺寸为 1000x520 ,合格117
point_1000x520 = [(32, 60), (355, 60), (678, 60), (32, 214), (355, 214), (678, 214), (32, 368), (355, 368), (678, 368)]
crop_size_1000x520 = (322,152)
# 原始图片尺寸为 1132x820, 合格120 ***
point_1332x820 = [(32, 60), (465, 60), (898, 60), (32, 314), (465, 314), (898, 314), (32, 568), (465, 568), (898, 568)]
crop_size_1332x820 = (432,252)
# 原始图片尺寸为 1492x640 ?
point_1492x640 = [(176,64),(603,81),(1090,82),(116,278),(603,278),(1090,277),(116,469), (603,465), (1090,460)]
crop_size_1492x640 = (306,167)
# 原始图片尺寸为 2300X1252 ?
point_2300x1252 = [(32, 60), (788, 60), (1544, 60), (32, 458), (788, 458), (1544, 458), (32, 856), (788, 856), (1544, 856)]
crop_size_2300x1252 = (755,396)
# 原始图片尺寸为 1000x528
point_1000x528 = [(32,58), (355,58), (678,58), (32,215), (355,215), 
        (678,215), (32,372), (355,372), (678,372)]
crop_size_1000x528 = (321,155)
# 原始图片尺寸为 1340x700
point_1340x700 = [(32,58),(469,58),(904,58),(32,272),(468,272),(904,272),(32,486),(468,486),(904,486)]
crop_size_1340x700 = (434,212)
# 原始图片尺寸为 1340x712
point_1340x712 = [(32,58),(468,58),(904,58),(32,276),(468,276),(904,276),(32,494),(468,494),(904,494)]
crop_size_1340x712 = (434,216)
# 原始图片尺寸为 1512x816
point_1512x816 = [(32,58),(526,58),(1020,58),(32,311),(526,311),(1020,311),(32,564),(526,564),(1020,564)]
crop_size_1512x816 = (492,251)
# 原始图片尺寸为 1856x1132
point_1856x1132 = [(32,58),(640,58),(1248,58),(32,416),(640,416),(1248,416),(32,774),(640,774),(1248,774)]
crop_size_1856x1132 = (606,356)
# 原始图片尺寸为 1312x768, 合格292
point_1312x768 = [(32,58),(459,58),(886,58),(32,295),(459,295),(886,295),(32,532),(459,532),(886,532)]
crop_size_1312x768 = (425,235)
# 原始图片尺寸为 1424x772, 不合格1040
point_1424x772 = [(32,58),(496,58),(960,58),(32,296),(496,296),(960,296),(32,534),(496,534),(960,534)]
crop_size_1424x772 = (463,236)
# 原始图片尺寸为 1660x1020, 中大补图 合格4
point_1660x1020 = [(32,58),(575,58),(1118,58),(32,379),(575,379),(1118,379),(32,700),(575,700),(1118,700)]
crop_size_1660x1020 = (541,319)
# 原始图片尺寸为 1632x1008, ***
point_1632x1008 = [(32,58), (565,58),(1098,58),(32,375),(565,375),(1098,375),(32,692),(565,692),(1098,692)]
crop_size_1632x1008 = (532,315)
# 原始图片尺寸为 1580x1072, ***
point_1580x1072 = [(32,60), (547,60),(1062,60),(32,398),(547,398),(1062,398),(32,735),(547,735),(1062,735)]
crop_size_1580x1072 = (514,336)
# 原始图片尺寸为 1140x730, ***
point_1140x730 = [(32, 60), (401, 60), (770, 60), (32, 283), (401, 283), (770, 283), (32, 506), (401, 506), (770, 506)]
crop_size_1140x730 = (368,221)
# 原始图片尺寸为 1856x1116, ***
point_1856x1116 = [(32, 60), (638, 60), (1244, 60), (32, 412), (638, 412), (1244, 412), (32, 764), (638, 764), (1244, 764)]
crop_size_1856x1116 = (605,350)
# 原始图片尺寸为 1552x964, ***
point_1552x964 = [(32, 60), (538, 60), (1044, 60), (32, 362), (538, 362), (1044, 362), (32, 664), (538, 664), (1044, 664)]
crop_size_1552x964 = (505,300)
# 原始图片尺寸为 720x576, 括约肌断裂-标准7.JPG
point_720x576 = [(46, 72), (257, 72), (468, 72), (46, 226), (257, 226), (468, 226), (46, 380), (257, 380), (468, 380)]
crop_size_720x576 = (210,152)
# 原始图片尺寸为 1136x852, 括约肌断裂-标准1.jpg
point_1136x852 = [(28, 67), (397, 67), (766, 67), (28, 326), (397, 326), (766, 326), (28, 585), (397, 585), (766, 585)]
crop_size_1136x852 = (368, 257)
# 原始图片尺寸为 968X708, 括约肌断裂-标准1.jpg
point_968X708 = [(25, 58), (340, 58), (655, 58), (25, 275), (340, 275), (655, 275), (25, 492), (340, 492), (655, 492)]
crop_size_968X708 = (968, 708)

# 设定随机种子
random.seed(100)

def cropImage_Save(img_list, save_dir):
    img_shape_list = []
    for img_path in tqdm(img_list):
        # 获取原始图片的序号名，作为该图片截取图片的保存目录
        img_name = os.path.basename(img_path)
        dir_name = re.search('\d+',img_name).group()
        save_path = os.path.join(save_dir,dir_name) # 截取图片的保存路径
        img = cv2.imread(img_path)   # 这里无需将BGR转换为RGB，因为后面还要保存
        img_shape_list.append(img.shape)
        
        if img.shape[0] == 520 and img.shape[1] == 1000:
            points = point_1000x520
            crop_size = crop_size_1000x520
        elif img.shape[0] == 820 and img.shape[1] == 1332:
            points = point_1332x820
            crop_size = crop_size_1332x820
        elif img.shape[0] == 640 and img.shape[1] == 1492:
            points = point_1492x640
            crop_size = crop_size_1492x640
        elif img.shape[0] == 1252 and img.shape[1] == 2300:
            points = point_2300x1252
            crop_size = crop_size_2300x1252
        elif img.shape[0] == 528 and img.shape[1] == 1000:
            points = point_1000x528
            crop_size = crop_size_1000x528
        elif img.shape[0] == 700 and img.shape[1] == 1340:
            points = point_1340x700
            crop_size = crop_size_1340x700
        elif img.shape[0] == 712 and img.shape[1] == 1340:
            points = point_1340x712
            crop_size = crop_size_1340x712
        elif img.shape[0] == 816 and img.shape[1] == 1512:
            points = point_1512x816
            crop_size = crop_size_1512x816
        elif img.shape[0] == 1132 and img.shape[1] == 1856:
            points = point_1856x1132
            crop_size = crop_size_1856x1132
        elif img.shape[0] == 772 and img.shape[1] == 1424:
            points = point_1424x772
            crop_size = crop_size_1424x772
        elif img.shape[0] == 768 and img.shape[1] == 1312:
            points = point_1312x768
            crop_size = crop_size_1312x768
        elif img.shape[0] == 1020 and img.shape[1] == 1660:
            points = point_1660x1020
            crop_size = crop_size_1660x1020
        elif img.shape[0] == 1008 and img.shape[1] == 1632:
            points = point_1632x1008
            crop_size = crop_size_1632x1008
        elif img.shape[0] == 1072 and img.shape[1] == 1580:
            points = point_1580x1072
            crop_size = crop_size_1580x1072
        elif img.shape[0] == 730 and img.shape[1] == 1140:
            points = point_1140x730
            crop_size = crop_size_1140x730
        elif img.shape[0] == 1116 and img.shape[1] == 1856:
            points = point_1856x1116
            crop_size = crop_size_1856x1116
        elif img.shape[0] == 964 and img.shape[1] == 1552:
            points = point_1552x964
            crop_size = crop_size_1552x964
        elif img.shape[0] == 576 and img.shape[1] == 720:
            points = point_720x576
            crop_size = crop_size_720x576
        elif img.shape[0] == 852 and img.shape[1] == 1136:
            points = point_1136x852
            crop_size = crop_size_1136x852
        elif img.shape[0] == 708 and img.shape[1] == 968:
            points = point_968X708
            crop_size = crop_size_968X708
        else :
            print('error size in {} , img_size={}'.format(img_path,img.shape))
            continue
        # 保存切割图片
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for idx,(x,y) in enumerate(points):    
            w, h = crop_size
            crop = img[y:y+h,x:x+w]
            crop = cv2.resize(crop, output_shape)
            cv2.imwrite(os.path.join(save_path, str(idx))+'.png', crop)
        
        # 预先检查切割结果
        # for idx,(x,y) in enumerate(points):    
        #     w, h = crop_size
        #     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        # cv2.imwrite(os.path.join(save_dir, img_name), img)

    print(set(img_shape_list))

def process_imgs_split(base_dir, save_base_dir, class_name, split_rate):

    train_save_dir = os.path.join(os.path.join(save_base_dir,'train'), class_name)
    test_save_dir = os.path.join(os.path.join(save_base_dir,'test'), class_name)

    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    file_list = list(os.path.join(base_dir,item) for item in os.listdir(base_dir))

    train_list = sample(file_list, int(len(file_list)*split_rate))
    valid_list = list(set(file_list).difference(train_list))
    cropImage_Save(train_list,train_save_dir)
    cropImage_Save(valid_list,test_save_dir)

def process_imgs(base_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(base_dir)
    print(save_dir)
    file_list = list(os.path.join(base_dir,item) for item in os.listdir(base_dir))
    cropImage_Save(file_list,save_dir)


if __name__ == '__main__':

    # base_dir = '原始数据集'
    # split_names = ["train", "valid"]
    # split_rate = 0.9
    
    base_dir1 = "/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/训练集/标准"
    save_dir1 = '/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/ultrasound1_7-13/train/standard'
    base_dir2 = "/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/训练集/非标准"
    save_dir2 = '/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/ultrasound1_7-13/train/nonstandard'
    print('*'*10+'processing test standard images'+'*'*10)
    process_imgs(base_dir1, save_dir1)
    print('*'*10+'processing test nonstandard images'+'*'*10)
    process_imgs(base_dir2, save_dir2)

    base_dir1 = "/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/外部验证—佛山/标准"
    save_dir1 = '/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/ultrasound1_7-13/test1/standard'
    base_dir2 = "/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/外部验证—佛山/非标准"
    save_dir2 = '/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/ultrasound1_7-13/test1/nonstandard'
    print('*'*10+'processing test standard images'+'*'*10)
    process_imgs(base_dir1, save_dir1)
    print('*'*10+'processing test nonstandard images'+'*'*10)
    process_imgs(base_dir2, save_dir2)

    base_dir1 = "/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/外部验证—湖南妇幼/标准"
    save_dir1 = '/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/ultrasound1_7-13/test2/standard'
    base_dir2 = "/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/外部验证—湖南妇幼/非标准"
    save_dir2 = '/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/ultrasound1_7-13/test2/nonstandard'
    print('*'*10+'processing test standard images'+'*'*10)
    process_imgs(base_dir1, save_dir1)
    print('*'*10+'processing test nonstandard images'+'*'*10)
    process_imgs(base_dir2, save_dir2)



############# 图片尺寸 #############
# 训练集：{(772, 1424, 3), (640, 1492, 3), (1132, 1856, 3), (520, 1000, 3), (816, 1512, 3), (700, 1340, 3),
#         (528, 1000, 3), (712, 1340, 3), (1252, 2300, 3), (820, 1332, 3), (768, 1312, 3)}
# 
# 验证集：{(520, 1000, 3), (700, 1340, 3)}
# 
# 测试集：{(520, 1000, 3), (700, 1340, 3)}