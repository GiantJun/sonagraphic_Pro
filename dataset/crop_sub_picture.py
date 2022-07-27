# # 数据增广实验
# ## 1.读取图像
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# in_path = "./原始数据集/standard/正常1.bmp" # 较大尺寸的图片
in_path = "/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/括约肌7月13日/内部验证集/非标准/220.bmp" # 较小尺寸图片
# in_path = "/media/huangyujun/disk/workspace/Altrasound_pro/seperated_dataset/standard/1/2.png"

img = cv2.imread(in_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 按照边缘截取图片
_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY) 
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
points = []
head = (32, 60)
crop_size = (322,152)
h_gap = 2
w_gap = 1

# 确定大致轮廓，获得左上顶点
# for idx in range(len(contours)):
#     temp = img.copy()
#     x,y,w,h = cv2.boundingRect(contours[idx])
    
#     # 选择合适的图片，并输出对应的序号
#     if h > 130 and w >50 :
#         crop = img[y:y+h,x:x+w]
#         print('left top point:({},{}) , right bottom point({},{}) , picture shape:{} '.format(x,y,x+w,y+h,crop.shape))
#         cv2.drawContours(temp,contours,idx,(0,0,255),3)
#         cv2.rectangle(temp, (x,y), (x+w,y+h), (0,255,0), 2)
#         plt.imshow(temp)
#         plt.show()
#     else:
#         continue

for i in range(9):
    row = i // 3
    column = i % 3
    w, h = crop_size
    cv2.rectangle(img, (head[0]+(w+w_gap)*column,head[1]+(h+h_gap)*row), (head[0]+(w+w_gap)*column+w,head[1]+(h+h_gap)*row+h), (0,255,0), 2)
    cv2.putText(img, str(i+1), (head[0]+(w+1)*column+100,head[1]+(h+1)*row+100),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, 
                    color=(255,0,0), thickness=5, lineType=cv2.LINE_AA)
    points.append((head[0]+(w+w_gap)*column,head[1]+(h+h_gap)*row))

print(img.shape)
print(points)
plt.imshow(img)
plt.show()


# cv2.imwrite(os.path.join(save_dir, img_name), img)



