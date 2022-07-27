import os
import re

data_dir = '/media/huangyujun/disk/data/盆底质控数据/括约肌/7月13日/中大补图6.22/外部验证集/非标准'
start_id = 178
origin_name_list = os.listdir(data_dir)

# 给文件名中数字加一个基数
for item in origin_name_list:
    name, suffix = item.split('.')
    origin_path = os.path.join(data_dir,item)
    target_path = os.path.join(data_dir, "{}.{}".format(start_id+int(name), suffix))
    os.system('mv {} {}'.format(origin_path, target_path))

# 重命名为只有数字
# for item in origin_name_list:
#     num_in_name = re.search('\d+', item).group()+'.bmp'
#     print(item+' -> '+num_in_name)
#     origin_path = os.path.join(data_dir,item)
#     target_path = os.path.join(data_dir, num_in_name)
#     os.system("mv '{}' {}".format(origin_path, target_path))

# 检查文件名称是否符合格式
# for item in origin_name_list:
#     if re.match("标准（[0-9]{1,4}）.bmp", item):
#         continue
#     elif re.match("标准\([0-9]{1,4}）.bmp", item):
#         continue
#     elif re.match("标准（[0-9]{1,4}\).bmp", item):
#         continue
#     elif re.match("标准\([0-9]{1,4}\).bmp", item):
#         continue
#     elif re.match("标准 \([0-9]{1,4}\).bmp", item):
#         continue
#     else:
#         print(item+' -> null')