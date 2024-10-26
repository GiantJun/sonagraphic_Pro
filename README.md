# 分类框架
## 目前已经实现的接口（功能）
+ k-fold validation
+ grad-cam
+ model ensembles
+ DP（多GPU）

## 环境准备
```
pip install -r ./requirements.txt
```

## 训练模型
下面以使用 finetune 方法训练模型为例，说明如何使用
```
python main.py --config options/finetune.yaml
```
为了方便，以及避免运行程序时输入过多参数，你可以通过指定初始化参数文件（在option目录下），对实验的参数进行初始化（更多参数可见utils/config.py）

## 测试模型
模型训练完后，会在 log 文件夹下记录训练过程中的输出及相关信息，如：
>> logs/finetune/resnet50_all_dataset1
>>     |--events.out.tfevents.1659611018    //tensorboard 输出
>>     |--XXX.log  //log信息，XXX为方法名，指代运行该方法时的输出
>>     |--trainer0_model_dict_150.pkl   //保存的模型权重

使用指定模型权重进行测试时，可以使用如下命令
```
python main.py --config options/test.yaml --pretrain_path=logs/finetune/resnet50_all_dataset1/trainer0_model_dict_150.pkl
```
命令执行完毕后，将会在目录下生成 test.log 、混淆矩阵（ROC曲线，分类错误图片路径文件XXX_mistake.csv）。同理，test.yaml 为参数初始化文件，--pretrain_path 为在命令行中指定加载预训练模型权重文件路径（也可在test.yaml中指定，但会优先使用命令行指定的方式）

生成 CAM 图像的方式与测试类似。

## 其他数据处理脚本
./crop_sub_picture.py —— 自动识别并框出子图边缘、打印子图的坐标参数，方便截取子图

./dataset/data_preprocess.py —— 依照子图坐标参数，批量截取完整图片数据中的子图、归类存储子图

./rename.py —— 批量修改文件名

./cp_error_img.py —— 复制模型分类错误的图片到指定目录