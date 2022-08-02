# 分类框架
## 目前已经实现的接口（功能）
+ k-fold validation
+ grad-cam
+ model ensembles
+ DP

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
模型训练完后，会在 log 文件夹下记录训练过程中的输出及相关信息
