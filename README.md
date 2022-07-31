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

## 运行方式
```
python main.py --config options/XXX.yaml
```
options目录下保存了 finetune 等初始化参数（更多参数可见utils/config.py），程序运行结果保存在 logs 文件夹中