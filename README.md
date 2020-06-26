# 交通标识分类
> author: Luvoy

这是一个多分类

## 流程&思路
0. 将一些超参数存放在一个文件中, 用的时候方便读取
   - 这里用了一个py文件, 其他py文件用```import```方式读取 
1. 先对数据进行了标注, 并生成csv文件
2. 用csv文件, 生成数据集
   - 定义了```MyDataset```类
   - 每条数据格式: ```{'image': image, 'class_id': class_id}``` 
   - transform操作在此文件完成, 包括**放缩**和**翻转**
   - dataloader也直接生成了, 用的话直接```import```
3. 验证数据集, 用显示图片和类别的方式可视化验证
   - 用```matplotlib```制作了一个显示器, 可以点击按钮来显示上一张下一张, 可以随机也可以按顺序
4. 搭建网络模型
   - 包括ResNet18和老项目那个简单的网络 
   - FC层返回的维度和分类的总数有关
5. 训练
   - 优化器SGD
   - Loss交叉熵
   - 借鉴了老项目的结构
   - 训练完成会将Loss下降过程可视化
   - 训练时会及时保存一个日志文件
   - 训练完会保存模型
6. 预测
   - 加载已保存的模型
   - 用test的数据集进行predict




## 文件结构

![add image](https://github.com/Luvoy/Traffic_Sign_Classification/raw/master/README_images/1.png)

## 训练结果

### 结果概述
- 之前自己定义了一个网络, 训练了50epoch准确率还是40多, 所以放弃了, 不予展示了
- 50Epoch
- 训练时间: 10h
- 最终准确率: 97.7%左右
- predict时, 2462张通过, 58张预测错误
  - 而且我发现predict的数据集, shuffle了,错误更少

### Loss和Accuracy走势情况:

![add image](https://github.com/Luvoy/Traffic_Sign_Classification/raw/master/README_images/2.png)

## 配置

python version = 3.7.6

torch version  = 1.4.0

CUDA version = 10.2

GPU: GTX 960m

CPU: i7-6700HQ

MEMORY: 16GB

Platform: Ubuntu 18.04
