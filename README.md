# Jittor 草图生成风景比赛 SPADE-Jittor

本项目以 CVPR2019 的 SPADE 模型作为基础框架，采用 Jittor 实现，[相关论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.pdf) 为 Semantic Image Synthesis with Spatially-Adaptive Normalization，部分结果展示如下：

![image.png](https://s2.loli.net/2022/06/27/MNADbPY1pjsZl2w.png)

![1031937_222ee03f9f_b](https://s2.loli.net/2022/06/27/TVrKgtyW7J5esN2.jpg)

## 简介

本项目为第二届计图挑战赛计图 - 草图生成风景比赛 A 榜的参赛项目，包含了全部的代码实现。本项目的特点是：使用 Jittor 框架以 SPADE 模型为基础进行复线，通过强化学习的方法对原始语义分割图进行处理，生成风景图片。

## 安装

本项目可在单张 GPU 上运行。

#### 运行环境

* python >= 3.7
* jittor >= 1.3.4

#### 安装依赖

执行以下命令安装 python 依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

请从 [比赛官网](https://www.educoder.net/competitions/index/Jittor-3) 或 [这里](https://cloud.tsinghua.edu.cn/f/70195945f21d4d6ebd94/?dl=1) 下载数据训练集以及测试集。请将解压后的数据放入 `<root>/data` 下。

文件结构组织如下: 

```
SPADE-Jittor
--data
 |--train
   |--imgs
   |--labels
 |--val
   |--labels
--options
--scripts
--spade
--util
__init__.py
datasets.py
pix2pix_model.py
pix2pix_trainer.py
test.py
train.py
README.md
requirements.txt
```

## 训练与测试

项目内包含了用于训练模型以及在官方测试集上测试模型的脚本。

训练模型，请执行 `scripts` 文件夹下的 `train.sh` ：

```bash
cd scripts
bash train.sh
```

测试模型，请执行 `scripts` 文件夹下的 `test.sh` ：

```bash
cd scripts
bash test.sh
```

其中的关键参数如下：

```bash
--name [the name of current training / test on which trained model] \
--n_epoch [total number of epoch in training] \
--batch_size [batch size of training / testing] \
--lr [learning rate] \
--save_epoch_freq [save the model every $~ epochs] \
```

其他超参数  `--no_instance`， `--preprocess_mode`， `--aspect_ratio` 以及 `--label_nc`，用于适配当前数据集，因而无需更改。 更多超参数及其默认值可参考 `options`。

## 参考文献

[1] Park T, Liu M Y, Wang T C, et al. Semantic image synthesis with spatially-adaptive normalization[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 2337-2346. 

[2]  Zhou X, Zhang B, Zhang T, et al. Cocosnet v2: Full-resolution correspondence learning for image translation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 11465-11475. 

[3]  Qu Y, Chen Y, Huang J, et al. Enhanced pix2pix dehazing network[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 8160-8168. 
