# kaggle cifar10 图像分类比赛

## 项目介绍

在本项目中，大家会学会使用卷积网络进行kaggle cifar10的图像分类，我们给定一个非常简单的 resnet18 的baseline，大家可以使用任意的网络结构进行训练和测试，通过该项目，我们能够对调参有更好的了解。

## 项目下载

打开终端，运行

```
git clone https://github.com/L1aoXingyu/kaggle-cifar10.git
```

进行项目的下载或者通过网页版下载

## 数据下载
通过[比赛界面](https://www.kaggle.com/c/cifar-10)根据图片中的显示进行数据下载

<div align=center>
<img src='https://ws1.sinaimg.cn/large/006tNbRwly1fwai15kmgvj31he13aq5k.jpg' width='500'>
</div>

然后在项目的根目录中创建`data`文件夹，将下载好的4个文件放入`data`中，接着运行下面的命令来得到预处理之后的数据

```
sudo apt-get install p7zip
cd data;
p7zip -d train.7z;
p7zip -d test.7z;
cd ..; python3 preprocess.py;
```

这里可能需要等待比较久的时间

## 训练 baseline
运行下面的代码

```
python3 train.py --bs=128 --use_gpu=True
```

就可以进行baseline训练, 其中 `bs` 表示 batch size, `use_gpu` 表示是否使用 gpu，还有一些额外的参数，请阅读 `train.py`，在训练过程中，会自动创建`checkpoints`文件夹，训练的模型会自动保存在`checkpoints`中。

## 提交结果
训练完成 baseline 之后，我们的模型会保存在 `checkpoints` 中，我们可以 load 我们想要的模型，进行结果的提交，运行下面的代码

```
python3 submission.py --model_path='checkpoints/model_best.pth.tar'
```

我们会在本地创建一个预测的结果 `submission.csv`，我们将这个文件提交到 kaggle，可以得到类似下面的比赛结果。

<div align=center>
<img src='https://ws2.sinaimg.cn/large/006tNbRwly1fwa9n14kymj30rf061aa5.jpg' width='800'>
</div>