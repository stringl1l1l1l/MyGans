# 准备工作
1. 按照README.md, 进行配置
2. 下载数据集，文件夹改名为raw_data，路径为data/raw_data/pos和data/raw_data/pos
3. 执行prepare.py进行数据的预处理，raw2jpg()将原始图像统一为RGB格式，jpg2tensor()将jpg转成tensor格式，并保证图片tensor大小[3, H, W]为三通道
4. 以cgan.py为例，我用data/cup_plus/dataset.py里的Cup类定义了奖杯数据集，人工指定了label，可参照该格式进行改写
5. 其他操作详见README.md，需要注意的是我改了项目结构，这是为了能顺利导入Cup类

关于cup_plus
1. pos类去除了奇形怪状的，黑色背景的，歪的，非居中的，重复的奖杯（高质量）
2. 图像分辨率较高，需要选择合适的预处理，改进网络结构（高分辨率）

傻瓜式
1. 下数据集, 名字改成raw_data
2. 预处理
   自己电脑的内存不够，因此对数据集做一个resize，顺便统一下分辨率
```
python
cd data/cup_plus
python3 prepare.py
cd ../../
```
1. 训练
```
python
python3 wgan.py
```

# WGAN炼丹心得

## train1
使用默认的超参数，数据集分辨率统一为128*128，训练1000轮，但随着训练轮次的增大效果反而变差。

## train2
发现生成器最后的tanh层会把像素限制到[-1,1], 遂将生成器最后的tanh层删除，减小lr为0.00002，batch_size为8，生成分辨率为64*64，训练1000轮，取得了不错的效果。 

#### 训练结果  
![147200 batch](images/goodres/2/147200.png)

## train3
本次尝试大训练轮次，并减小生成图片的分辨率到32*32，试试量变能否引起质变，结果效果比前几次训练都要好，奖杯的轮廓逐渐清晰，奖杯的形状也比较多样，
但是在训练到8000轮次左右时，生成图像的质量趋近稳定，猜测此时模型已经接近收敛，判别器的损失函数值在[-1,1]范围振荡，生成器的损失函数在[-10,10]范围振荡，
此时图像虽然逐步成形，但依旧比较模糊。
此外生成器会尝试变化亮度以欺骗判别器，我认为这可能会妨碍生成器生成奖杯的多样性，所以是否可以对生成图片的亮度做一个限制？或者使用HSI模型训练？

#### 训练结果(1w~13w batch)  

![](images/goodres/3/100000.png)
![](images/goodres/3/200000.png)
![](images/goodres/3/300000.png)
![](images/goodres/3/400000.png)

![](images/goodres/3/500000.png)
![](images/goodres/3/600000.png)
![](images/goodres/3/700000.png)
![](images/goodres/3/800000.png)

![](images/goodres/3/900000.png)
![](images/goodres/3/1000000.png)
![](images/goodres/3/1100000.png)
![](images/goodres/3/1200000.png)

![](images/goodres/3/1300000.png)

#### 超参数  
```
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=10000, help="interval between image samples")
```  

## train4
第三次训练似乎证明"大力出奇迹"，于是本次训练继续加大轮次至2W，并调整生成分辨率为64*64，尝试生成更清晰的图像，其他参数不变，跑了一晚上，没想到在20000批和200000批生成的图像质量居然没什么差别，甚至还更差了。  

#### 训练结果
![20000 batch](images/bad/4/20000.png)  
20000 batch

![149200 batch](images/bad/4/2330000.png)  
2330000 batch

猜测有可能的原因：  
1. 潜在空间维度过小
2. 学习率太大

#### 超参数  
```
parser.add_argument("--n_epochs", type=int, default=20000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=10000, help="interval between image samples")
```  

### train5
根据第四次训练经验，本次训练调整了潜在空间大小，继续尝试生成64*64分辨率的图像，下面是训练的结果展示：

### 结果展示


### train6
前几次训练的网络架构使用的还是全连接层，最后的效果也是一言难尽，因此本次训练尝试更换了网络架构，参考DCGAN，
生成器采用三层反卷积+LN+ReLU再加一层tanh架构，判别器采用三层卷积层+LN+LeakyReLU。训练结果确实比原来更清晰，
但效果仍然不好，特别是在长时间训练过后，发现判别器的损失函数始终在-1上下波动，却始终小于0，这说明判别器过强，训练难以进行。

### train7
继续改进网络架构，将LN层替换为BN层，并扩充了一些高质量的数据集。对于判别器过强，生成器过弱的问题，
我猜测是由于梯度消失导致的，WGAN梯度裁剪无法解决这一问题，因此尝试使用WGAN-GP，梯度惩罚的方式做正则化，
该方法的训练效果是目前为止最好的。

但是GP的权重超参数如何选择又是一个问题，本次训练我选择了10作为权重参数，结果在训练到10w批次的时候，生成器突然崩了，生成出一堆噪声图像，D loss变成了一个极大的负数。

#### 超参数  
这组超参是目前效果最好的
```
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=0.005, help="generator learning rate")
parser.add_argument("--lr_d", type=float, default=0.005, help="discriminator learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=10000, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image samples")
parser.add_argument("--model_interval", type=int, default=10000, help="interval between model samples")
parser.add_argument("--model_depth", type=int, default=3, help="depth of model's net")
parser.add_argument("--gp_weight", type=int, default=10, help="gradient penalty weight")
parser.add_argument("--b1", type=float, default=0, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
```




### train12

这次跑到8924轮，尽管损失函数很稳定，最后还是生成了一堆噪声图像，可以看到此时G loss已经很大了
[Epoch 8924/100000] [Batch 15/133] [D loss: 12.595483] [G loss: 4765408.000000]
[Epoch 8924/100000] [Batch 20/133] [D loss: -3.912184] [G loss: 4765635.500000]
[Epoch 8924/100000] [Batch 25/133] [D loss: 6.527401] [G loss: 4765576.000000]
[Epoch 8924/100000] [Batch 30/133] [D loss: -23.478380] [G loss: 4765754.000000]
[Epoch 8924/100000] [Batch 35/133] [D loss: -11.447525] [G loss: 4765677.000000]
[Epoch 8924/100000] [Batch 40/133] [D loss: -9.967720] [G loss: 4765659.500000]
[Epoch 8924/100000] [Batch 45/133] [D loss: -6.349782] [G loss: 4765589.500000]
[Epoch 8924/100000] [Batch 50/133] [D loss: -17.848280] [G loss: 4765641.500000]
[Epoch 8924/100000] [Batch 55/133] [D loss: -6.949971] [G loss: 4765597.500000]
[Epoch 8924/100000] [Batch 60/133] [D loss: -1.480281] [G loss: 4765685.000000]
[Epoch 8924/100000] [Batch 65/133] [D loss: -6.868345] [G loss: 4765728.000000]
[Epoch 8924/100000] [Batch 70/133] [D loss: -2.923069] [G loss: 4765566.000000]
[Epoch 8924/100000] [Batch 75/133] [D loss: -1.481089] [G loss: 4765335.000000]
[Epoch 8924/100000] [Batch 80/133] [D loss: 9.014521] [G loss: 4765289.500000]
[Epoch 8924/100000] [Batch 85/133] [D loss: -29.439777] [G loss: 4765282.000000]
[Epoch 8924/100000] [Batch 90/133] [D loss: -46.476624] [G loss: 4765298.000000]
[Epoch 8924/100000] [Batch 95/133] [D loss: -18.470098] [G loss: 4765179.000000]
[Epoch 8924/100000] [Batch 100/133] [D loss: -6.422441] [G loss: 4765119.000000]
[Epoch 8924/100000] [Batch 105/133] [D loss: -13.307665] [G loss: 4765222.500000]
[Epoch 8924/100000] [Batch 110/133] [D loss: -3.416743] [G loss: 4765458.500000]
[Epoch 8924/100000] [Batch 115/133] [D loss: -4.972701] [G loss: 4765306.000000]
[Epoch 8924/100000] [Batch 120/133] [D loss: -6.873494] [G loss: 4765138.000000]
[Epoch 8924/100000] [Batch 125/133] [D loss: -7.450010] [G loss: 4765272.000000]
[Epoch 8924/100000] [Batch 130/133] [D loss: -10.917979] [G loss: 4765218.000000]
[Epoch 8925/100000] [Batch 0/133] [D loss: 14.615849] [G loss: 4765203.500000]
[Epoch 8925/100000] [Batch 5/133] [D loss: -4.442653] [G loss: 4765181.500000]
[Epoch 8925/100000] [Batch 10/133] [D loss: -6.969006] [G loss: 4765371.000000]
[Epoch 8925/100000] [Batch 15/133] [D loss: -2.907098] [G loss: 4765205.500000]
[Epoch 8925/100000] [Batch 20/133] [D loss: 6.759264] [G loss: 4765253.000000]
[Epoch 8925/100000] [Batch 25/133] [D loss: -55.443371] [G loss: 4765353.500000]
[Epoch 8925/100000] [Batch 30/133] [D loss: 7.076761] [G loss: 4765086.000000]
[Epoch 8925/100000] [Batch 35/133] [D loss: 28.162670] [G loss: 4765164.000000]
[Epoch 8925/100000] [Batch 40/133] [D loss: 47.741386] [G loss: 4765057.500000]
[Epoch 8925/100000] [Batch 45/133] [D loss: -1.473396] [G loss: 4765051.000000]
[Epoch 8925/100000] [Batch 50/133] [D loss: -11.967417] [G loss: 4764900.500000]
[Epoch 8925/100000] [Batch 55/133] [D loss: 11.614349] [G loss: 4765287.500000]
[Epoch 8925/100000] [Batch 60/133] [D loss: 16.389233] [G loss: 4765031.000000]
[Epoch 8925/100000] [Batch 65/133] [D loss: 7.016325] [G loss: 4765163.000000]
[Epoch 8925/100000] [Batch 70/133] [D loss: -18.928215] [G loss: 4765032.000000]
[Epoch 8925/100000] [Batch 75/133] [D loss: 4.028571] [G loss: 4764911.500000]
[Epoch 8925/100000] [Batch 80/133] [D loss: 22.282904] [G loss: 4765143.500000]


ckpt15  290000 FID：91
ckpt151 300000 