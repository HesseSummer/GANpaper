[TOC]



# DCGAN论文信息点

## Abstract

1. DCGAN应用于图像数据集
2. 设计DCGAN的最终目的是应用于无监督学习
3. DCGAN不仅能够学习局部的对象，还能学习场景

## 1. Introdution

1. DCGAN如何达到作者希望的目的：***“将其用作监督任务的特征提取器”***
2. GAN的优点：作为最大似然技术的替代方案；GAN的学习过程；GAN没有heuristic cost function (例如pixel-wise independent mean-square error)
3. GAN的缺点：generator的输出不稳定，易为无意义的输出；关于理解及可视化GAN的学习内容的研究甚少；关于多层GAN的中间表示的研究甚少
4. 本论文的贡献：①提出DCGAN架构；②展示DCGAN用于无监督学习；③可视化***GAN学习的过滤器***；④展示generator具有向量计算属性、能操纵生成样本的语义

## 2. Related Work

### 2.1 Representation Learning from unlabeled data

无监督学习的相关研究有：

- 聚类
- 对图像块进行层次聚类
- 自动编码器
- 深度信念网络
- 

### 2.2 Generating natural images

生成图像的模型/方法有：

- 无参模型
  - 与现有图像的数据库匹配
- 含参模型
  - 变分抽样法
  - 迭代向前扩散法
  - GAN及变种
  - 反卷积网络

### 2.3 Visualizing the internals of CNNs

可视化cnn内部的研究有：

- 反卷积、过滤最大激活
- 在输入端使用梯度下降

## 3. Approach and Model Architecture

1. DCGAN采用、修改其他人研究出的优秀做法，来改进原本的GAN：
   1. 在generator和discriminator中，应用全卷积(all convolutional net)，即用跨步卷积(strided convolutions) 替代 池化函数。其优点是：允许网络学习自身的空间降采样。
   2. 在generator的首层和discriminator的尾层，用全局平均池(global average pooling)替代全连接层。优点是提高了稳定性，缺点是降低了收敛速度。
   3. 除了generator的输出层和discriminator的输入层，其他的每一层都应用归一化(batch normal)操作。优点是：有助于处理由于初始化不佳而产生的训练问题、有助于更深的模型中梯度的流动、防止generator***将所有样本折叠到一个点***。不将归一化应用于那两层，是为了避免样本震荡、模型不稳定。
   4. 除了generator的输出层使用tanh作为激活函数外，其他所有层都使用leakyRelu作为激活函数。因为使用有界激活可以使模型更快地学习到颜色表示，leakyRelu对于高分辨率的图片表现好。

## 4. Details of adversarial training

1. 在3个数据集上训练了GAN（LSUN、Imagenet-1k、新组织的Faces数据集）
2. 预处理：仅，将图片缩放到tanh的定义域[-1, 1]中。
3. 反向传播使用mini-batch SGD，且mini-batch为128.
4. 权重初始化：所有权重均按标准正态分布初始化，标准差为0.02
5. 使用Adam优化器，学习率设为0.0002（发现建议值学习率0.001太高了）
6. 将***momentum term β1*** 设置为0.5（发现建议值0.9导致训练震荡）

### 4.1 LSUN

1. 将DCGAN应用于使用LSUN数据集（图片高达300多万、分辨率高）展示其性能
2. 展示了两张图（训练过程中产生的、模型收敛后产生的），表示论文中的DCGAN不是因为过拟合才产生那么好的图像的。

#### 4.1.1 Deduplication

为了降低generator记住输入的图片、生成输入图片的可能性，执行了一个简单的图像重复消除过程：

1. ***32×32 downsampled center-crops of training examples***

2. 在该切片上，施加一个***3072-128-3072的去噪 dropout***

3. 还施加***ReLU自编码器***，编码结果层使用ReLU激活阈值进行二值化。

   它提供了一个语义hash的简单形式，允许在线性时间内进行二分。这个hash编码可视化结果的错误率不超过1%。此技术的召回率也很好，在复制品的选择和删除上逼近275000.

### 4.2 Faces

网上爬的图片、经过openCV人脸检测，包括1万人、35万图片。

### 4.3 Imagenet-1k

无监督学习使用imagenet-1k数据集，取图像正中间32×32的截图来训练。

## 5. Empirical Validation of DCGANs capabilities

### 5.1 Classifying CIFAR-10 using GANs as a feature extractor

1. 如何评估无监督学习算法的性能：将该算法作为特征提取器，应用在有监督的数据集上，计算拟合表现。***具体是怎么做的呢？***
2. ***我们在Image-1K上训练，之后使用判别器所有层上的卷积特征，使用maxpooling将每一层的表征表示为一个4×44×4的方块，这些特征平展后串联，用于表示28672空间向量，然后使用一个正则L2−SVML2−SVM来做分类器，这样做达到了82.8%。***结合代码看这段话！
3. DCGAN比起经典的CNN模型还是有差距
4. 由于我们的DCGAN只在Imagenet-1k上训练过，从来没有在CIFAR-10上训练过，所以这个结果（82.8%的准确率）也表现出算法自身的强大鲁棒性。

### 5.2 Classifying SVHN digits using GANs as a feature extractor

在SVHM数据集上应用DCGAN（作为分类器）：

1. 抽取1万个样本用于调参、模型选择
2. 类似于CIFAR-10，用DCGAN做一个分类器，在SVHM数据集上测试拟合度。
3. 发现测试集上有22.48%的误差。而CNN作为分类器时，CNN在验证集上的误差是28.87%。
4. 证明了DCGAN中使用CNN架构并不是影响模型性能的关键因素

## 6. Investigating and visualizing the internals of the networks

### 6.1 Walking in the latent space

1. 如果隐层中可以看到图像生成的语义变化，比如添加/删除对象，则可以判断该模型已经学习了相关的有趣的表示

### 6.2 Visualizing the Discriminator Features

1. 图5展示了discriminator隐层的样子（左边是随机滤波器基准，右边是前6个卷积特征子的最大轴对称响应）

### 6.3 Manipulating the Generator Representation

#### 6.3.1 Forgetting to draw certain objects

1. 做了一个实验，来展示generator确实能够学习到特征：
   1. 构建一个generator：在从高到低的第二个卷积层上，设置损失时使用logistics，判断输入的图片中有没有画窗户，画了窗户的标记为正，没画的标记为负，训练的目的是舍弃所有正值（即画了窗户的）。
   2. 利用该模型生成图片，发现确实没有画窗户

#### 6.3.2 Vector arithmetic on face samples

1. 向量算数，打个比方就是，网络能够计算出 vector(“国王”) - vector(“男人”) + vector(“女人”)  的最近邻居是 vector(“皇后”)
2. 单例（图7下）上的实验结果并不稳定，但是三样例（图7上）的结果明显具备算术特性。
3. 揭示了脸的姿态在Z空间上是线性的；揭示出无监督得到的模型也能像条件生成模型一样，学习到诸如伸缩，旋转和平移这样的操作。

## 7. Conclusion and Future Work

1. 结论：提出了DCGAN，虽然它仍然存在一些形式的模型不稳定性，但是随着训练时间变长，会有一部分特征子坍塌为一个单震荡的模式。
2. 未来工作：解决不稳定的问题、将该框架拓展到视频（帧预测）、音频领域（语音的预训练）、进一步探索隐层的表现。

## 8. Supplementary Material

### 8.1 Evaluating DCGANs capability to capture data distributions

评估dcgan捕获数据分布的能力：

1. 使用DCGAN和近邻分类器做MNIST数据集上的分类任务
2. 发现，在两个模型中，从batchnorm中删除比例和偏差参数对两个模型都能产生更好的结果。所以推测，batchnorm引入的噪声有助于generator更好地从探索数据分布、生成图片。
3. 在训练集上，DCGAN和近邻分类器一样出色。

