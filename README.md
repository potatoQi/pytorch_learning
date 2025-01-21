最近在学 PyTorch, 写点小项目练练手。

代码在目前我俩台机子均可正常运行，环境配置分别是：

1. python=3.8 + cuda11.5 + conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch + conda install numpy=1.21.6 + conda install pandas=1.4.4
2. ... ...

- data: 存数据集的地方
  - 数据集 github 大小限制传不上来，我把数据集都放我网盘里了

- model: 存模型的地方
  - 同样，大小限制我也没传上来。运行程序就会自动在 model 目录下产生对应模型了

## 跑分
没有完全遵循论文里的参数设置(但是网络架构基本上是遵循了), 因为要适应各种数据集图片大小以及我弱小显卡的性能, 所以这个跑分就图一乐, 这个跑分对参数量大的模型不公平, 因为都是用的小数据集和很少的 epochs 。

不过大模型确实要考虑的更多，小模型随便训训每次结果都差不多。而大模型可能遇到各种问题，我在实测中就遇到了训练无法进行的情况，估计是梯度消失了，loss 一直不变。或者即使训动了 (ResNet), 也会过拟合等等问题。

这些问题目前我就不管了，因为这个仓库就是纯粹练习下简单 PyTorch 的应用的。调参和工程 trick 这些事不是目前这个仓库需要关心的主要矛盾。

- FashionMNIST
  - MLP: loss:0.78  | acc:71.18%
  - LetNet: loss:   0.28 | acc:  89.68%
  - AlexNet: loss:   0.35 | acc:  88.02%
  - ResNet: loss:   0.81 | acc:  76.22%
- CIFAR10
  - MLP: loss:1.94  | acc:31.20%
  - LetNet: loss:   1.09 | acc:  61.01%
  - AlexNet: loss:   1.22 | acc:  56.98%
  - VGGNet: 训不动
  - ResNet: loss:   1.72 | acc:  46.11%
- TinyImagenet
  - LetNet: loss:   3.86 | acc:  16.06%
  - AlexNet: 训不动
  - VGGNet: 训不动
  - ResNet: loss:   5.91 | acc:   5.98%

## 练习

如果以后手生了, 又忘了咋写代码了, 就把这个项目复现一次就行了, 热热手。

- 任务一
  - 读入 MNIST, FashionMNIST, CIFAR10 数据集; 进行 3x3 的可视化; 搭建 MLP 去训练; 保存/加载模型;
- 任务二
  - 读入 FashionMNIST, CIFAR10, TinyImagenet 数据集; 进行TinyImagenet数据集的可视化;
  - 搭建 LetNet, AlexNet, VGGNet, ResNet; 写 train/eval loop; 保存/加载模型