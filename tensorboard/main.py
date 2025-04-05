'''
tensorboard 的基本用法其实很简单:

    注意, tensorboard 为了节约资源会限制显示对象的数量, 这就会造成有时候某迭代时的数据不显示, 别担心, 用命令
    tensorboard --logdir=xxx --samples_per_plugin=images=1000 即可
    代表 IMAGES 页面最多显示 1000 张图片, 其它页面 (如 SCALARS) 也可以这样设置

    from torch.utils.tensorboard import SummaryWriter

    1. 创建一个 SummaryWriter 对象, 指定一个文件夹路径, 用于存放 tensorboard 的数据
        writer = SummaryWriter(save_path)

    2. 几种常用方法

        指标
        writer.add_scalar(
            tag=<str>,
            scalar_value=<value>,
            global_step=<int>,
        )
        writer.add_scalars(
            main_tag=<str>,
            tag_scalar_dict=<dict>,    # 例如: {"train": 0.25, "validation": 0.30}
            global_step=<int>,
        )
        图片
        writer.add_image(
            tag=<str>,
            img_tensor=<torch.Tensor>,  # [c h w]
            global_step=<int>,
        )
        writer.images(
            tag=<str>,
            img_tensor=<torch.Tensor>,  # [b c h w]
            global_step=<int>,
        )
        writer.add_figure(
            tag=<str>,
            figure=<matplotlib.figure.Figure>,
            global_step=<int>,
        )
        视频
        writer.add_video(
            tag=<str>,
            vid_tensor=<torch.Tensor>,  # [n t c h w]
            global_step=<int>,
            fps=<int>,
        )
        计算图
        writer.add_graph(
            model=<torch.nn.Module>,
            input_to_model=<torch.Tensor>,
        )
        投影图
        writer.add_embedding(
            mat=<torch.Tensor>,         # 二维 tensor [b d], 每一行是一个样本的特征向量
            metadata=<list>,            # [b], 每个样本的标签
            label_img=<torch.Tensor>,   # [b c h w], 每个样本的图片
        )

    3. 最后记得 writer.close() 关闭 writer
'''

from einops import rearrange
from utils import matplotlib_imshow, select_n_random, plot_classes_preds
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([            # transformers.Compose([xx, xx]): 将多个 transformers 组合在一起
    transforms.ToTensor(),                  # 将 PIL.Image 或 numpy.ndarray 转换为 tensor, 并且归一化到 [0, 1]
    transforms.Normalize((0.5,), (0.5,))    # 逐元素进行 (x - mean) / std 操作, 即将数据归一化到 [-1, 1]
    # 多通道写法: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.FashionMNIST(
    '../data',
    download=True,
    train=True,
    transform=transform
)
testset = torchvision.datasets.FashionMNIST(
    '../data',
    download=True,
    train=False,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 1. 创建 tensorboard writer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/fashion_mnist_experiment_1')   # 此时会创建一个 runs/fashion_mnist_experiment_1 文件夹

# 2. 画一些图到 tensorboard 里: writer.add_image()
images, labels = next(iter(trainloader))
img_grid = torchvision.utils.make_grid(images)  # .make_grid: 将多张图片的 tensor 在 w 维度上拼接成一张图片, -> [c h w*n]
matplotlib_imshow(img_grid, one_channel=True)

writer.add_image('four_fashion_mnist_images', img_grid) # 插入一个名为 'four_fashion_mnist_images' 的图片 (图片是 tensor 类型)

# 3. 可视化模型数据流动过程到 tensorboard 里: writer.add_graph()
writer.add_graph(net, images)   # images 这个 tensor 会作为输入, 进入到 net 的 forward 中

# 4. 可视化列向量集合到 tensorboard 里: writer.add_embedding()
images, labels = select_n_random(trainset.data, trainset.targets)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
class_labels = [classes[lab] for lab in labels]

features = rearrange(images, 'b h w -> b (h w)')
writer.add_embedding(
    mat=features,                   # features: 二维 tensor, 每一行是一个样本的特征
    metadata=class_labels,          # metadata: 每个样本的标签
    label_img=images.unsqueeze(1),  # label_img: 特性向量对应的图片 [b c h w]
)

# 5. 绘制训练指标和阶段性成果到 tensorboard 里: writer.add_scalar(), writer.add_figure()
running_loss = 0.0
criterion = nn.CrossEntropyLoss()
print('Start Training')
for epoch in range(1):
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if i % 1000 == 999:    # 每 1000 个 batch 打印一次指标
            writer.add_scalar(
                tag='training loss',
                scalar_value=running_loss / 1000,
                global_step=epoch * len(trainloader) + i
            )
            running_loss = 0.0
            # writer.add_figure() 与 writer.add_image() 的区别就是
            # add_image 是直接把 tensor 作出图片输出, add_figure 是把 matplotlib 对象输出
            writer.add_figure(
                tag='predictions vs. actuals',
                figure=plot_classes_preds(net, inputs, labels),
                global_step=epoch * len(trainloader) + i
            )
print('Finished Training')

# 关闭 tensorboard writer
writer.close()