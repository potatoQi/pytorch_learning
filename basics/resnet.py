import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from draw import draw

# 残差块 (内含 2 个卷积层 + 跳跃连接)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # ⬆️: bias 就是每层输出通道要减去的那个常数, 默认是 True
        # 这里设为 False 是因为后面接着的 nn.BatchNorm2d() 里, 有 beta, gamma 这俩学习参数
        # 其中 beta 就已经是偏置的作用了

        self.bn1 = nn.BatchNorm2d(out_channels)
        # ⬆️: 标准化每一个通道, 标准化就是均值为0, 方差为1
        # 除此之外, 还需注意内置还有学习参数 beta, gamma, 用于调整标准化后的均值和方差
        
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False) # 这一层卷积不会改变 HxW 尺寸, 和通道数
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接 (shortcut means "捷径" )
        self.shortcut = nn.Sequential() # 若尺寸和通道图不变, 也就是前后维度一样, 那么直接加上 input 即可
        if stride != 1 or in_channels != out_channels:  # 若尺寸改变 or 通道数改变
        # ⬆️: 若 stride=1 尺寸是不会变的, 因为此时 stride=1, padding=1, kernel_size=3
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # ⬆️: 通道对齐; 尺寸对齐
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 因为这里有跳跃连接, 所以要用 out, 而不是一直用 x
        out = self.relu(out)
        return out

# 为了适配 CIFAR-10 数据集, 这里的 ResNet 架构略微做了修改
class ResNet(nn.Module):
    # block 是残差块类, num_blocks 是每个 layer 里有多少个残差块
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # BatchNorm2d 接收的 shape 是 [b c h w], 括号里写的是通道数, 即对同一通道的所有批次的数据进行归一化
        # 因此, 对于某个通道 c, 会将 (b h w) 上所有像素视作整体来计算均值和方差, 来归一化
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.in_channels = 64   # 这里是 64 是因为要接着上边的 conv1 的 out_channels

        # 每个layer包含多个残差块
        # 每一个 layer, 都会对通道数翻倍 (out_channels 翻倍), 然后尺寸减半(除了第一层 layer 不会下采样, stride=1)
        self.layer1 = self._make_layer(block, num_blocks[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, num_blocks[1], 128, 2)
        self.layer3 = self._make_layer(block, num_blocks[2], 256, 2)
        self.layer4 = self._make_layer(block, num_blocks[3], 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 把每个通道都变为一个 (1, 1) 的矩阵, 其实就是一个数了此时
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, num_blocks, out_channels, stride):
        layers = []
        # 每一层的第一个残差块负责改变尺寸和通道数翻倍
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels # 这里记得更新一下, 衔接起来
        # 后续的残差块不改变尺寸和通道
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        
        return logits
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)     # ResNet18

    # ResNet 模型的泛化能力比较差, 不做数据增强的话正确率太差
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=test_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 画出前 4 张图片
    images, labels = next(iter(train_loader))
    draw(size=(2, 2), images=images[:4], labels=labels[:4])

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logits = model(images)  # [b c]
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    model.eval()
    total = 0
    correct = 0
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)  # [b c]
            pred = torch.argmax(logits, dim=-1) # [b]    
            total += labels.shape[0]
            correct += (pred == labels).sum().item()
    print(f"Accuracy: {correct / total:.4f}")