import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from draw import draw

# 原本的 AlexNet 是为 224x224 的 ImageNet 数据集设计的，但是 CIFAR-10 数据集的图片大小为 32x32，所以需要对模型进行一定的修改。
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # 输入通道为 3，输出通道为 64，卷积核大小为 3 3，填充为 2
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),   # [b c h w] -> [b (c*h*w)], nn.Flatten() 默认参数屎 为 start_dim=1, end_dim=-1
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes=10).to(device)
    
    # 因为 CIFAR-10 数据集的图片大小为 32x32，所以 AlexNet 架构需要进行一定的修改
    train_dataset = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 画出前 16 张图片
    images, labels = next(iter(train_loader))
    draw(size=(4, 4), images=images[:16], labels=labels[:16])

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logits = model(images)  # [b c]
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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