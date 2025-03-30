import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from draw import draw

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x: [b 1 28 28]
        x = self.flatten(x) # x: [b 784]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(num_classes=10).to(device)

    train_dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_dataset = datasets.MNIST(
        root='../data',
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 画出前 3 张图片
    images, labels = next(iter(train_loader))
    draw(size=(1, 3), images=images[:3], labels=labels[:3])

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)   # [b 10]
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{10}, Loss: {loss.item()}")

    model.eval()
    correct = 0
    total = 0
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)   # [b 10]
        pred = logits.argmax(dim=-1)    # [b]
        total += y.shape[0]
        correct += (pred == y).sum().item()
    print(f"Accuracy: {correct / total:.4f}")