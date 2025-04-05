import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class AnimeDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode

        full_dataset = load_dataset("parquet", data_dir="../data/anime-faces", split="train")
        split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
        
        if mode == 'train':
            self.dataset = split_dataset['train']
        else:
            self.dataset = split_dataset['test']
        
        # 调整大小到 (128,128) 并转为 Tensor
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),  # 转换为 Tensor，自动将像素值归一化到 [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1,1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]  # image 是一个 PIL.Image 对象
        image = self.transform(image)
        return image, 0  # 返回图片和 0, 为了和 MNIST 数据集对齐

def get_anime_dataloader(batch_size=32):
    dataset_train = AnimeDataset(mode='train')
    dataset_test = AnimeDataset(mode='test')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_anime_dataloader(batch_size=32)
    batch = next(iter(train_loader))
    print(batch[0].shape)  # 应该输出 (32, 3, 128, 128)
    # 可视化一张图片
    image = batch[0][0]  # 获取第一张图片
    image = image.permute(1, 2, 0)  # 转换为 HWC 格式
    image = (image + 1) / 2  # 反归一化到 [0,1]
    plt.imshow(image.numpy())
    plt.axis('off')
    plt.show()