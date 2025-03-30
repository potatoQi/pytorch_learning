import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 这个包以前叫 pytorch_lightning, 现在叫 lightning
import lightning as L
from lightning import seed_everything

class LitAutoEncoder(L.LightningModule):    # lightning 类型的 AE, 继承 L.LightningModule
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
        )
    
    # training_step 是 lightning model 必写的一个函数
    def training_step(self, batch, batch_idx):
        x, _ = batch    # x: [b c h w]
        x = self.flatten(x)    # x: [b, c*h*w]
        z = self.encoder(x)
        x_hat = self.decoder(z) # x_hat: [b, c*h*w]
        loss = F.mse_loss(x_hat, x)
        # 下面这句话是直接 log 到 tensorboard
        self.log("train_loss", loss)
        return loss
    
    # configure_optimizers 是 lightning model 必写的一个函数
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
    # 如果传入了 val_dataloader, 那么就要实现一下 validation_step
    # lightning 默认每个 epoch 训练完就会跑一遍验证集
    def validation_step(self, batch, batch_idx):
        x, _ = batch    # x: [b c h w]
        x = self.flatten(x)    # x: [b, c*h*w]
        z = self.encoder(x)
        x_hat = self.decoder(z) # x_hat: [b, c*h*w]
        loss = F.mse_loss(x_hat, x)
        # prog_bar=True: 在进度条上显示 val_loss
        # on_epoch=True: tensorboard 里只记录验证集执行完的指标
        self.log("val_loss", loss, prog_bar=True)
        return loss

if __name__ == '__main__':
    autoencoder = LitAutoEncoder()  # 不用 .to(device), lightning 会自动处理(如果有 gpu 会放到 gpu 上)

    train_dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    val_dataset = datasets.MNIST(
        root='../data',
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 设置随机种子, 保证 torch, numpy, random 的随机性, workers=True 让 dataloader 多进程加载数据时也保证可重复性
    seed_everything(42, workers=True)
    # 准备训练对象
    trainer = L.Trainer(
        limit_train_batches=None,       # 只训练 x 个 batch 
        max_epochs=10,                  # 只训练 10 个 epoch
        deterministic=True,             # 尽可能保证底层算子的可重复性, 尽可能使得能够复现 (即便如此，由于硬件差异、线程调度、浮点舍入等原因，仍然有些情况下无法保证 100% 完全复现。)
    )
    # 开训
    trainer.fit(
        model=autoencoder,                  # 必须是继承 L.LightningModule 的模型
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 验证模型 (相当于跑一遍验证集, 调用的是 model.validation_step 方法), 会在控制台打印结果
    trainer.validate(model=autoencoder, dataloaders=val_dataloader)

    # 测试模型 (相当于跑一遍测试集, 调用的是 model.test_step 方法), 会在控制台打印结果
    # trainer.test(model=autoencoder, dataloaders=test_dataloader)

