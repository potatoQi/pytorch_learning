'''
其实 ddpm 很简单。

首先先把 NoiseScheduler 写好, 这个类就负责两件事, 按照公式写就好了:
    给它 x0, noise, t, 它返回 xt (即前向加噪)
    给它 xt, t, noise_pred, 它返回 x_{t-1} (即反向去噪)

然后再把 UNet 的仨组件写好。

即把 DownBlock, MidBlock, UpBlock 写好。按照图去搭积木就好了。有几个注意点：
    1. 每个 resnetblock 是由俩卷积构成的, 注意第一个卷积后要插一个时间 emb。
    2. resnetblock, attentionblock 都要有残差连接

接着开始搭 UNet, 搭 UNet 的时候最好画画图手玩一下, 把 "分辨率", "通道数" 这两个关键点要对齐好。以及注意在 forward 里要把数字时间步 t 变为 temb。
'''

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime, os
import matplotlib.pyplot as plt
from get_anime_dataloader import get_anime_dataloader

def get_time_embedding(timesteps, temb_dim):
    # timesteps: [b]
    # temb_dim: []

    assert temb_dim % 2 == 0, "temb_dim must be even"
    factor = 10000 ** (torch.arange(0, temb_dim // 2, dtype=torch.float32, device=timesteps.device) / (temb_dim // 2))   # [temb_dim//2]
    factor = rearrange(factor, 'd -> 1 d')    # [1 temb_dim//2]

    t_emb = repeat(timesteps, 'b -> b l', l=temb_dim//2) / factor # [b temb_dim//2]
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)  # [b temb_dim]
    return t_emb

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end, device):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = (1. - self.betas).to(device)
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod).to(device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod).to(device)

    def add_noise(self, x0, noise, t):
        # x0: [b c h w]
        # noise: [b c h w]
        # t: [b]

        B = x0.shape[0]
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t] # [b]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t] # [b]

        sqrt_alpha_cumprod = rearrange(sqrt_alpha_cumprod, 'b -> b 1 1 1')
        sqrt_one_minus_alpha_cumprod = rearrange(sqrt_one_minus_alpha_cumprod, 'b -> b 1 1 1')

        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        # xt: [b c h w]
        # noise_pred: [b c h w]
        # t: [b]

        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t] # [b]
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t]    # [b]
        betas = self.betas[t]    # [b]
        alphas = self.alphas[t]  # [b]
        alpha_cumprod_1 = self.alpha_cumprod[t-1]    # [b]
        alpha_cumprod_2 = self.alpha_cumprod[t]    # [b]

        sqrt_one_minus_alpha_cumprod = rearrange(sqrt_one_minus_alpha_cumprod, 'b -> b 1 1 1')
        sqrt_alpha_cumprod = rearrange(sqrt_alpha_cumprod, 'b -> b 1 1 1')
        betas = rearrange(betas, 'b -> b 1 1 1')
        alphas = rearrange(alphas, 'b -> b 1 1 1')
        alpha_cumprod_1 = rearrange(alpha_cumprod_1, 'b -> b 1 1 1')
        alpha_cumprod_2 = rearrange(alpha_cumprod_2, 'b -> b 1 1 1')

        x0_pred = (xt - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod    # [b c h w]
        if torch.all(t == 0):
            return x0_pred, x0_pred

        mean = xt - betas * noise_pred / sqrt_one_minus_alpha_cumprod
        mean = mean / torch.sqrt(alphas)    # [b c h w]

        variance = betas * (1. - alpha_cumprod_1) / (1. - alpha_cumprod_2)    # [b 1 1 1]
        sigma = torch.sqrt(variance)    # [b 1 1 1]

        z = torch.randn_like(xt).to(xt.device)    # [b c h w]
        return mean + sigma * z, x0_pred
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, down_sample, num_heads, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_layers)
        ])
        self.temb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(temb_dim, out_channels),
            ) for _ in range(num_layers)
        ])
        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ])
        self.residual_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()

    def forward(self, x, temb):
        # x: [b c h w]
        # temb: [b temb_dim]
        for i in range(self.num_layers):
            res = x
            x = self.resnet_conv_1[i](x)    # [b out_channels h w]
            x = x + self.temb_layers[i](temb).unsqueeze(-1).unsqueeze(-1)    # [b out_channels h w]
            x = self.resnet_conv_2[i](x)    # [b out_channels h w]
            x = self.residual_conv[i](res) + x    # [b out_channels h w]

            b, c, h, w = x.shape
            res = x
            x = self.norm_layers[i](x)    # [b out_channels h w]
            x = rearrange(x, 'b c h w -> b (h w) c')    # [b h*w out_channels]
            x = self.attention_layers[i](x, x, x)[0]    # [b h*w out_channels]
            x = rearrange(x, 'b (h w) c -> b c h w', h=h)    # [b out_channels h w]
            x = res + x    # [b out_channels h w]

        x = self.down_sample_conv(x)    # [b out_channels h/2 w/2]
        return x
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, num_heads, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers + 1)  # +1 for the first layer
        ])
        self.temb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(temb_dim, out_channels),
            ) for _ in range(num_layers + 1)
        ])
        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers + 1)
        ])

        self.residual_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers + 1)
        ])

        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(8, out_channels)
            for _ in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, x, temb):
        # x: [b c h w]
        # temb: [b temb_dim]
        res = x
        x = self.resnet_conv_1[0](x)
        x = x + self.temb_layers[0](temb).unsqueeze(-1).unsqueeze(-1)
        x = self.resnet_conv_2[0](x)
        x = self.residual_conv[0](res) + x

        for i in range(self.num_layers):
            b, c, h, w = x.shape
            res = x
            x = self.norm_layers[i](x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.attention_layers[i](x, x, x)[0]
            x = rearrange(x, 'b (h w) c -> b c h w', h=h)
            x = res + x

            res = x
            x = self.resnet_conv_1[i + 1](x)
            x = x + self.temb_layers[i + 1](temb).unsqueeze(-1).unsqueeze(-1)
            x = self.resnet_conv_2[i + 1](x)
            x = self.residual_conv[i + 1](res) + x
        
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, up_sample, num_heads, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1) if up_sample else nn.Identity()

        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_layers)
        ])
        self.temb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(temb_dim, out_channels),
            ) for _ in range(num_layers)
        ])
        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers)
        ])

        self.residual_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(8, out_channels)
            for _ in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, x, out_down, temb):
        # x: [b in_channels//2 h//2 w//2]
        # out_down: [b in_channels//2 h w]
        # temb: [b temb_dim]

        x = self.up_sample_conv(x)  # [b in_channels//2 h w]
        x = torch.cat([x, out_down], dim=1) # [b in_channels h w]

        for i in range(self.num_layers):
            res = x
            x = self.resnet_conv_1[i](x)
            x = x + self.temb_layers[i](temb).unsqueeze(-1).unsqueeze(-1)
            x = self.resnet_conv_2[i](x)
            x = self.residual_conv[i](res) + x

            b, c, h, w = x.shape
            res = x
            x = self.norm_layers[i](x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.attention_layers[i](x, x, x)[0]
            x = rearrange(x, 'b (h w) c -> b c h w', h=h)
            x = res + x

        return x

class UNet(nn.Module):
    def __init__(
        self,
        device='cuda',
        im_channels=3,
        temb_dim=16,

        down_channels=[32, 64, 128],  # 表示三个 downblock 的输入通道数
        down_sample=[True, True, False],
        mid_channels=[256, 256, 256], # 表示三个 midblock 的输入通道数

        num_heads=4,
        num_down_layers=1,
        num_mid_layers=1,
        num_up_layers=1,
    ):
        super().__init__()
        self.im_channels = im_channels
        self.device = device
        self.temb_dim = temb_dim

        self.t_proj = nn.Sequential(
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        self.conv_in = nn.Conv2d(im_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        self.downs = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i] * 2,
                temb_dim=temb_dim,
                down_sample=down_sample[i],
                num_heads=num_heads,
                num_layers=num_down_layers
            ) for i in range(len(down_channels))
        ])
        self.mids = nn.ModuleList([
            MidBlock(
                in_channels=mid_channels[i],
                out_channels=mid_channels[i+1] if i != len(mid_channels) - 1 else down_channels[-1],
                temb_dim=temb_dim,
                num_heads=num_heads,
                num_layers=num_mid_layers
            ) for i in range(len(mid_channels))
        ])
        self.ups = nn.ModuleList([
            UpBlock(
                in_channels=down_channels[i] * 2,
                out_channels=down_channels[i-1] if i != 0 else down_channels[0],
                temb_dim=temb_dim,
                up_sample=down_sample[i],
                num_heads=num_heads,
                num_layers=num_up_layers
            ) for i in range(len(down_channels)-1, -1, -1)  # 从最后 len(down_channels)-1 ~ 0
        ])

        self.norm = nn.GroupNorm(8, down_channels[0])
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(down_channels[0], im_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        # x: [b c h w]
        # t: [t]

        temb = get_time_embedding(t, self.temb_dim)    # [b temb_dim]
        temb = self.t_proj(temb)    # [b temb_dim]
        x = self.conv_in(x) # [b down_channels[0] h w]

        down_outs = []
        for block in self.downs:
            down_outs.append(x)
            x = block(x, temb)

        for block in self.mids:
            x = block(x, temb)

        for block in self.ups:
            x = block(x, down_outs.pop(), temb)

        # x: [b down_channels[0] h w]
        x = self.norm(x)
        x = self.silu(x)
        x = self.conv_out(x)    # [b im_channels h w]

        return x
    
    @torch.no_grad()
    def sample(
        self,
        scheduler,
        im_channels=3,
        hw=64,
        sample_num=3,
        t_end=1000,
    ):
        xt = torch.randn(sample_num, im_channels, hw, hw).to(self.device)
        x0_pred_list = []

        for idx in tqdm(range(t_end-1, -1, -1), total=t_end, desc='Sampling'):
            t = torch.full((sample_num,), idx, dtype=torch.long, device=xt.device)
            noise_pred = self(xt, t)
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t)

            x0_pred = torch.clamp(x0_pred, -1., 1.)
            x0_pred = (x0_pred + 1) / 2
            x0_pred_list.append(x0_pred)

        images = torch.clamp(xt, -1., 1.)
        images = (images + 1) / 2
        return images, x0_pred_list
    

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')

    # 数据集
    dataset_name = config.Dataset.name
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(
            root='../data',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),   # 归一化到 [-1, 1]
            ]),
            download=True,
        )
        test_dataset = datasets.MNIST(
            root='../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
            download=True,
        )
        train_dataloader = DataLoader(train_dataset, config.Dataset.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, config.Dataset.batch_size, shuffle=False)
    elif dataset_name == 'anime-faces':
        train_dataloader, test_dataloader = get_anime_dataloader(batch_size=config.Dataset.batch_size)

    # 调度器 & 模型
    device = config.UNet.device
    scheduler = instantiate(config.LinearNoiseScheduler)
    model = instantiate(config.UNet).to(device)

    # 训练
    if config.Others.load_path:
        checkpoint = torch.load(config.Others.load_path, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f'Model loaded from {config.Others.load_path}')
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join('runs', timestamp)
        writer = SummaryWriter(log_dir)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.Train.lr)
        loss_fn = nn.MSELoss()
        for epoch in range(config.Train.epochs):
            model.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.Train.epochs}")
            for step, batch in pbar:
                optimizer.zero_grad()

                x = batch[0].to(device)    # [b c h w]
                noise = torch.randn_like(x).to(device)  # [b c h w]
                t = torch.randint(0, config.LinearNoiseScheduler.num_timesteps, (x.shape[0],)).to(device)   # [b]

                xt = scheduler.add_noise(x, noise, t)   # [b c h w]
                noise_pred = model(xt, t)

                loss = loss_fn(noise_pred, noise)
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix(loss=loss.item())
                writer.add_scalar('train_loss', loss.item(), epoch * len(train_dataloader) + step)
            
            model.eval()
            loss_avg = 0
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Test {epoch+1}/{config.Train.epochs}")
            for step, batch in pbar:
                with torch.no_grad():
                    x = batch[0].to(device)
                    noise = torch.randn_like(x).to(device)
                    t = torch.randint(0, config.LinearNoiseScheduler.num_timesteps, (x.shape[0],)).to(device)
                    xt = scheduler.add_noise(x, noise, t)
                    noise_pred = model(xt, t)
                    loss_avg += loss_fn(noise_pred, noise)
                    pbar.set_postfix(loss=(loss_avg.item() / len(test_dataloader)))
            writer.add_scalar('test_loss', loss_avg.item() / len(test_dataloader), epoch)

            with torch.no_grad():
                _, x0_pred_list = model.sample(
                    scheduler=scheduler,
                    im_channels=config.UNet.im_channels,
                    hw=config.Dataset.hw,
                    sample_num=config.Eval.sample_num,
                    t_end=config.Eval.t_end,
                )
                # x0_pred_list: [list], 有 t_end 个元素, 每个元素为 [0, 1] 范围的 tensor: [b c h w]
                for i in range(config.Eval.t_end):
                    writer.add_images(f'images/{epoch+1}', x0_pred_list[i], config.Eval.t_end - i)

        os.makedirs(config.Others.save_path, exist_ok=True)
        save_path = os.path.join(config.Others.save_path, 'ddpm.pth')
        torch.save(model.state_dict(), save_path)
        print(f'Model saved in {save_path}')
        writer.close()

    # 推理
    model.eval()
    images, _ = model.sample(
        scheduler=scheduler,
        im_channels=config.UNet.im_channels,
        hw=config.Dataset.hw,
        sample_num=config.Eval.sample_num,
        t_end=config.Eval.t_end,
    )
    grid = make_grid(images).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()