'''
实验发现

1. 固定住加噪后的隐变量 xt, 每次反向采样因为带有随机性, 所以每次生成的图像都不同

2. 固定住原图片, 每次正向加噪得到的 xt 服从标准联合正态分布, 单看一个元素, 也服从标准正态分布

3. 固定住加噪的隐变量 xt, 然后只改变左上角像素的值, 从 -4 ~ 4, 步长 0.1, 每次都去生成长图。发现图之间并不是连续变化的。我并不认为这就代表着隐空间不是连续的了, 而是有可能因为去噪也是有随机性的, 这个随机性直接把位置全弄偏了。
'''

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import make_grid
from einops import repeat

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

config = OmegaConf.load('config.yaml')
device = config.UNet.device
scheduler = instantiate(config.LinearNoiseScheduler)
model = instantiate(config.UNet).to(device)
checkpoint = torch.load('../checkpoints/ddpm.pth', weights_only=True)
model.load_state_dict(checkpoint)
print(f'Model loaded from {config.Others.load_path}')

dataset = datasets.MNIST(
    root='../data',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # 归一化到 [-1, 1]
    ]),
    download=True,
)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

batch = next(iter(dataloader))
images, labels = batch
image = images[4].to(device)    # 数字 9
# 可视化
# image = (image + 1) / 2
# plt.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
# plt.show()

###################################################
num = 56
t_end = config.LinearNoiseScheduler.num_timesteps
t = torch.full((image.shape[0],), t_end-1).to(device)   # [b]
noise = torch.randn_like(image).to(device)
xt = scheduler.add_noise(image, noise, t)
xt = repeat(xt, '1 c h w -> b c h w', b=num)
xt = torch.load('xt_tensor.pt', weights_only=True)      # 相当于固定住加噪后的隐变量

with torch.no_grad():
    for idx in tqdm(range(t_end-1, -1, -1), total=t_end, desc='Sampling'):
        t = torch.full((num,), idx, dtype=torch.long, device=device)
        noise_pred = model(xt, t)
        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t)
    images = torch.clamp(xt, -1., 1.)
    images = (images + 1) / 2

grid = make_grid(images).permute(1, 2, 0).cpu().numpy()
plt.figure(figsize=(8, 8))
plt.imshow(grid)
plt.axis('off')
plt.show()

#######################################################
xt_orig  = torch.load('xt_tensor.pt', weights_only=True)      # 相当于固定住加噪后的隐变量
xt_orig  = xt_orig[0].unsqueeze(0)   # [1 c h w]

# 定义参数扫描范围
start_val = -4.0
end_val = 4.0
step = 0.1
values = np.arange(start_val, end_val + step, step)

results = []

# 对每个修改的值进行采样
for new_val in tqdm(values, desc="Modifying left-top pixel"):
    # 复制一份原始隐变量并修改左上角元素
    xt_mod = xt_orig.clone()
    # 修改 [batch=0, channel=0, row=0, col=0] 的值
    xt_mod[0, 0, 0, 0] = new_val
    
    # 进行反扩散采样，得到最终图片
    xt_current = xt_mod.clone()
    with torch.no_grad():
        for idx in range(config.LinearNoiseScheduler.num_timesteps-1, -1, -1):
            t_tensor = torch.full((xt_current.shape[0],), idx, dtype=torch.long, device=device)
            noise_pred = model(xt_current, t_tensor)
            # 注意 scheduler.sample_prev_timestep 可能返回 tuple，取第一个元素
            xt_current, _ = scheduler.sample_prev_timestep(xt_current, noise_pred, t_tensor)
        # 处理最终输出图片：裁剪到 [-1,1]，然后映射到 [0,1]
        image_out = torch.clamp(xt_current, -1., 1.)
        image_out = (image_out + 1) / 2
    results.append(image_out.cpu())

# 将所有图片拼接成一个大 batch
# 每个 image_out 的形状应为 [1, c, h, w]，这里拼接后形状为 [N, c, h, w]
images_batch = torch.cat(results, dim=0)

# 可视化，设置 nrow 根据你希望每行显示多少张图片
grid = make_grid(images_batch, nrow=10)  # 每行10张图片
plt.figure(figsize=(20, 20))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis('off')
plt.title("反扩散采样结果（左上角像素从 {:.2f} 到 {:.2f}）".format(start_val, end_val))
plt.show()