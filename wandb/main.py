'''
这个 wandb 挺 cool 的。
个人觉得, 做大型实验/项目的时候, 一定要用 wandb。wandb 又简单又强大。缺点就是需要外网, 上传/加载速度比较慢, 需要保证网络畅通。
    网络的问题有两种解决方式:
        1. wandb.init(mode="offline"), 数据会保存到本地, 然后把 logs 文件夹 scp 到本地, 然后 wandb sync <run_directory>
        2. 用 docker 在服务器上部署 wandb
tensorboard 个人认为会用就行, 跟 git 一样, 有手册就行, 不需要熟练的掌握。
下面的 wandb 教程目前只涉及最基础的几个功能: 写指标、写图片、写视频、写表格、写模型参数和梯度。
还有一些高级功能, 比如
    Reports(轻松编写实验报告)
    Sweeps(自动调参)
    Artifact(项目版本管理)
    这些后续需要用到再学。
'''

import torch
import torch.nn as nn
import wandb
import random
import numpy as np
import datetime

wandb.login()   # 或者通过 export WANDB_API_KEY=<your_api_key> 的方式登录

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 开始一次 wandb run
run = wandb.init(
    # Team 名称
    entity="Error_666",
    # 项目名称
    project="project0",
    # 运行名称 (便于在运行列表中识别)
    name=timestamp,
    # 运行的 ID (便于在域名中识别, id 不能重复, 重复会有奇怪的 bug)
    id=timestamp,
    # 要记录的 metadata     可以通过 run.config.update() 追加
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)


# 创建一个 Table，包含 epoch、ground truth 和 prediction 三列 (下面会用到)
comparison_table = wandb.Table(columns=["epoch", "Ground Truth", "Prediction"])

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # 记录指标
    run.log({"acc": acc, "loss": loss}, step=epoch)

    # 记录图像
    # 这里生成一个 100x100 像素的随机 RGB 图像
    random_image = np.uint8(np.random.rand(20, 20, 3) * 255)
    # 将 NumPy 数组转换为 wandb.Image 对象，并添加描述
    image = wandb.Image(
        data_or_path=random_image,      # 可以是 NumPy 数组 (h w c)、PIL 图像、文件路径
        caption=f"random image",
    )
    run.log({"one image": image}, step=epoch)

    image_list = []
    image_list.append(image)    # 添加 wandb.Image 对象到列表
    image_list.append(image)    # 添加 wandb.Image 对象到列表
    run.log({"two images": image_list}, step=epoch)

    # 记录视频
    frames = np.random.randint(low=0, high=256, size=(2, 10, 3, 20, 20), dtype=np.uint8)
    video = wandb.Video(
        data_or_path=frames,            # 可以是 NumPy 数组 (t c h w)、文件路径
        caption=f"random video",
        fps=4,                          # 帧率
    )
    run.log({"one video": video}, step=epoch)

    video_list = []
    video_list.append(video)        # 添加 wandb.Video 对象到列表
    video_list.append(video)        # 添加 wandb.Video 对象到列表
    run.log({"two videos": video_list}, step=epoch)

    # 记录表格
    # 模拟生成两幅图
    gt_array = np.uint8(np.random.rand(20, 20, 3) * 255)
    pred_array = np.uint8(np.random.rand(20, 20, 3) * 255)
    # 将 numpy 数组转换为 wandb.Image
    gt_image = wandb.Image(gt_array, caption=f"GT Epoch {epoch}")
    pred_image = wandb.Image(pred_array, caption=f"Pred Epoch {epoch}")
    # 添加当前 epoch 的数据到 table
    comparison_table.add_data(epoch, gt_image, pred_image)

run.log({"comparison_table": comparison_table}) # 记录表格只能在表格全部添加完后进行 log

# 记录模型梯度和参数
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 开始监控模型
'''
model: 你需要监控的模型实例。
log: 控制监控内容。常见选项包括：
    "gradients"：只记录梯度；
    "parameters"：只记录参数；
    "all": 同时记录梯度和参数;
log_freq: 指定每隔多少步记录一次信息。
log_graph: 是否记录模型的结构图。
'''
wandb.watch(model, log="all", log_freq=2, log_graph=True)

x_train = torch.randn(100, 10)
y_train = torch.randint(0, 5, (100,))
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    # 这里到时候 UI 上的 x 轴需要自己改为 epoch (因为 wandb 的特性是所有 .log 的 step 参数都是严格递增的, 不能重复)
    wandb.log({"epoch": epoch, "model loss": loss.item()})


# 结束 wandb run
run.finish()