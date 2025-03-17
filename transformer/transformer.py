# 写一个数字排序的 transformer gpt 模型
# 最精简的实现, 仅包含必要的部分, 用于展示 transformer 的基本结构
# topk, sample, 文本预测, 只保留了最基本的 transformer 结构

# 发现位置编码对于收敛效果和收敛速度都有很大影响, 不加的话效果差了非常多
# 以及 mask 也很重要, 加了 mask 后预测准确率大大提升

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from einops import rearrange

class MyDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        voc_size=7,
        max_len=5,
    ):
        self.num_samples = num_samples
        self.voc_size = voc_size
        self.max_len = max_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = torch.randint(low=0, high=self.voc_size-1, size=(self.max_len,))
        y = torch.sort(x).values
        x = torch.cat([x, y], dim=0)
        # 我们想要模型学到的能力是, input 前 max_len 个未排序的数字进去, 她能逐帧预测出排序后的数字
        data = x[:-1].clone()
        label = x[1:].clone()
        label[:self.max_len-1] = -1    # 这里是因为我们不需要预测前 max_len 个数字, 相反前 max_len 个数字是必须已知信息, 用来预测后面的数字
        res = {
            'data': data,
            'label': label,
        }
        return res

class Block(nn.Module):
    def __init__(
        self,
        d_dim=64,
        num_heads=8,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_dim)
        self.attn = nn.MultiheadAttention(d_dim, num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.ln2 = nn.LayerNorm(d_dim)
        self.fc1 = nn.Linear(d_dim, d_dim * 4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_dim * 4, d_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, attn_mask):    # 目前主流的 transformer block 实现
        x_res = x
        x = self.ln1(x)
        x = self.attn(x, x, x, attn_mask=attn_mask)[0]
        x = x_res + self.dropout1(x)

        x_res = x
        x = self.ln2(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = x_res + self.dropout2(x)
        return x

class Model(nn.Module):
    def __init__(
        self,
        voc_size=7,
        max_len=5,
        d_dim=64,
        num_heads=8,
        num_layers=16,
        device='cuda',
    ):
        super().__init__()
        self.voc_size = voc_size
        self.max_len = max_len
        self.d_dim = d_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        self.tokenizer = nn.Embedding(voc_size, d_dim)          # 词表大小为 voc_size, encode 后的维度为 d_dim
        self.pos_embedding = nn.Embedding(max_len*2-1, d_dim)   # 位置下标有 max_len*2-1 个
        self.blocks = nn.ModuleList([
            Block(d_dim=d_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_dim)
        self.proj = nn.Linear(d_dim, voc_size, bias=False)  # 投影回词表大小 (这里记得不要加 bias!)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # 忽略类别下标为 -1 的样本的 loss 计算

    # 这个 forward 函数就是前向传播, 所以训练和推理的时候都会用到, 如果 y 不为 None, 就是训练模式, 会返回 loss
    def forward(self, x, y=None):
        # x: [B, T]
        # y: [B, T]
        B, T = x.shape
        x_embedding = self.tokenizer(x)                                               # -> [B, T, D]
        pos = torch.arange(start=0, end=T, dtype=torch.long).unsqueeze(0).to(self.device)  # 生成位置下标 0-T-1
        pos_embedding = self.pos_embedding(pos)                                       # -> [1, T, D]
        x = x_embedding + pos_embedding    # -> [B, T, D] (广播机制)

        attn_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1).to(self.device)  # 生成上三角矩阵, 用于 mask
        for block in self.blocks:
            x = block(x, attn_mask)
        
        x = self.ln(x)
        logits = self.proj(x)   # -> [B, T, voc_size]

        loss = None
        if y is not None:
            logits = rearrange(logits, 'b t d -> (b t) d')    # -> [B*T, D]
            y = rearrange(y, 'b t -> (b t)')                  # -> [B*T,]
            # nn.CrossEntropyLoss() 需要 x1:(T, C) 和 x2:(T,) 两个输入
            # 其中 T 为样本数, C 为类别数
            # x1 就表示每个样本对 C 个类别们的 logits, x2 就表示每个样本应该属于的类别下标 (0~C-1)
            loss = self.loss_fn(logits, y)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, x):
        # x: [B, T]
        for _ in range(self.max_len):
            logits, _ = self(x)
            logits = logits[:, -1, :]                     # 只关心最后一个 token 的 logits, -> [B, voc_size]
            indices = logits.argmax(dim=-1).unsqueeze(-1) # -> [B, 1]
            x = torch.cat([x, indices], dim=-1)           # -> [B, T+1]
        return x[:, self.max_len:]


if __name__ == '__main__':
    ############################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = 5
    voc_size = 7
    d_dim = 64
    num_heads = 8
    num_layers = 16
    epoch_num = 10
    lr = 1e-4
    ############################################

    dataset = MyDataset(num_samples=1000, voc_size=voc_size, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = Model(
        voc_size=voc_size,
        max_len=max_len,
        d_dim=d_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epoch_num):
        for step, batch in tqdm(enumerate(dataloader), desc=f'Epoch {epoch}', total=len(dataloader)):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Loss: {loss.item()}')

    save_dict = {
        'model': model.state_dict(),
    }
    torch.save(save_dict, 'model.pth')
    checkpoint = torch.load('model.pth')
    model.load_state_dict(checkpoint['model'])

    model.eval()
    x = torch.randint(low=0, high=voc_size-1, size=(max_len,))
    y = torch.sort(x).values
    print(f'原序列: {x}')
    print(f'标准答案: {y}')
    y_pred = model.generate(x.unsqueeze(0).to(device))
    print(f'预测答案: {y_pred.squeeze(0).cpu()}')