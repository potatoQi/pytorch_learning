'''
Author: Error_666
Date: 2025-03-27

要实现一个最简易的 transformer gpt, 所以先要有数据集。
因为为了最简易, 所以数据不采用文本, 而是数字, 即做一个可以预测排序后的数字序列的 transformer gpt
具体来说, 数据集是若干 pairs, 每一个 pair 里有 x 和 y, x 就是一个乱序的数字序列拼接排序后的数字序列, y 就是 x 向左平移一位
举个例子, 假设乱序数字序列是 3 1 2 4, 那么:
x: 3 1 2 (4 |  1 2 3) 4
y: 1 2 4 (1 |  2 3 4) x
然后能够训练的部分是 () 括起来的部分, 因为这部分就是要预测的内容, 前面部分是作为条件输入进去的, 不能训练。
所以不能训练的地方 y 用 -1 填充, 这样在计算 loss 时就可以忽略这部分。

那规定一下参数, 首先数字范围规定为 max_num (0 ~ max_num), 然后乱序数字序列长度规定为 max_len
那么数据集里的 pair, x 就是 max_len + max_len - 1 长度 (去掉最后一位数), y 长度同理

那么至此, 自定义数据集代码可以很轻松的写出来了。有了数据集后, 想想 transformer gpt 怎么写。

首先就是一个 gpt class , 里面初始化好 embedding (token embedding + position embedding), transformer block, 等等。
然后 forward 就搭积木 (小技巧是 y=None 这样可以推理和训练都用一个 forward 函数)
然后再去写一个 generate 函数, 用来生成序列。
generate 函数传入一个乱序数字序列, 然后生成排序后的数字序列。
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from hydra.utils import instantiate
from einops import rearrange, repeat
from tqdm import tqdm
import os

class NumDataset(Dataset):
    def __init__(
            self,
            sample_num=1000,    # 数据集大小
            max_num=5,          # 数字范围  0 ~ max_num
            max_len=5,          # 乱序数字序列长度
            ):
        self.sample_num = sample_num
        self.max_num = max_num
        self.max_len = max_len
    
    def __len__(self): return self.sample_num

    def __getitem__(self, index):
        x = torch.randint(low=0, high=self.max_num, size=(self.max_len,))
        x_sort = torch.sort(x).values
        x = torch.cat([x, x_sort], dim=0)   # [max_len + max_len]
        y = torch.roll(x, shifts=-1, dims=0) # [max_len + max_len]
        x = x[:-1]  # [2 * max_len - 1]
        y = y[:-1]  # [2 * max_len - 1]
        y[:self.max_len-1] = -1
        data = {'x': x, 'y': y}
        return data

class MyMultiheadAttention(nn.Module):
    def __init__(
            self,
            embedding_dim,  # 词向量维度
            num_heads,      # 多头注意力头数
            ):
        super().__init__()
    
    def forward(self):
        pass

class Block(nn.Module):
    def __init__(
            self,
            embedding_dim,  # 词向量维度
            pos_embeddings, # 位置编码大小
            num_heads,      # 多头注意力
            block_pdrop,    # dropout 概率
            use_my_multiheadattention=False,    # 是否使用自定义多头注意力
            ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.register_buffer('mask', torch.triu(torch.full((pos_embeddings, pos_embeddings), float('-inf')), diagonal=1))
        if not use_my_multiheadattention:
            self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        else:
            self.attn = MyMultiheadAttention(embedding_dim, num_heads)
        self.drop1 = nn.Dropout(block_pdrop)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*4),
            nn.GELU(),
            nn.Linear(embedding_dim*4, embedding_dim),
        )
        self.drop2 = nn.Dropout(block_pdrop)

    def forward(self, x):
        # x: [b l d]
        x_res = x
        x = self.ln1(x)
        seq_len = x.shape[1]
        x = self.attn(x, x, x, attn_mask=self.mask[:seq_len, :seq_len])[0]
        x = x_res + self.drop1(x)

        x_res = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x_res + self.drop2(x)
        return x

class GPT(nn.Module):
    def __init__(
            self,
            num_embeddings,     # 词表大小
            embedding_dim,      # 词向量维度
            pos_embeddings,     # 位置编码大小
            block_num=16,       # transformer block 数量
            num_heads=8,        # 多头注意力头数
            block_pdrop=0.1,    # transformer block 里的 dropout 概率
            use_my_multiheadattention=False,    # 是否使用自定义的多头注意力
            ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.pos_embeddings = pos_embeddings
        self.block_num = block_num
        self.num_heads = num_heads
        self.block_pdrop = block_pdrop
        self.use_my_multiheadattention = use_my_multiheadattention

        self.token_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=pos_embeddings, embedding_dim=embedding_dim)
        self.blocks = nn.ModuleList([
            Block(
                embedding_dim=embedding_dim,
                pos_embeddings=pos_embeddings,
                num_heads=num_heads,
                block_pdrop=block_pdrop,
                use_my_multiheadattention=use_my_multiheadattention,
            ) for _ in range(block_num)
        ])
        self.ln = nn.LayerNorm(embedding_dim)
        self.proj = nn.Linear(embedding_dim, num_embeddings)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None):
        # x: [b l]
        # y: [b l]
        T = x.shape[1]
        x = self.token_embedding(x) # [b l d]
        pos = torch.arange(start=0, end=T, device=x.device)   # [l]
        pos = self.pos_embedding(pos).unsqueeze(0)   # [1 l d]
        pos = repeat(pos, '1 l d -> b l d', b=x.shape[0])   # [b l d]
        x = x + pos # [b l d]

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        x = self.proj(x)    # [b l d] -> [b l num_embeddings]
        x = rearrange(x, 'b l num_embeddings -> (b l) num_embeddings')

        loss = None
        if y is not None:
            y = rearrange(y, 'b l -> (b l)')
            loss = self.loss_fn(x, y)

        return x, loss  # x: [b*l num_embeddings]
    
    @torch.no_grad()
    def generate(self, x):
        # x: [l']
        ans = torch.tensor([], dtype=torch.long).to(x.device)
        T = x.shape[0]
        x = x.unsqueeze(0)  # [1 l']
        for _ in range(T):
            logits, _ = self(x)                      # logits: [1*l' num_embeddings]
            logits = logits[-1, :]                   # [num_embeddings]
            idx = torch.argmax(logits)               # [0]
            idx = idx.unsqueeze(0).unsqueeze(0)      # [1, 1]
            x = torch.cat([x, idx], dim=1)           # [1 l'+1]
            ans = torch.cat([ans, idx.squeeze(0)])
        return ans


if __name__ == '__main__':
    # 拿到配置文件
    config = OmegaConf.load('config.yaml')

    # 数据集准备
    dataset = instantiate(config.Dataset)
    dataloader = DataLoader(dataset, config.DataLoader.batch_size, shuffle=config.DataLoader.shuffle)

    # 模型准备
    config_gpt = config.GPT
    config_gpt.num_embeddings = config.Dataset.max_num + 1      # 词表大小为 max_num+1, 因为数字范围为 0~max_num
    config_gpt.pos_embeddings = config.Dataset.max_len * 2 - 1  # 位置编码大小为 2*max_len-1
    device = config.Other.device
    model = instantiate(config_gpt).to(device)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Other.lr)
    model.train()
    for epoch in range(config.Other.epochs):
        for step, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}/{config.Other.epochs}'):
            x = data['x'].to(device)
            y = data['y'].to(device)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch+1}/{config.Other.epochs}, loss: {loss.item()}')

    # 保存模型
    os.makedirs(config.Other.save_path, exist_ok=True)
    save_path = os.path.join(config.Other.save_path, 'gpt.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved in {save_path}')

    # 加载模型
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint)
    print(f'Model loaded from {save_path}')

    # 推理
    x = torch.randint(low=0, high=config.Dataset.max_num, size=(config.Dataset.max_len,)).to(device)
    print('Input:', x.cpu())
    y = model.generate(x)
    print('Output:', y.cpu())
    z = torch.sort(x).values
    print('Ground Truth:', z.cpu())
    print('Is correct:', torch.equal(y, z))