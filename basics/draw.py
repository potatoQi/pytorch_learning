import matplotlib.pyplot as plt

def draw(
        figsize=(8,8),
        size=(1,1),
        images=None,    # ToTensor() 后的 tensor, [b c h w]
        labels=None,    # [b]
    ):
    plt.figure(figsize=figsize)
    rows, cols = size
    assert labels.shape[0] == rows * cols, 'size 必须与图片数量相等'
    for i in range(1, rows * cols + 1):
        plt.subplot(rows, cols, i)
        plt.title(labels[i-1].item())
        plt.axis('off')
        if images.shape[1] == 1:
            plt.imshow(images[i-1][0], cmap='gray') # gray 模式的 input 为 H x W
        elif images.shape[1] == 3:
            plt.imshow(images[i-1].permute(1, 2, 0)) # RGB 模式的 input 为 H x W x C
        else:
            raise ValueError('不支持的图片通道数')
    plt.show()