import torch
import cppcuda  # 易错: 导入 cppcuda 之前一定要导入 torch
import time

def trilinear_interpolation_py(feats, points):
    N, _, F = feats.shape
    
    u = (points[:, 0:1] + 1) / 2    # [N 1]
    v = (points[:, 1:2] + 1) / 2    # [N 1]
    w = (points[:, 2:3] + 1) / 2    # [N 1]

    w000 = (1 - u) * (1 - v) * (1 - w)     # [N 1]
    w001 = (1 - u) * (1 - v) * w
    w010 = (1 - u) * v * (1 - w)
    w011 = (1 - u) * v * w
    w100 = u * (1 - v) * (1 - w)
    w101 = u * (1 - v) * w
    w110 = u * v * (1 - w)
    w111 = u * v * w

    res = w000 * feats[:, 0] + w001 * feats[:, 1] + w010 * feats[:, 2] + w011 * feats[:, 3] + \
          w100 * feats[:, 4] + w101 * feats[:, 5] + w110 * feats[:, 6] + w111 * feats[:, 7]
    return res

class Trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points):
        res = cppcuda.trilinear_interpolation_forward(feats, points)
        ctx.save_for_backward(feats, points)
        return res

    @staticmethod
    def backward(ctx, dl_dres):
        feats, points = ctx.saved_tensors
        dl_dfeats = cppcuda.trilinear_interpolation_backward(dl_dres.contiguous(), feats, points)
        return dl_dfeats, None

if __name__ == '__main__':
    N = 65546
    F = 256

    feats = torch.rand(N, 8, F).cuda()
    points = torch.rand(N, 3, device='cuda') * 2 - 1
    feats_c = feats.clone()
    points_c = points.clone()

    # cuda
    t = time.time()
    res_cuda = Trilinear_interpolation_cuda.apply(feats.requires_grad_(), points)
    loss = res_cuda.sum()
    loss.backward()
    torch.cuda.synchronize()    # 强制让 cpu 等待, 直到所有提交给当前设备的 cuda 任务完成
    print('CUDA time:', time.time() - t, 's')

    # pytorch
    t = time.time()
    res_py = trilinear_interpolation_py(feats_c.requires_grad_(), points_c)
    loss_c = res_py.sum()
    loss_c.backward()
    torch.cuda.synchronize()
    print('PyTorch time:', time.time() - t, 's')

    # 判断是否在误差之内
    print(torch.allclose(res_cuda, res_py, atol=1e-8))
    print(torch.allclose(feats.grad, feats_c.grad, atol=1e-8))
