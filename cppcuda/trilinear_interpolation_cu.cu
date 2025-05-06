#include <torch/extension.h>

// 函数头就抄板子就行了
// __global__ 意思是这个函数在 cpu 上调用, gpu 上执行
template <typename scalar_t>
__global__ void forward_kernel(
    // 因为 feats, points 输入是不会变的, 所以我用 const; 但是 res 是要一个一个往里填的, 所以不加 const
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,         // [N 8 F]
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,        // [N 3]
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> res                  // [N F]
) {
    // 算出当前 thread 的编号 
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断这个 thread 是否是我们需要的 (是否在 res 的范围内)
    if (n >= feats.size(0) || f >= feats.size(2)) return;

    // 现在开始计算第 n 个体素中的点的第 f 个特征值的插值
    // 因为体素是 1x1x1 的, 所以 8 个角点坐标分别是 (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    // 这里因为 points 输入是在 [-1, 1] 的, 所以做一个归一化
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;
    /*
        | i   | (x, y, z) | 权重公式                                 |
        |-----|-----------|------------------------------------------|
        | 0   | (0, 0, 0) | $w_{000} = (1 - u)(1 - v)(1 - w)$    |
        | 1   | (0, 0, 1) | $w_{001} = (1 - u)(1 - v)w$          |
        | 2   | (0, 1, 0) | $w_{010} = (1 - u)v(1 - w)$          |
        | 3   | (0, 1, 1) | $w_{011} = (1 - u)vw$                |
        | 4   | (1, 0, 0) | $w_{100} = u(1 - v)(1 - w)$          |
        | 5   | (1, 0, 1) | $w_{101} = u(1 - v)w$                |
        | 6   | (1, 1, 0) | $w_{110} = uv(1 - w)$                |
        | 7   | (1, 1, 1) | $w_{111} = uvw$                      |
    */
    const scalar_t w000 = (1 - u) * (1 - v) * (1 - w);
    const scalar_t w001 = (1 - u) * (1 - v) * w;
    const scalar_t w010 = (1 - u) * v * (1 - w);
    const scalar_t w011 = (1 - u) * v * w;
    const scalar_t w100 = u * (1 - v) * (1 - w);
    const scalar_t w101 = u * (1 - v) * w;
    const scalar_t w110 = u * v * (1 - w);
    const scalar_t w111 = u * v * w;
    res[n][f] = w000 * feats[n][0][f] + w001 * feats[n][1][f] + w010 * feats[n][2][f] + w011 * feats[n][3][f] +
                w100 * feats[n][4][f] + w101 * feats[n][5][f] + w110 * feats[n][6][f] + w111 * feats[n][7][f];

    return;
}

torch::Tensor forward_cu(
    const torch::Tensor feats,    // [N 8 F]
    const torch::Tensor points    // [N 3]
) {
    const int N = feats.size(0), F = feats.size(2);
    
    // 先定义一个占位结果
    torch::Tensor res = torch::empty({N, F}, feats.options());  // feats.options() 表示 dtype, device, requires_grad 等等跟 feats 一样
    // 如果 res 想和 feats 的 dtype 不一样, 但是 device 一样, 可以这样写:
    // torch::Tensor res = torch::empty({N, F}, torch::dtype(torch::kInt32).device(feats.device).requires_grad(false));
    
    // 定义一个 block 里有多少个 threads
    // 写 dim3 是因为 cuda 的线程组织是三维的, 即并行的变量最多三维. 如果并行变量是二维, 也用 dim3, cuda 会自动把最后一维的大小设为 1.
    // 下面我这种写法是给一个 block 里分配了 16x16 个 threads, 其中 x 轴线程数 16, y 轴线程数 16, z 轴线程数 1. (因为 N, F 维度可以平行计算)
    // 至于一个 block 最多分配多少个 threads, 这个要看 GPU 的架构, 分多分少会跟运算速度挂钩, 这个要自己尝试
    const dim3 threads(16, 16);
    
    // 接下来定义 block 的数量
    // block 所需数量是有计算公式的, 原理就是前面定义了一个 block 里有多少个 threads 嘛, 然后 res 又是一个 [N F] 的矩阵, 所以就是密铺, 需要多少个 block
    // 才能把 res 铺满. 假设 res 大小是 [20 10], 那么显然就要竖着铺两个 block, 每个 block 是 [16 16]
    // 所以 block 所需数量就是 [2 1]
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    // 现在就很形象了, res 不是 [N F] 嘛, 然后现在定义的 block 已经可以密铺把 res 盖起来了
    // res 每个点就是由盖到它的 thread 去计算

    // 在定义完需要的 block 和 thread 数量之后, 开始调 kernel 去进行正式运算
    // AT_DISPATCH_FLOATING_TYPES 是一个宏, 这个宏会根据第一次参数来选择合适的 kernel 去执行
    // 第一次参数填的是数据类型, 这里我填的是 feats 的数据类型, 也就是 feats.type()
    // 第二次参数执行 kernel 报错时抛出的名字, 建议跟所在函数名一样, 方便调试
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "forward_cu", ([&] {
        // 下面就是标准的 cuda kernel launch 语法
        // 照着写就好了, 也不用改啥地方, 里面的 3/2 就是 tensor 的 ndim
        forward_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            res.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return res;
}

/********************************************************************************************/

template <typename scalar_t>
__global__ void backward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dl_dres,       // [N F]
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,         // [N 8 F]
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,        // [N 3]
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dl_dfeats            // [N 8 F]
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= feats.size(0) || f >= feats.size(2)) return;
    
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;
    /*
        | i   | (x, y, z) | 权重公式                                 |
        |-----|-----------|------------------------------------------|
        | 0   | (0, 0, 0) | $w_{000} = (1 - u)(1 - v)(1 - w)$    |
        | 1   | (0, 0, 1) | $w_{001} = (1 - u)(1 - v)w$          |
        | 2   | (0, 1, 0) | $w_{010} = (1 - u)v(1 - w)$          |
        | 3   | (0, 1, 1) | $w_{011} = (1 - u)vw$                |
        | 4   | (1, 0, 0) | $w_{100} = u(1 - v)(1 - w)$          |
        | 5   | (1, 0, 1) | $w_{101} = u(1 - v)w$                |
        | 6   | (1, 1, 0) | $w_{110} = uv(1 - w)$                |
        | 7   | (1, 1, 1) | $w_{111} = uvw$                      |
    */
    const scalar_t w000 = (1 - u) * (1 - v) * (1 - w);
    const scalar_t w001 = (1 - u) * (1 - v) * w;
    const scalar_t w010 = (1 - u) * v * (1 - w);
    const scalar_t w011 = (1 - u) * v * w;
    const scalar_t w100 = u * (1 - v) * (1 - w);
    const scalar_t w101 = u * (1 - v) * w;
    const scalar_t w110 = u * v * (1 - w);
    const scalar_t w111 = u * v * w;
    
    dl_dfeats[n][0][f] = dl_dres[n][f] * w000;
    dl_dfeats[n][1][f] = dl_dres[n][f] * w001;
    dl_dfeats[n][2][f] = dl_dres[n][f] * w010;
    dl_dfeats[n][3][f] = dl_dres[n][f] * w011;
    dl_dfeats[n][4][f] = dl_dres[n][f] * w100;
    dl_dfeats[n][5][f] = dl_dres[n][f] * w101;
    dl_dfeats[n][6][f] = dl_dres[n][f] * w110;
    dl_dfeats[n][7][f] = dl_dres[n][f] * w111;

    return;
}

torch::Tensor backward_cu(
    const torch::Tensor dl_dres,  // [N F]
    const torch::Tensor feats,    // [N 8 F]
    const torch::Tensor points    // [N 3]
) {
    const int N = feats.size(0), F = feats.size(2);
    
    torch::Tensor dl_dfeats = torch::empty({N, 8, F}, feats.options());
    
    // 这里仍然用二维的 threads 是因为 dl_dfeats 需要并行化的逻辑仍然是二维的: N 和 F
    const dim3 threads(16, 16);
    
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "backward_cu", ([&] {
        backward_kernel<scalar_t><<<blocks, threads>>>(
            dl_dres.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dl_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dl_dfeats;
}