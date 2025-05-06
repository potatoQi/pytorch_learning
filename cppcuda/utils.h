#include <torch/extension.h>

// 这里的 magic code 是安全性检查
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 对 .cu 文件里函数的声明
torch::Tensor forward_cu(
    const torch::Tensor feats,
    const torch::Tensor points
);

torch::Tensor backward_cu(
    const torch::Tensor dl_dres,  // [N F]
    const torch::Tensor feats,    // [N 8 F]
    const torch::Tensor points    // [N 3]
);