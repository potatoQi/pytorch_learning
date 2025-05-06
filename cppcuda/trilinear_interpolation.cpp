// 这个相当于打竞赛的 <bits/stdc++.h>
#include <torch/extension.h>
#include "utils.h"

torch::Tensor trilinear_interpolation_backward(
    const torch::Tensor dl_dres,  // [N F]
    const torch::Tensor feats,    // [N 8 F]
    const torch::Tensor points    // [N 3]
) {
    CHECK_INPUT(dl_dres);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);
    return backward_cu(dl_dres, feats, points);
}

torch::Tensor trilinear_interpolation_forward(
    const torch::Tensor feats,    // [N 8 F]
    const torch::Tensor points    // [N 3]
) {
    CHECK_INPUT(feats);
    CHECK_INPUT(points);
    return forward_cu(feats, points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("<str1>", &<str2>): 在 python 程序中, 通过 str1 呼叫 cpp 中的 str2 函数
    m.def("trilinear_interpolation_forward", &trilinear_interpolation_forward);
    m.def("trilinear_interpolation_backward", &trilinear_interpolation_backward);
}