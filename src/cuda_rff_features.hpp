#pragma once

#include <torch/extension.h>

namespace kf::rff_cuda {

torch::Tensor rff_features_cuda(
    const torch::Tensor &X, const torch::Tensor &W, const torch::Tensor &b
);

std::tuple<torch::Tensor, torch::Tensor> rff_gramian_symm_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    int chunk_size
);

std::tuple<torch::Tensor, torch::Tensor> rff_full_gramian_symm_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    const torch::Tensor &F,
    int energy_chunk,
    int force_chunk
);

torch::Tensor rff_predict_energy_cuda(
    const torch::Tensor &X,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
);

torch::Tensor rff_predict_force_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
);

torch::Tensor rff_features_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b
);

std::tuple<torch::Tensor, torch::Tensor> rff_gramian_elemental_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    int chunk_size
);

torch::Tensor rff_predict_energy_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
);

std::tuple<torch::Tensor, torch::Tensor> rff_full_gramian_elemental_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    const torch::Tensor &F,
    int energy_chunk,
    int force_chunk
);

torch::Tensor rff_predict_force_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
);

}  // namespace kf::rff_cuda
