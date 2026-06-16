#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <cassert>
#include <vector>


#define CONCAT  0
#define MEAN    1
#define INTERP  2

#ifndef LAYER_REDUCE
#define LAYER_REDUCE CONCAT
#endif


void launch_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
);

void launch_forward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
);

void launch_backward(
    const torch::Tensor pos,
    const torch::Tensor grad_output,
    const std::vector<torch::Tensor>& grad_grids
);

void launch_backward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const torch::Tensor grad_output,
    const std::vector<torch::Tensor>& grad_grids
);

torch::Tensor forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids
) {
    TORCH_CHECK(LAYER_REDUCE != INTERP, "forward is unavailable when LAYER_REDUCE=INTERP");

    auto N = pos.size(0);
    auto L = grids.size();
    auto D = grids[0].size(2);
    auto options = grids[0].options();

    assert(L == LEVELS);
    assert(D == DIMENSIONS);

    torch::Tensor result;
    if (LAYER_REDUCE == MEAN) {
        result = torch::zeros({N, D}, options);
    } else if (LAYER_REDUCE == CONCAT) {
        result = torch::zeros({N, L * D}, options);
    } else {
        throw std::invalid_argument("Invalid layer reduce option" + LAYER_REDUCE);
    }

    launch_forward(pos, grids, result);

    return result;
}

torch::Tensor forward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const std::vector<torch::Tensor>& grids
) {
    auto N = pos.size(0);
    auto L = grids.size();
    auto D = grids[0].size(2);
    auto options = grids[0].options();

    assert(L == LEVELS);
    assert(D == DIMENSIONS);

    torch::Tensor result = torch::zeros({N, D}, options);

    launch_forward_layer_interp(pos, point_size, grids, result);

    return result;
}

std::vector<torch::Tensor> backward(
    const torch::Tensor pos,
    const torch::Tensor grad_output,
    const std::vector<torch::Tensor>& grids
) {
    TORCH_CHECK(LAYER_REDUCE != INTERP, "backward is unavailable when LAYER_REDUCE=INTERP");

    auto N = pos.size(0);
    auto L = grids.size();
    auto D = grids[0].size(2);

    TORCH_CHECK(L == LEVELS, "Expected ", LEVELS, " levels but got ", L);
    TORCH_CHECK(D == DIMENSIONS, "Expected ", DIMENSIONS, " features but got ", D);
    TORCH_CHECK(
        grad_output.size(0) == N,
        "grad_output batch size mismatch: expected ", N, " but got ", grad_output.size(0)
    );

    if (LAYER_REDUCE == MEAN) {
        TORCH_CHECK(
            grad_output.size(1) == D,
            "grad_output feature size mismatch for mean reduce: expected ", D, " but got ", grad_output.size(1)
        );
    } else if (LAYER_REDUCE == CONCAT) {
        TORCH_CHECK(
            grad_output.size(1) == L * D,
            "grad_output feature size mismatch for concat reduce: expected ", L * D, " but got ", grad_output.size(1)
        );
    }

    std::vector<torch::Tensor> grad_grids;
    grad_grids.reserve(L);
    for (const auto& grid : grids) {
        grad_grids.push_back(torch::zeros_like(grid));
    }

    launch_backward(pos, grad_output, grad_grids);

    return grad_grids;
}

std::vector<torch::Tensor> backward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const torch::Tensor grad_output,
    const std::vector<torch::Tensor>& grids
) {
    auto N = pos.size(0);
    auto L = grids.size();
    auto D = grids[0].size(2);

    TORCH_CHECK(L == LEVELS, "Expected ", LEVELS, " levels but got ", L);
    TORCH_CHECK(D == DIMENSIONS, "Expected ", DIMENSIONS, " features but got ", D);
    TORCH_CHECK(
        point_size.size(0) == N,
        "point_size batch size mismatch: expected ", N, " but got ", point_size.size(0)
    );
    TORCH_CHECK(
        grad_output.size(0) == N && grad_output.size(1) == D,
        "grad_output shape mismatch for layer interp: expected [", N, ", ", D, "]"
    );

    std::vector<torch::Tensor> grad_grids;
    grad_grids.reserve(L);
    for (const auto& grid : grids) {
        grad_grids.push_back(torch::zeros_like(grid));
    }

    launch_backward_layer_interp(pos, point_size, grad_output, grad_grids);

    return grad_grids;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward,
        py::arg("position"),
        py::arg("grids"),
        "Multi-resolution hash grid with layer reduction (CUDA)"
    );
    m.def("backward", &backward,
        py::arg("position"),
        py::arg("grad_output"),
        py::arg("grids"),
        "Backward pass for multi-resolution hash grid (CUDA)"
    );
    m.def("forward_layer_interp", &forward_layer_interp,
        py::arg("position"),
        py::arg("point_size"),
        py::arg("grids"),
        "Multi-resolution hash grid with layer interpolation (CUDA)"
    );
    m.def("backward_layer_interp", &backward_layer_interp,
        py::arg("position"),
        py::arg("point_size"),
        py::arg("grad_output"),
        py::arg("grids"),
        "Backward pass for layer-interpolated hash grid (CUDA)"
    );
}
