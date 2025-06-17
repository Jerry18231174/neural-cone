#include <torch/extension.h>


#ifndef N_CLUSTERS
#define N_CLUSTERS 4
#endif


void launch_kmeans1d(
    torch::Tensor input,
    torch::Tensor centers,
    torch::Tensor counts,
    torch::Tensor stds,
    int n_iter);

std::vector<torch::Tensor> kmeans1d_forward(torch::Tensor input, int n_iter) {
    auto N = input.size(0);
    auto options = input.options();
    auto centers = torch::zeros({N, N_CLUSTERS}, options);
    auto counts  = torch::zeros({N, N_CLUSTERS}, options);
    auto stds    = torch::zeros({N, N_CLUSTERS}, options);

    launch_kmeans1d(input, centers, counts, stds, n_iter);
    return {centers, counts, stds};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kmeans1d", &kmeans1d_forward, "Row-wise 1D KMeans (CUDA)");
}