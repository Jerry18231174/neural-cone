#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>


#ifndef N_CLUSTERS
#define N_CLUSTERS 4
#endif


template <typename scalar_t>
__device__ void kmeans1d_single_row(
    const scalar_t *row_data,
    scalar_t *out_centers,
    scalar_t *out_counts,
    scalar_t *out_std,
    int D, int n_iter
) {
    __shared__ scalar_t shared_data[1024];
    __shared__ scalar_t centers[N_CLUSTERS];
    __shared__ int cluster_idx[1024];

    assert(D <= 1024);
    assert(threadIdx.x < blockDim.x);
    
    int tid = threadIdx.x;

    shared_data[tid] = row_data[tid];
    __syncthreads();

    bool is_valid = isfinite(shared_data[tid]);

    // Find min/max
    scalar_t thread_min = (is_valid) ? shared_data[tid] : INFINITY;
    scalar_t thread_max = (is_valid) ? shared_data[tid] : -INFINITY;

    for (int i = tid + blockDim.x; i < D; i += blockDim.x) {
        scalar_t v = shared_data[i];
        thread_min = min(thread_min, v);
        thread_max = max(thread_max, v);
    }

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        thread_min = min(thread_min, __shfl_down_sync(0xFFFFFFFF, thread_min, offset));
        thread_max = max(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
    }

    if (tid == 0) {
        for (int k = 0; k < N_CLUSTERS; ++k) {
            centers[k] = thread_min + (thread_max - thread_min) * k / (N_CLUSTERS - 1);
        }
    }
    __syncthreads();
    // Find min/max End

    __shared__ scalar_t new_centers[N_CLUSTERS];
    __shared__ int counts[N_CLUSTERS];

    for (int iter = 0; iter < n_iter; ++iter) {
        if (tid < N_CLUSTERS) {
            new_centers[tid] = 0;
            counts[tid] = 0;
        }
        __syncthreads();

        for (int i = tid; i < D; i += blockDim.x) {
            if (!is_valid) continue;
            scalar_t val = shared_data[i];
            int best_k = 0;
            scalar_t best_dist = fabsf(val - centers[0]);
            for (int k = 1; k < N_CLUSTERS; ++k) {
                scalar_t dist = fabsf(val - centers[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            atomicAdd(&new_centers[best_k], val);
            atomicAdd(&counts[best_k], 1);
            cluster_idx[i] = best_k;
        }
        __syncthreads();

        if (tid < N_CLUSTERS) {
            scalar_t count = max((float)counts[tid], 1.0f);
            centers[tid] = new_centers[tid] / count;
        }
        __syncthreads();
    }

    if (tid < N_CLUSTERS) {
        out_centers[tid] = centers[tid];
        out_counts[tid] = 0;
        out_std[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < D; i += blockDim.x) {
        if (!is_valid) continue;
        int k = cluster_idx[i];
        atomicAdd(&out_counts[k], 1.0f);
        scalar_t diff = shared_data[i] - centers[k];
        atomicAdd(&out_std[k], diff * diff);
    }
    __syncthreads();

    if (tid < N_CLUSTERS) {
        scalar_t count = max(out_counts[tid], 1.0f);
        out_std[tid] = sqrtf(out_std[tid] / count);
    }
}

template <typename scalar_t>
__global__ void kmeans1d_kernel(
    const scalar_t *input,
    scalar_t *centers,
    scalar_t *counts,
    scalar_t *stds,
    int N, int D, int n_iter
) {
    int rowIdx = blockIdx.x;
    int colIdx = threadIdx.x;
    if (rowIdx >= N || colIdx >= D) return;

    const scalar_t *row_ptr = input + rowIdx * D;
    scalar_t *out_c = centers + rowIdx * N_CLUSTERS;
    scalar_t *out_n = counts + rowIdx * N_CLUSTERS;
    scalar_t *out_s = stds + rowIdx * N_CLUSTERS;

    kmeans1d_single_row<scalar_t>(row_ptr, out_c, out_n, out_s, D, n_iter);    
}

void launch_kmeans1d(
    torch::Tensor input,
    torch::Tensor centers,
    torch::Tensor counts,
    torch::Tensor stds,
    int n_iter
) {
    int N = input.size(0);
    int D = input.size(1);
    
    int threads = 1;
    while (threads < D) threads *= 2;
    threads = std::min(threads, 1024);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "kmeans1d_kernel", ([&] {
        kmeans1d_kernel<scalar_t><<<N, threads>>>(
            input.data_ptr<scalar_t>(),
            centers.data_ptr<scalar_t>(),
            counts.data_ptr<scalar_t>(),
            stds.data_ptr<scalar_t>(),
            N, D, n_iter
        );
    }));
}