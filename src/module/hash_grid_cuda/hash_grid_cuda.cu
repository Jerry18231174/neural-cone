#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define PRIME_X 1
#define PRIME_Y 19349663
#define PRIME_Z 83492791

#ifndef LEVELS
#define LEVELS 4
#endif

#ifndef DIMENSIONS
#define DIMENSIONS 8
#endif

#ifndef LOG_HASHMAP_SIZE
#define LOG_HASHMAP_SIZE 19
#endif

#ifndef BASE_RESOLUTION
#define BASE_RESOLUTION 32
#endif

#ifndef PER_LEVEL_SCALE
#define PER_LEVEL_SCALE 2.0
#endif

#define CONCAT  0
#define MEAN    1
#define INTERP  2

#ifndef LAYER_REDUCE
#define LAYER_REDUCE CONCAT
#endif

#ifndef INTERP_PARALLEL
#define INTERP_PARALLEL  4
#endif

#ifndef INTERP_RATIO
#define INTERP_RATIO  0.33
#endif

#ifndef THREADS
#define THREADS 128
#endif


template <typename scalar_t>
struct scalar_t3 {
    scalar_t x, y, z;

    __host__ __device__ scalar_t3() : x(0), y(0), z(0) {}
    __host__ __device__ scalar_t3(scalar_t x_, scalar_t y_, scalar_t z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ scalar_t3 operator+(const scalar_t3& other) const {
        return scalar_t3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ scalar_t3 operator-(const scalar_t3& other) const {
        return scalar_t3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ scalar_t3 operator*(scalar_t s) const {
        return scalar_t3(x * s, y * s, z * s);
    }

    __host__ __device__ scalar_t3 operator/(scalar_t s) const {
        return scalar_t3(x / s, y / s, z / s);
    }

    __host__ __device__ scalar_t dot(const scalar_t3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ scalar_t norm() const {
        return sqrt(dot(*this));
    }

    __host__ __device__ scalar_t3 normalized() const {
        scalar_t n = norm();
        return n > 0 ? (*this) / n : scalar_t3(0, 0, 0);
    }

    __host__ __device__ int3 to_int3() const {
        return make_int3(
            static_cast<int>(floorf(x)),
            static_cast<int>(floorf(y)),
            static_cast<int>(floorf(z))
        );
    }
};


__device__ __forceinline__ int3 _corner_offset(int c) {
    return make_int3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
}


// Should be initialized in host function
__constant__ int resolutions[LEVELS];     // BASE_RESOLUTION * scale^level
__constant__ int grid_sizes[LEVELS];      // (res + 1)^3
__constant__ float grid_scales[LEVELS];   // INTERP_RATIO / res

__device__ __forceinline__ uint32_t _hash_index(int3 index, int level) {
    constexpr uint32_t MAX_HASH = 1 << LOG_HASHMAP_SIZE;

    int res1 = resolutions[level] + 1;
    int dense_max = grid_sizes[level];

    uint32_t hashed = 0;

    if (dense_max > MAX_HASH) {
        // Hash mode
        int64_t result =
            static_cast<int64_t>(index.x) * PRIME_X +
            static_cast<int64_t>(index.y) * PRIME_Y +
            static_cast<int64_t>(index.z) * PRIME_Z;
        hashed = static_cast<uint32_t>(llabs(result % MAX_HASH));
    } else {
        // Dense indexing mode
        int32_t dense_idx =
            res1 * res1 * index.x +
            res1 * index.y +
            index.z;
        hashed = static_cast<uint32_t>(labs(dense_idx % dense_max));
    }

    return hashed;
}


template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t *pos,
    const scalar_t **grids,
    scalar_t *result,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * LEVELS;

    if (tid >= total_threads) return;

    int bid = tid / LEVELS;
    int level = tid % LEVELS;

    int res = resolutions[level];
    
    const scalar_t *pos_ptr = pos + bid * 3;
    scalar_t3<scalar_t> pos3 = scalar_t3<scalar_t>(pos_ptr[0], pos_ptr[1], pos_ptr[2]);

    // Get hash index
    scalar_t3<scalar_t> pos_grid = pos3 * res;
    int3 base = pos_grid.to_int3();
    scalar_t3<scalar_t> offset = scalar_t3<scalar_t>(
        pos_grid.x - base.x,
        pos_grid.y - base.y,
        pos_grid.z - base.z
    );
    const scalar_t *grid_ptr = grids[level];
    scalar_t feature_accum[DIMENSIONS];
    for (int i = 0; i < DIMENSIONS; ++i) {
        feature_accum[i] = 0;
    }

    for (int corner = 0; corner < 8; ++corner) {
        int3 corner3 = _corner_offset(corner);
        int3 index3 = make_int3(
            base.x + corner3.x,
            base.y + corner3.y,
            base.z + corner3.z
        );
        uint32_t index = _hash_index(index3, level);
        const scalar_t *grid = grid_ptr + index * DIMENSIONS;

        scalar_t weight = (corner3.x ? offset.x : (1 - offset.x)) *
                          (corner3.y ? offset.y : (1 - offset.y)) *
                          (corner3.z ? offset.z : (1 - offset.z));

        for (int i = 0; i < DIMENSIONS; ++i) {
            feature_accum[i] += grid[i] * weight;
        }
    }

#if (LAYER_REDUCE == CONCAT)
    for (int i = 0; i < DIMENSIONS; ++i) {
        result[bid * LEVELS * DIMENSIONS + level * DIMENSIONS + i] = feature_accum[i];
    }
#elif (LAYER_REDUCE == MEAN)
    for (int i = 0; i < DIMENSIONS; ++i) {
        atomicAdd(&result[bid * DIMENSIONS + i], feature_accum[i] / LEVELS);
    }
#endif
}


template <typename scalar_t>
__global__ void forward_layer_interp_kernel(
    const scalar_t *pos,
    const scalar_t *point_size,
    const scalar_t **grids,
    scalar_t *result,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    const scalar_t *pos_ptr = pos + tid * 3;
    scalar_t psize_val = point_size[tid];
    scalar_t3<scalar_t> pos3 = scalar_t3<scalar_t>(pos_ptr[0], pos_ptr[1], pos_ptr[2]);

    scalar_t feature_accum[DIMENSIONS];
    for (int i = 0; i < DIMENSIONS; ++i) {
        feature_accum[i] = 0;
    }

    int levels[2] = {0, -1};
    scalar_t layer_weights[2] = {0, 0};

    if (psize_val >= static_cast<scalar_t>(grid_scales[0])) {
        levels[0] = 0;
        layer_weights[0] = 1;
    } else if (psize_val <= static_cast<scalar_t>(grid_scales[LEVELS - 1])) {
        levels[0] = LEVELS - 1;
        layer_weights[0] = 1;
    } else {
        for (int i = 0; i < LEVELS - 1; ++i) {
            scalar_t coarser_size = static_cast<scalar_t>(grid_scales[i]);
            scalar_t finer_size = static_cast<scalar_t>(grid_scales[i + 1]);
            if (psize_val <= coarser_size && psize_val >= finer_size) {
                scalar_t denom = coarser_size - finer_size;
                levels[0] = i;
                levels[1] = i + 1;
                layer_weights[0] = (psize_val - finer_size) / denom;
                layer_weights[1] = (coarser_size - psize_val) / denom;
                break;
            }
        }
    }

    for (int li = 0; li < 2; ++li) {
        int level = levels[li];
        scalar_t layer_weight = layer_weights[li];
        if (level < 0 || layer_weight == 0) continue;

        int res = resolutions[level];
        scalar_t3<scalar_t> pos_grid = pos3 * res;
        int3 base = pos_grid.to_int3();
        scalar_t3<scalar_t> offset = scalar_t3<scalar_t>(
            pos_grid.x - base.x,
            pos_grid.y - base.y,
            pos_grid.z - base.z
        );

        const scalar_t *grid_ptr = grids[level];

        for (int corner = 0; corner < 8; ++corner) {
            int3 corner3 = _corner_offset(corner);
            int3 index3 = make_int3(
                base.x + corner3.x,
                base.y + corner3.y,
                base.z + corner3.z
            );
            uint32_t index = _hash_index(index3, level);
            const scalar_t *grid = grid_ptr + index * DIMENSIONS;

            scalar_t weight = (corner3.x ? offset.x : (1 - offset.x)) *
                              (corner3.y ? offset.y : (1 - offset.y)) *
                              (corner3.z ? offset.z : (1 - offset.z));

            for (int i = 0; i < DIMENSIONS; ++i) {
                feature_accum[i] += grid[i] * weight * layer_weight;
            }
        }
    }

    for (int i = 0; i < DIMENSIONS; ++i) {
        result[tid * DIMENSIONS + i] = feature_accum[i];
    }
}


template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t *pos,
    const scalar_t *grad_output,
    scalar_t **grad_grids,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * LEVELS;

    if (tid >= total_threads) return;

    int bid = tid / LEVELS;
    int level = tid % LEVELS;
    int res = resolutions[level];

    const scalar_t *pos_ptr = pos + bid * 3;
    scalar_t3<scalar_t> pos3 = scalar_t3<scalar_t>(pos_ptr[0], pos_ptr[1], pos_ptr[2]);
    scalar_t3<scalar_t> pos_grid = pos3 * res;
    int3 base = pos_grid.to_int3();
    scalar_t3<scalar_t> offset = scalar_t3<scalar_t>(
        pos_grid.x - base.x,
        pos_grid.y - base.y,
        pos_grid.z - base.z
    );
    scalar_t *grad_grid_ptr = grad_grids[level];

    for (int corner = 0; corner < 8; ++corner) {
        int3 corner3 = _corner_offset(corner);
        int3 index3 = make_int3(
            base.x + corner3.x,
            base.y + corner3.y,
            base.z + corner3.z
        );
        uint32_t index = _hash_index(index3, level);
        scalar_t weight = (corner3.x ? offset.x : (1 - offset.x)) *
                          (corner3.y ? offset.y : (1 - offset.y)) *
                          (corner3.z ? offset.z : (1 - offset.z));

        for (int i = 0; i < DIMENSIONS; ++i) {
#if (LAYER_REDUCE == CONCAT)
            atomicAdd(
                &grad_grid_ptr[index * DIMENSIONS + i],
                grad_output[bid * LEVELS * DIMENSIONS + level * DIMENSIONS + i] * weight
            );
#elif (LAYER_REDUCE == MEAN)
            atomicAdd(
                &grad_grid_ptr[index * DIMENSIONS + i],
                (grad_output[bid * DIMENSIONS + i] / LEVELS) * weight
            );
#endif
        }
    }
}


template <typename scalar_t>
__global__ void backward_layer_interp_kernel(
    const scalar_t *pos,
    const scalar_t *point_size,
    const scalar_t *grad_output,
    scalar_t **grad_grids,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    const scalar_t *pos_ptr = pos + tid * 3;
    const scalar_t *grad_ptr = grad_output + tid * DIMENSIONS;
    scalar_t psize_val = point_size[tid];
    scalar_t3<scalar_t> pos3 = scalar_t3<scalar_t>(pos_ptr[0], pos_ptr[1], pos_ptr[2]);

    int levels[2] = {0, -1};
    scalar_t layer_weights[2] = {0, 0};

    if (psize_val >= static_cast<scalar_t>(grid_scales[0])) {
        levels[0] = 0;
        layer_weights[0] = 1;
    } else if (psize_val <= static_cast<scalar_t>(grid_scales[LEVELS - 1])) {
        levels[0] = LEVELS - 1;
        layer_weights[0] = 1;
    } else {
        for (int i = 0; i < LEVELS - 1; ++i) {
            scalar_t coarser_size = static_cast<scalar_t>(grid_scales[i]);
            scalar_t finer_size = static_cast<scalar_t>(grid_scales[i + 1]);
            if (psize_val <= coarser_size && psize_val >= finer_size) {
                scalar_t denom = coarser_size - finer_size;
                levels[0] = i;
                levels[1] = i + 1;
                layer_weights[0] = (psize_val - finer_size) / denom;
                layer_weights[1] = (coarser_size - psize_val) / denom;
                break;
            }
        }
    }

    for (int li = 0; li < 2; ++li) {
        int level = levels[li];
        scalar_t layer_weight = layer_weights[li];
        if (level < 0 || layer_weight == 0) continue;

        int res = resolutions[level];
        scalar_t3<scalar_t> pos_grid = pos3 * res;
        int3 base = pos_grid.to_int3();
        scalar_t3<scalar_t> offset = scalar_t3<scalar_t>(
            pos_grid.x - base.x,
            pos_grid.y - base.y,
            pos_grid.z - base.z
        );
        scalar_t *grad_grid_ptr = grad_grids[level];

        for (int corner = 0; corner < 8; ++corner) {
            int3 corner3 = _corner_offset(corner);
            int3 index3 = make_int3(
                base.x + corner3.x,
                base.y + corner3.y,
                base.z + corner3.z
            );
            uint32_t index = _hash_index(index3, level);
            scalar_t weight = (corner3.x ? offset.x : (1 - offset.x)) *
                              (corner3.y ? offset.y : (1 - offset.y)) *
                              (corner3.z ? offset.z : (1 - offset.z));

            for (int i = 0; i < DIMENSIONS; ++i) {
                atomicAdd(
                    &grad_grid_ptr[index * DIMENSIONS + i],
                    grad_ptr[i] * weight * layer_weight
                );
            }
        }
    }
}


// Host functions

template<typename scalar_t>
const scalar_t **tensor_list_to_device_ptrs(const std::vector<torch::Tensor>& tensors) {
    size_t n = tensors.size();
    std::vector<const scalar_t *> host_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
        host_ptrs[i] = tensors[i].contiguous().data_ptr<scalar_t>();
    }

    const scalar_t **device_ptrs;
    cudaMalloc(&device_ptrs, sizeof(const scalar_t *) * n);
    cudaMemcpy(device_ptrs, host_ptrs.data(), sizeof(const scalar_t *) * n, cudaMemcpyHostToDevice);
    return device_ptrs;
}

template<typename scalar_t>
scalar_t **tensor_list_to_device_mut_ptrs(const std::vector<torch::Tensor>& tensors) {
    size_t n = tensors.size();
    std::vector<scalar_t *> host_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
        host_ptrs[i] = tensors[i].contiguous().data_ptr<scalar_t>();
    }

    scalar_t **device_ptrs;
    cudaMalloc(&device_ptrs, sizeof(scalar_t *) * n);
    cudaMemcpy(device_ptrs, host_ptrs.data(), sizeof(scalar_t *) * n, cudaMemcpyHostToDevice);
    return device_ptrs;
}

void launch_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
) {
    auto N = pos.size(0);

    // Initalize resolution & grid size table
    int res_table[LEVELS];
    int grid_table[LEVELS];
    for (int i = 0; i < LEVELS; ++i) {
        int res = static_cast<int>(BASE_RESOLUTION * std::pow(PER_LEVEL_SCALE, i));
        res_table[i] = res;
        grid_table[i] = (res + 1) * (res + 1) * (res + 1);
    }
    cudaMemcpyToSymbol(resolutions, res_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_sizes, grid_table, sizeof(int) * LEVELS);

    // Launch device function
    int n_threads = THREADS;
    int n_blocks = (N * LEVELS + THREADS - 1) / THREADS;

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "forward_kernel", ([&] {
        // Move grids vector/list to device pointer array
        const scalar_t **device_ptrs = tensor_list_to_device_ptrs<scalar_t>(grids);

        forward_kernel<scalar_t><<<n_blocks, n_threads>>>(
            pos.data_ptr<scalar_t>(),
            device_ptrs,
            result.data_ptr<scalar_t>(),
            N
        );

        // Free device pointer array
        cudaFree(device_ptrs);
    }));
}

void launch_forward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
) {
    auto N = pos.size(0);

    // Initalize resolution & grid size table
    int res_table[LEVELS];
    int grid_table[LEVELS];
    float scale_table[LEVELS];
    for (int i = 0; i < LEVELS; ++i) {
        int res = static_cast<int>(BASE_RESOLUTION * std::pow(PER_LEVEL_SCALE, i));
        res_table[i] = res;
        grid_table[i] = (res + 1) * (res + 1) * (res + 1);
        scale_table[i] = INTERP_RATIO / res;
    }
    cudaMemcpyToSymbol(resolutions, res_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_sizes, grid_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_scales, scale_table, sizeof(float) * LEVELS);

    // Launch device function
    int n_threads = THREADS;
    int n_blocks = (N + THREADS - 1) / THREADS;

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "forward_layer_interp_kernel", ([&] {
        // Move grids vector/list to device pointer array
        const scalar_t **device_ptrs = tensor_list_to_device_ptrs<scalar_t>(grids);

        forward_layer_interp_kernel<scalar_t><<<n_blocks, n_threads>>>(
            pos.data_ptr<scalar_t>(),
            point_size.data_ptr<scalar_t>(),
            device_ptrs,
            result.data_ptr<scalar_t>(),
            N
        );

        // Free device pointer array
        cudaFree(device_ptrs);
    }));
}

void launch_backward(
    const torch::Tensor pos,
    const torch::Tensor grad_output,
    const std::vector<torch::Tensor>& grad_grids
) {
    auto N = pos.size(0);

    int res_table[LEVELS];
    int grid_table[LEVELS];
    for (int i = 0; i < LEVELS; ++i) {
        int res = static_cast<int>(BASE_RESOLUTION * std::pow(PER_LEVEL_SCALE, i));
        res_table[i] = res;
        grid_table[i] = (res + 1) * (res + 1) * (res + 1);
    }
    cudaMemcpyToSymbol(resolutions, res_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_sizes, grid_table, sizeof(int) * LEVELS);

    int n_threads = THREADS;
    int n_blocks = (N * LEVELS + THREADS - 1) / THREADS;

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "backward_kernel", ([&] {
        scalar_t **device_grad_ptrs = tensor_list_to_device_mut_ptrs<scalar_t>(grad_grids);

        backward_kernel<scalar_t><<<n_blocks, n_threads>>>(
            pos.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            device_grad_ptrs,
            N
        );

        cudaFree(device_grad_ptrs);
    }));
}

void launch_backward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const torch::Tensor grad_output,
    const std::vector<torch::Tensor>& grad_grids
) {
    auto N = pos.size(0);

    int res_table[LEVELS];
    int grid_table[LEVELS];
    float scale_table[LEVELS];
    for (int i = 0; i < LEVELS; ++i) {
        int res = static_cast<int>(BASE_RESOLUTION * std::pow(PER_LEVEL_SCALE, i));
        res_table[i] = res;
        grid_table[i] = (res + 1) * (res + 1) * (res + 1);
        scale_table[i] = INTERP_RATIO / res;
    }
    cudaMemcpyToSymbol(resolutions, res_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_sizes, grid_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_scales, scale_table, sizeof(float) * LEVELS);

    int n_threads = THREADS;
    int n_blocks = (N + THREADS - 1) / THREADS;

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "backward_layer_interp_kernel", ([&] {
        scalar_t **device_grad_ptrs = tensor_list_to_device_mut_ptrs<scalar_t>(grad_grids);

        backward_layer_interp_kernel<scalar_t><<<n_blocks, n_threads>>>(
            pos.data_ptr<scalar_t>(),
            point_size.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            device_grad_ptrs,
            N
        );

        cudaFree(device_grad_ptrs);
    }));
}
