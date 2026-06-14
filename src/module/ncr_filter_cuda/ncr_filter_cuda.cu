#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr int kThreadsX = 16;
constexpr int kThreadsY = 16;
constexpr float kFXAAReduceMin = 1.0f / 128.0f;
constexpr float kFXAAReduceMul = 1.0f / 8.0f;
constexpr float kFXAASpanMax = 8.0f;

struct Float3 {
    float x;
    float y;
    float z;
};

__device__ __forceinline__ Float3 make_float3_(float x, float y, float z) {
    Float3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

__device__ __forceinline__ Float3 add(const Float3& a, const Float3& b) {
    return make_float3_(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Float3 sub(const Float3& a, const Float3& b) {
    return make_float3_(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Float3 mul(const Float3& a, float s) {
    return make_float3_(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ Float3 maxf3(const Float3& a, float v) {
    return make_float3_(fmaxf(a.x, v), fmaxf(a.y, v), fmaxf(a.z, v));
}

__device__ __forceinline__ float dot_luma(const Float3& a) {
    return a.x * 0.299f + a.y * 0.587f + a.z * 0.114f;
}

__device__ __forceinline__ float dot3(const Float3& a, const Float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

__device__ __forceinline__ float clamp_float(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

__device__ __forceinline__ Float3 load_rgb(
    const float* img,
    int height,
    int width,
    int y,
    int x
) {
    const int cy = clamp_int(y, 0, height - 1);
    const int cx = clamp_int(x, 0, width - 1);
    const int offset = (cy * width + cx) * 3;
    return make_float3_(img[offset + 0], img[offset + 1], img[offset + 2]);
}

__device__ __forceinline__ float load_scalar(
    const float* img,
    int height,
    int width,
    int y,
    int x
) {
    const int cy = clamp_int(y, 0, height - 1);
    const int cx = clamp_int(x, 0, width - 1);
    return img[cy * width + cx];
}

__device__ __forceinline__ Float3 log1p_rgb(const Float3& a) {
    const Float3 clamped = maxf3(a, 0.0f);
    return make_float3_(log1pf(clamped.x), log1pf(clamped.y), log1pf(clamped.z));
}

__device__ __forceinline__ Float3 apply_cross_bilateral(
    const float* img,
    int height,
    int width,
    int y,
    int x,
    const float* position,
    const float* normal,
    const float* albedo,
    const float* roughness,
    int kernel_radius,
    float sigma_spatial,
    float sigma_range,
    float sigma_color,
    float roughness_threshold
) {
    const Float3 center_color = load_rgb(img, height, width, y, x);
    if (roughness == nullptr || position == nullptr || normal == nullptr || albedo == nullptr) {
        return center_color;
    }

    const float center_roughness = load_scalar(roughness, height, width, y, x);
    if (center_roughness >= roughness_threshold) {
        return center_color;
    }

    const Float3 center_position = load_rgb(position, height, width, y, x);
    const Float3 center_normal = load_rgb(normal, height, width, y, x);
    const Float3 center_albedo = load_rgb(albedo, height, width, y, x);
    const Float3 center_log_color = log1p_rgb(center_color);

    const float spatial_denom = fmaxf(2.0f * sigma_spatial * sigma_spatial, 1e-8f);
    const float range_denom = fmaxf(2.0f * sigma_range * sigma_range, 1e-8f);
    const float color_denom = fmaxf(2.0f * sigma_color * sigma_color, 1e-8f);

    Float3 accum = make_float3_(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    for (int j = -kernel_radius; j <= kernel_radius; ++j) {
        for (int i = -kernel_radius; i <= kernel_radius; ++i) {
            const int ny = y + j;
            const int nx = x + i;
            const Float3 sample_color = load_rgb(img, height, width, ny, nx);
            const Float3 sample_position = load_rgb(position, height, width, ny, nx);
            const Float3 sample_normal = load_rgb(normal, height, width, ny, nx);
            const Float3 sample_albedo = load_rgb(albedo, height, width, ny, nx);
            const Float3 sample_log_color = log1p_rgb(sample_color);

            const Float3 pos_diff = sub(sample_position, center_position);
            const Float3 normal_diff = sub(sample_normal, center_normal);
            const Float3 albedo_diff = sub(sample_albedo, center_albedo);
            const Float3 color_diff = sub(sample_log_color, center_log_color);

            const float spatial_dist2 = static_cast<float>(i * i + j * j);
            const float guide_dist2 =
                dot3(pos_diff, pos_diff) +
                dot3(normal_diff, normal_diff) +
                dot3(albedo_diff, albedo_diff);
            const float color_dist2 = dot3(color_diff, color_diff);
            const float weight =
                expf(-spatial_dist2 / spatial_denom) *
                expf(-guide_dist2 / range_denom) *
                expf(-color_dist2 / color_denom);

            accum = add(accum, mul(sample_color, weight));
            weight_sum += weight;
        }
    }

    if (weight_sum <= 0.0f) {
        return center_color;
    }
    return mul(accum, 1.0f / weight_sum);
}

__device__ __forceinline__ Float3 filtered_integer_sample(
    const float* img,
    int height,
    int width,
    int y,
    int x
) {
    return load_rgb(img, height, width, y, x);
}

__device__ __forceinline__ Float3 filtered_bilateral_sample(
    const float* img,
    int height,
    int width,
    int y,
    int x,
    const float* position,
    const float* normal,
    const float* albedo,
    const float* roughness,
    int kernel_radius,
    float sigma_spatial,
    float sigma_range,
    float sigma_color,
    float roughness_threshold
) {
    return apply_cross_bilateral(
        img,
        height,
        width,
        y,
        x,
        position,
        normal,
        albedo,
        roughness,
        kernel_radius,
        sigma_spatial,
        sigma_range,
        sigma_color,
        roughness_threshold
    );
}

__device__ __forceinline__ Float3 filtered_bilinear_sample(
    const float* img,
    int height,
    int width,
    float y,
    float x
) {
    const float sample_x = clamp_float(x, 0.5f, static_cast<float>(width) - 0.5f);
    const float sample_y = clamp_float(y, 0.5f, static_cast<float>(height) - 0.5f);

    const float px = sample_x - 0.5f;
    const float py = sample_y - 0.5f;
    const int x0 = static_cast<int>(floorf(px));
    const int y0 = static_cast<int>(floorf(py));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = px - x0;
    const float ty = py - y0;

    const Float3 c00 = filtered_integer_sample(img, height, width, y0, x0);
    const Float3 c10 = filtered_integer_sample(img, height, width, y0, x1);
    const Float3 c01 = filtered_integer_sample(img, height, width, y1, x0);
    const Float3 c11 = filtered_integer_sample(img, height, width, y1, x1);

    Float3 top = add(mul(c00, 1.0f - tx), mul(c10, tx));
    Float3 bottom = add(mul(c01, 1.0f - tx), mul(c11, tx));
    return add(mul(top, 1.0f - ty), mul(bottom, ty));
}

__device__ __forceinline__ Float3 fxaa_pixel(
    const float* img,
    int height,
    int width,
    int y,
    int x
) {
    const Float3 rgb_nw = filtered_integer_sample(img, height, width, y - 1, x - 1);
    const Float3 rgb_ne = filtered_integer_sample(img, height, width, y - 1, x + 1);
    const Float3 rgb_sw = filtered_integer_sample(img, height, width, y + 1, x - 1);
    const Float3 rgb_se = filtered_integer_sample(img, height, width, y + 1, x + 1);
    const Float3 rgb_m = filtered_integer_sample(img, height, width, y, x);

    const float luma_nw = dot_luma(rgb_nw);
    const float luma_ne = dot_luma(rgb_ne);
    const float luma_sw = dot_luma(rgb_sw);
    const float luma_se = dot_luma(rgb_se);
    const float luma_m = dot_luma(rgb_m);
    const float luma_min = fminf(luma_m, fminf(fminf(luma_nw, luma_ne), fminf(luma_sw, luma_se)));
    const float luma_max = fmaxf(luma_m, fmaxf(fmaxf(luma_nw, luma_ne), fmaxf(luma_sw, luma_se)));

    float dir_x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
    float dir_y = ((luma_nw + luma_sw) - (luma_ne + luma_se));

    const float dir_reduce = fmaxf(
        (luma_nw + luma_ne + luma_sw + luma_se) * (0.25f * kFXAAReduceMul),
        kFXAAReduceMin
    );
    const float rcp_dir_min = 1.0f / (fminf(fabsf(dir_x), fabsf(dir_y)) + dir_reduce);
    dir_x = clamp_float(dir_x * rcp_dir_min, -kFXAASpanMax, kFXAASpanMax);
    dir_y = clamp_float(dir_y * rcp_dir_min, -kFXAASpanMax, kFXAASpanMax);

    const float frag_x = static_cast<float>(x) + 0.5f;
    const float frag_y = static_cast<float>(y) + 0.5f;

    const Float3 sample_a0 = filtered_bilinear_sample(
        img, height, width, frag_y + dir_y * (1.0f / 3.0f - 0.5f), frag_x + dir_x * (1.0f / 3.0f - 0.5f)
    );
    const Float3 sample_a1 = filtered_bilinear_sample(
        img, height, width, frag_y + dir_y * (2.0f / 3.0f - 0.5f), frag_x + dir_x * (2.0f / 3.0f - 0.5f)
    );
    const Float3 rgb_a = mul(add(sample_a0, sample_a1), 0.5f);

    const Float3 sample_b0 = filtered_bilinear_sample(
        img, height, width, frag_y + dir_y * -0.5f, frag_x + dir_x * -0.5f
    );
    const Float3 sample_b1 = filtered_bilinear_sample(
        img, height, width, frag_y + dir_y * 0.5f, frag_x + dir_x * 0.5f
    );
    const Float3 rgb_b = add(mul(rgb_a, 0.5f), mul(add(sample_b0, sample_b1), 0.25f));

    const float luma_b = dot_luma(rgb_b);
    if (luma_b < luma_min || luma_b > luma_max) {
        return rgb_a;
    }
    return rgb_b;
}

__global__ void bilateral_filter_kernel(
    const float* img,
    const float* position,
    const float* normal,
    const float* albedo,
    const float* roughness,
    int height,
    int width,
    int kernel_radius,
    float sigma_spatial,
    float sigma_range,
    float sigma_color,
    float roughness_threshold,
    float* tmp
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    Float3 color = filtered_bilateral_sample(
        img,
        height,
        width,
        y,
        x,
        position,
        normal,
        albedo,
        roughness,
        kernel_radius,
        sigma_spatial,
        sigma_range,
        sigma_color,
        roughness_threshold
    );

    const int offset = (y * width + x) * 3;
    tmp[offset + 0] = color.x;
    tmp[offset + 1] = color.y;
    tmp[offset + 2] = color.z;
}

__global__ void fxaa_kernel(
    const float* img,
    int height,
    int width,
    float* output
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const Float3 color = fxaa_pixel(img, height, width, y, x);
    const int offset = (y * width + x) * 3;
    output[offset + 0] = color.x;
    output[offset + 1] = color.y;
    output[offset + 2] = color.z;
}

}  // namespace

void launch_forward(
    const torch::Tensor& img,
    const c10::optional<torch::Tensor>& position,
    const c10::optional<torch::Tensor>& normal,
    const c10::optional<torch::Tensor>& albedo,
    const c10::optional<torch::Tensor>& roughness,
    int64_t kernel_size,
    bool use_bilateral,
    bool use_fxaa,
    double bilateral_sigma_spatial,
    double bilateral_sigma_range,
    double bilateral_sigma_color,
    double roughness_threshold,
    torch::Tensor output
) {
    const int height = static_cast<int>(img.size(0));
    const int width = static_cast<int>(img.size(1));
    const int kernel_radius = std::max<int>(0, static_cast<int>(kernel_size) / 2);

    const dim3 threads(kThreadsX, kThreadsY);
    const dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    if (!use_bilateral && !use_fxaa) {
        output.copy_(img);
        return;
    }

    const float* fxaa_input = img.data_ptr<float>();
    torch::Tensor tmp;

    if (use_bilateral) {
        tmp = torch::zeros_like(img);
        bilateral_filter_kernel<<<blocks, threads>>>(
            img.data_ptr<float>(),
            position.has_value() ? position->data_ptr<float>() : nullptr,
            normal.has_value() ? normal->data_ptr<float>() : nullptr,
            albedo.has_value() ? albedo->data_ptr<float>() : nullptr,
            roughness.has_value() ? roughness->data_ptr<float>() : nullptr,
            height,
            width,
            kernel_radius,
            static_cast<float>(bilateral_sigma_spatial),
            static_cast<float>(bilateral_sigma_range),
            static_cast<float>(bilateral_sigma_color),
            static_cast<float>(roughness_threshold),
            tmp.data_ptr<float>()
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "ncr_filter bilateral kernel launch failed: ", cudaGetErrorString(err));
        fxaa_input = tmp.data_ptr<float>();
    }

    if (use_fxaa) {
        fxaa_kernel<<<blocks, threads>>>(
            fxaa_input,
            height,
            width,
            output.data_ptr<float>()
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "ncr_filter FXAA kernel launch failed: ", cudaGetErrorString(err));
        return;
    }

    output.copy_(tmp);
}
