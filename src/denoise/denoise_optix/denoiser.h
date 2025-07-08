#pragma once
// TODO: you should include the torch header first
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "optix_helper.h"
#include <iostream>
#include <memory>

class Denoiser {
private:
    bool m_initialized = false;
    bool m_aux = false;
    bool m_temporal = false;
    bool m_have_previous_denoised_img = false;
    torch::Tensor m_zero_flow = {};

    OptixDenoiserSizes m_memory_sizes = {};
    torch::Tensor m_previous_denoised_img = {};
    torch::Tensor m_internal_mem_in = {};
    torch::Tensor m_internal_mem_out = {};

    // CUstream m_stream = nullptr;
    cudaDeviceProp m_device_props = {};
    CUcontext m_cuda_context = nullptr;
    OptixDeviceContext m_optix_context = nullptr;

    OptixDenoiser m_denoiser = nullptr;
    torch::Tensor m_denoiser_scratch = {};
    torch::Tensor m_denoiser_state = {};
    torch::Tensor m_denoised_img = {};
    int3 m_size = {0, 0, 0};

public:
    // Delete copy constructor and assignment operator
    Denoiser();
    Denoiser(const Denoiser &) = delete;
    Denoiser &operator=(const Denoiser &) = delete;

    // Singleton
    static std::shared_ptr<Denoiser> s_instance;
    static std::shared_ptr<Denoiser> get_instance();
    static void free_instance();
    ~Denoiser();
    void init();
    void resize(const int3 size, const bool aux = false, const bool temporal = false);
    void resize_denoised_images();
    void resize_temporal_images();
    void resize_optix_denoiser();

    torch::Tensor denoise(const torch::Tensor *img_with_noise,
                          const torch::Tensor *albedo,
                          const torch::Tensor *normal);

    static void context_log_cb(unsigned int level,
                               const char *tag,
                               const char *message,
                               void *);
};
