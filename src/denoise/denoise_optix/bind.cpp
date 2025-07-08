// TODO: you should include the torch header first
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include "denoiser.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#ifdef _WIN32
#include <windows.h>
LONG WINAPI MyUnhandledExceptionFilter(struct _EXCEPTION_POINTERS *ExceptionInfo) {
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        MI_LOG_ERROR("Access violation caught, exiting gracefully\n");
        exit(1);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

void init_denoiser() {
    static bool is_init = false;
    if (is_init) {
        return;
    }

    if (is_init == false) {
#ifdef _WIN32
        SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
#endif
        is_init = true;
    }
}

torch::Tensor optix_denoise(const torch::Tensor &noisy, bool aux, bool temporal) {
    init_denoiser();
    const int height = noisy.size(0);
    const int width = noisy.size(1);
    const int channels = (noisy.size(2) - (aux ? 6 : 0));
    const int element_size = channels * sizeof(float);

    assert(element_size == sizeof(float3) || element_size == sizeof(float4));

    std::shared_ptr<Denoiser> denoiser = Denoiser::get_instance();
    denoiser->resize({width, height, element_size}, aux, temporal);
    if (aux) {
        const torch::Tensor color = noisy.slice(2, 0, channels).clone();
        const torch::Tensor albedo = noisy.slice(2, channels, channels + 3).clone();
        const torch::Tensor normal = noisy.slice(2, channels + 3, channels + 6).clone();
        return denoiser->denoise(&color, &albedo, &normal);
    } else {
        return denoiser->denoise(&noisy, nullptr, nullptr);
    }
}

torch::Tensor optix_denoise_separate(const torch::Tensor &noisy, const torch::Tensor &aux_tensor, bool aux, bool temporal) {
    init_denoiser();
    const int height = noisy.size(0);
    const int width = noisy.size(1);
    const int channels = (noisy.size(2));
    const int element_size = channels * sizeof(float);

    assert(element_size == sizeof(float3) || element_size == sizeof(float4));

    std::shared_ptr<Denoiser> denoiser = Denoiser::get_instance();
    denoiser->resize({width, height, element_size}, aux, temporal);
    if (aux) {
        const torch::Tensor color = noisy.clone();
        const torch::Tensor albedo = aux_tensor.slice(2, 0, 3).clone();
        const torch::Tensor normal = aux_tensor.slice(2, 3, 6).clone();
        return denoiser->denoise(&color, &albedo, &normal);
    } else {
        return denoiser->denoise(&noisy, nullptr, nullptr);
    }
}

void free_denoiser() {
    std::shared_ptr<Denoiser> denoiser = Denoiser::get_instance();
    denoiser->free_instance();
}

PYBIND11_MODULE(setup_optix_example, m) {
    m.def("denoise", &optix_denoise, "Optix Denoiser");
    m.def("denoise_separate", &optix_denoise_separate, "Optix Denoiser");
    m.def("free_denoiser", &free_denoiser, "Free Optix Denoiser");
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
