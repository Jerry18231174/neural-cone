#include "denoiser.h"

std::shared_ptr<Denoiser> Denoiser::s_instance = nullptr;

std::shared_ptr<Denoiser> Denoiser::get_instance() {
    if (Denoiser::s_instance.get() == nullptr) {
        Denoiser::s_instance = std::make_shared<Denoiser>();
    }
    return Denoiser::s_instance;
}

void Denoiser::free_instance() {
    if (Denoiser::s_instance.get() != nullptr) {
        Denoiser::s_instance.reset();
    }
}

Denoiser::Denoiser() {
    init();
}

Denoiser::~Denoiser() {
    try {
        if (m_denoiser != nullptr) {
            OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
        }

        if (m_initialized) {
            OPTIX_CHECK(optixDeviceContextDestroy(m_optix_context));
            // CUDA_CHECK(cudaStreamDestroy(m_stream));
        }
    } catch (const std::exception &e) {
        MI_LOG_E("Error when destroying denoiser: %s\n", e.what());
    } catch (...) {
        MI_LOG_E("Error when destroying denoiser: unknown error\n");
    }
}

void Denoiser::init() {
    if (OPTIX_VERSION < 80000) {
        MI_LOG_E("Optix Should >= %d, yours is %d\n", 80000, OPTIX_VERSION);
        exit(1);
    }

    if (m_initialized) {
        return;
    }

    // init optix
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("[ERROR]: no CUDA capable devices found!");
    OPTIX_CHECK(optixInit());

    // create context
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    // CUDA_CHECK(cudaStreamCreate(&m_stream));

    cudaGetDeviceProperties(&m_device_props, deviceID);
    MI_LOG("optix version: %d\n", OPTIX_VERSION);
    MI_LOG("cuda version: %d\n", CUDART_VERSION);
    MI_LOG("[optix] device: %s\n", m_device_props.name);

    CUresult cuRes = cuCtxGetCurrent(&m_cuda_context);

    if (cuRes != CUDA_SUCCESS) {
        MI_LOG_E("Error querying current context: error code %d\n", cuRes);
    }

    OPTIX_CHECK(optixDeviceContextCreate(m_cuda_context, 0, &m_optix_context));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optix_context, context_log_cb, nullptr, 4));

    m_initialized = true;
}

void Denoiser::resize_denoised_images() {
    const int width = m_size.x;
    const int height = m_size.y;
    const int element_size = m_size.z;
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
    m_denoised_img = torch::empty({height, width, int(element_size / sizeof(float))}, tensor_options);
}

void Denoiser::resize_temporal_images() {
    const int width = m_size.x;
    const int height = m_size.y;
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
    m_zero_flow = torch::empty({height, width, 2}, tensor_options);
}

void Denoiser::resize_optix_denoiser() {
    if (m_denoiser != nullptr) {
        OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
    }

    const int width = m_size.x;
    const int height = m_size.y;
    const int element_size = m_size.z;

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiser_options = {};
    if (m_aux) {
        denoiser_options.guideNormal = 1;
        denoiser_options.guideAlbedo = 1;
    }
    const OptixDenoiserModelKind kind = m_temporal ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
    OPTIX_CHECK(optixDenoiserCreate(m_optix_context, kind, &denoiser_options, &m_denoiser));
    // .. then compute and allocate memory resources for the denoiser
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, width, height,
                                                    &m_memory_sizes));

    const int denoiser_scratch_size = (std::max)(m_memory_sizes.withOverlapScratchSizeInBytes,
                                                 m_memory_sizes.withoutOverlapScratchSizeInBytes);
    auto tensor_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA, 0).requires_grad(false);
    m_denoiser_scratch = torch::empty(denoiser_scratch_size, tensor_options);

    const int denoiser_state_size = m_memory_sizes.stateSizeInBytes;
    m_denoiser_state = torch::empty(denoiser_state_size, tensor_options);

    // ------------------------------------------------------------------
    OPTIX_CHECK(
        optixDenoiserSetup(m_denoiser, 0,
                           width, height,
                           (CUdeviceptr)m_denoiser_state.data_ptr(),
                           m_denoiser_state.sizes()[0],
                           (CUdeviceptr)m_denoiser_scratch.data_ptr(),
                           m_denoiser_scratch.sizes()[0]));
}

void Denoiser::resize(const int3 size, const bool aux, const bool temporal) {

    const bool change_size = (size != m_size);
    const bool change_setting = (m_size.x != size.x) || (m_size.y != size.y) || (m_temporal != temporal) || (m_aux != aux);
    const bool change_temporal = (m_temporal != temporal);

    // update
    m_have_previous_denoised_img = !change_temporal;
    m_size = size;
    m_temporal = temporal;
    m_aux = aux;

    // temporal buffers
    if (change_temporal || change_size) {
        resize_temporal_images();
    }

    // output denoised image
    if (change_size) {
        resize_denoised_images();
    }

    // optix setup
    if (change_setting) {
        resize_optix_denoiser();
    }
}

torch::Tensor Denoiser::denoise(const torch::Tensor *img_with_noise,
                                const torch::Tensor *albedo,
                                const torch::Tensor *normal) {
    // guard
    if (m_aux) {
        assert(albedo != nullptr);
        assert(normal != nullptr);
    }

    const int height = img_with_noise->size(0);
    const int width = img_with_noise->size(1);
    const int element_size = img_with_noise->size(2) * sizeof(float);

    // TODO: const vars for now
    const bool accumulate = false;
    const int frame_id = 10;

    OptixDenoiserParams denoiserParams = {};
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    if (accumulate) {
        denoiserParams.blendFactor = 1.f / frame_id;
    } else {
        denoiserParams.blendFactor = 0.0f;
    }

    // -------------------------------------------------------
    OptixImage2D inputLayer = {};
    inputLayer.data = (CUdeviceptr)img_with_noise->data_ptr();
    /// Width of the image (in pixels)
    inputLayer.width = width;
    /// Height of the image (in pixels)
    inputLayer.height = height;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = width * element_size;
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = element_size;
    /// Pixel format.
    inputLayer.format = (element_size == sizeof(float4)) ? OPTIX_PIXEL_FORMAT_FLOAT4 : OPTIX_PIXEL_FORMAT_FLOAT3;

    // -------------------------------------------------------
    OptixImage2D outputLayer = inputLayer;
    outputLayer.data = (CUdeviceptr)m_denoised_img.data_ptr();

    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;
    if (m_aux) {
        // albedo
        inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;
        inputLayer.data = (CUdeviceptr)albedo->data_ptr();
        denoiserGuideLayer.albedo = inputLayer;
        // normal
        inputLayer.data = (CUdeviceptr)normal->data_ptr();
        denoiserGuideLayer.normal = inputLayer;
    }

    if (m_temporal) {
        // flow
        inputLayer.data = (CUdeviceptr)m_zero_flow.data_ptr();
        inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT2;
        denoiserGuideLayer.flow = inputLayer;
        // flow trustworthiness
        inputLayer.data = (CUdeviceptr) nullptr;
        inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT1;
        denoiserGuideLayer.flowTrustworthiness = inputLayer;
        // last frame
        inputLayer.format = (element_size == sizeof(float4)) ? OPTIX_PIXEL_FORMAT_FLOAT4 : OPTIX_PIXEL_FORMAT_FLOAT3;
        if (m_have_previous_denoised_img) {
            inputLayer.data = (CUdeviceptr)m_previous_denoised_img.data_ptr();
        } else {
            inputLayer.data = (CUdeviceptr)img_with_noise->data_ptr();
        }
        denoiserLayer.previousOutput = inputLayer;

        // previous output & output internal guide layer
        size_t internal_size = width * height * sizeof(float) * m_memory_sizes.internalGuideLayerPixelSizeInBytes;
        auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
        // TODO: resize instead of new
        m_internal_mem_in = torch::zeros(internal_size, tensor_options);
        m_internal_mem_out = torch::empty(internal_size, tensor_options);
        inputLayer.data = (CUdeviceptr)m_internal_mem_in.data_ptr();
        inputLayer.pixelStrideInBytes = (unsigned int)m_memory_sizes.internalGuideLayerPixelSizeInBytes;
        inputLayer.rowStrideInBytes = (unsigned int)(width * m_memory_sizes.internalGuideLayerPixelSizeInBytes);
        inputLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
        denoiserGuideLayer.previousOutputInternalGuideLayer = inputLayer;
        denoiserGuideLayer.outputInternalGuideLayer = inputLayer;
    }

    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser,
                                    /*stream*/ 0,
                                    &denoiserParams,
                                    (CUdeviceptr)m_denoiser_state.data_ptr(),
                                    m_denoiser_state.sizes()[0],
                                    &denoiserGuideLayer,
                                    &denoiserLayer, 1,
                                    /*inputOffsetX*/ 0,
                                    /*inputOffsetY*/ 0,
                                    (CUdeviceptr)m_denoiser_scratch.data_ptr(),
                                    m_denoiser_scratch.sizes()[0]));

    if (m_temporal) {
        m_previous_denoised_img = m_denoised_img.clone();
        m_have_previous_denoised_img = true;
    }

    return m_denoised_img;
}

void Denoiser::context_log_cb(unsigned int level,
                              const char *tag,
                              const char *message,
                              void *) {
    MI_LOG("[%2d][%12s]: %s\n", (int)level, tag, message);
}
