#include <torch/extension.h>
#include <c10/util/Optional.h>

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
);

namespace {

void check_input(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_optional_input(
    const c10::optional<torch::Tensor>& tensor,
    const char* name,
    int64_t height,
    int64_t width
) {
    if (!tensor.has_value()) {
        return;
    }

    check_input(*tensor, name);
    TORCH_CHECK(
        tensor->dim() >= 2 && tensor->size(0) == height && tensor->size(1) == width,
        name,
        " must match image spatial size [",
        height,
        ", ",
        width,
        "]"
    );
}

void check_optional_channels(
    const c10::optional<torch::Tensor>& tensor,
    const char* name,
    int64_t channels
) {
    if (!tensor.has_value()) {
        return;
    }

    TORCH_CHECK(
        tensor->dim() == 3 && tensor->size(2) == channels,
        name,
        " must have shape [H, W, ",
        channels,
        "]"
    );
}

void check_optional_roughness(const c10::optional<torch::Tensor>& tensor) {
    if (!tensor.has_value()) {
        return;
    }

    TORCH_CHECK(
        tensor->dim() == 2 || (tensor->dim() == 3 && tensor->size(2) == 1),
        "roughness must have shape [H, W] or [H, W, 1]"
    );
}

}  // namespace

torch::Tensor forward(
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
    double roughness_threshold
) {
    check_input(img, "img");
    TORCH_CHECK(img.dim() == 3, "img must have shape [H, W, 3]");
    TORCH_CHECK(img.size(2) == 3, "img must have 3 channels in the last dimension");
    TORCH_CHECK(kernel_size >= 1, "kernel_size must be >= 1");

    const auto height = img.size(0);
    const auto width = img.size(1);
    check_optional_input(position, "position", height, width);
    check_optional_input(normal, "normal", height, width);
    check_optional_input(albedo, "albedo", height, width);
    check_optional_input(roughness, "roughness", height, width);
    check_optional_channels(position, "position", 3);
    check_optional_channels(normal, "normal", 3);
    check_optional_channels(albedo, "albedo", 3);
    check_optional_roughness(roughness);

    if (use_bilateral) {
        TORCH_CHECK(position.has_value(), "position is required when use_bilateral=true");
        TORCH_CHECK(normal.has_value(), "normal is required when use_bilateral=true");
        TORCH_CHECK(albedo.has_value(), "albedo is required when use_bilateral=true");
        TORCH_CHECK(roughness.has_value(), "roughness is required when use_bilateral=true");
    }

    auto output = torch::zeros_like(img);
    launch_forward(
        img,
        position,
        normal,
        albedo,
        roughness,
        kernel_size,
        use_bilateral,
        use_fxaa,
        bilateral_sigma_spatial,
        bilateral_sigma_range,
        bilateral_sigma_color,
        roughness_threshold,
        output
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward,
        py::arg("img"),
        py::arg("position") = py::none(),
        py::arg("normal") = py::none(),
        py::arg("albedo") = py::none(),
        py::arg("roughness") = py::none(),
        py::arg("kernel_size") = 5,
        py::arg("use_bilateral") = false,
        py::arg("use_fxaa") = true,
        py::arg("bilateral_sigma_spatial") = 2.0,
        py::arg("bilateral_sigma_range") = 0.1,
        py::arg("bilateral_sigma_color") = 0.25,
        py::arg("roughness_threshold") = 0.5,
        "Fused NCR bilateral filter + FXAA forward pass (CUDA)"
    );
}
