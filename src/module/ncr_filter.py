import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load


class NCRFilter:
    def __init__(
        self,
        width: int,
        height: int,
        kernel_size: int = 5,
        fxaa: bool = True,
        bilateral: bool = False,
        bilateral_sigma_spatial: float = 2.0,
        bilateral_sigma_range: float = 0.1,
        bilateral_sigma_color: float = 0.25,
        roughness_threshold: float = 0.5,
        use_kernel: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.use_fxaa = fxaa
        self.use_bilateral = bilateral
        self.bilateral_sigma_spatial = bilateral_sigma_spatial
        self.bilateral_sigma_range = bilateral_sigma_range
        self.bilateral_sigma_color = bilateral_sigma_color
        self.roughness_threshold = roughness_threshold
        self.use_kernel = use_kernel
        self.cuda_kernel = None

    def load_kernel(self, kernel_name: str = "ncr_filter_cuda") -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cuda_kernel = load(
            name=kernel_name,
            sources=[
                os.path.join(current_dir, "ncr_filter_cuda", "ncr_filter_bindings.cpp"),
                os.path.join(current_dir, "ncr_filter_cuda", "ncr_filter_cuda.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=True,
        )

    def _ensure_kernel_loaded(self) -> None:
        if self.cuda_kernel is None:
            self.load_kernel()

    def _check_image(self, img: torch.Tensor) -> None:
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Expected img with shape [H, W, 3], got {tuple(img.shape)}")
        if not img.is_cuda:
            raise ValueError("NCRFilter expects img to be a CUDA tensor")
        if img.dtype != torch.float32:
            raise ValueError(f"NCRFilter currently expects float32 input, got {img.dtype}")

        height, width = img.shape[:2]
        if height != self.height or width != self.width:
            self.height = height
            self.width = width

    def _check_aux_tensor(self, name: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            return
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor when provided")
        if tensor.ndim < 2:
            raise ValueError(f"{name} must have at least 2 dimensions, got {tuple(tensor.shape)}")
        if tensor.shape[0] != self.height or tensor.shape[1] != self.width:
            raise ValueError(
                f"{name} must match image spatial size [{self.height}, {self.width}], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} currently expects float32 input, got {tensor.dtype}")

    def _check_feature_tensor(self, name: str, tensor: Optional[torch.Tensor], channels: int) -> None:
        self._check_aux_tensor(name, tensor)
        if tensor is None:
            return
        if tensor.ndim != 3 or tensor.shape[2] != channels:
            raise ValueError(
                f"{name} must have shape [H, W, {channels}], got {tuple(tensor.shape)}"
            )

    def _check_roughness_tensor(self, roughness: Optional[torch.Tensor]) -> None:
        self._check_aux_tensor("roughness", roughness)
        if roughness is None:
            return
        if roughness.ndim == 2:
            return
        if roughness.ndim == 3 and roughness.shape[2] == 1:
            return
        raise ValueError(
            f"roughness must have shape [H, W] or [H, W, 1], got {tuple(roughness.shape)}"
        )

    def apply(
        self,
        img: torch.Tensor,
        position: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        albedo: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._check_image(img)
        self._check_feature_tensor("position", position, 3)
        self._check_feature_tensor("normal", normal, 3)
        self._check_feature_tensor("albedo", albedo, 3)
        self._check_roughness_tensor(roughness)

        if not self.use_kernel or (not self.use_fxaa and not self.use_bilateral):
            return img

        if self.use_bilateral:
            missing = [
                name
                for name, tensor in (
                    ("position", position),
                    ("normal", normal),
                    ("albedo", albedo),
                    ("roughness", roughness),
                )
                if tensor is None
            ]
            if missing:
                raise ValueError(
                    "Cross bilateral filter requires position, normal, albedo, and roughness; "
                    f"missing {', '.join(missing)}"
                )

        self._ensure_kernel_loaded()

        return self.cuda_kernel.forward(
            img.contiguous(),
            None if position is None else position.contiguous(),
            None if normal is None else normal.contiguous(),
            None if albedo is None else albedo.contiguous(),
            None if roughness is None else roughness.contiguous(),
            self.kernel_size,
            self.use_bilateral,
            self.use_fxaa,
            self.bilateral_sigma_spatial,
            self.bilateral_sigma_range,
            self.bilateral_sigma_color,
            self.roughness_threshold,
        )

    __call__ = apply
