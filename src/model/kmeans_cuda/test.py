import os
import time
import torch
from torch.utils.cpp_extension import load

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 设置为1以便于调试CUDA错误
# os.environ["TORCH_USE_CUDA_DSA"] = "1"  # 启用CUDA DSA（Device Side Allocation）

# 自动编译 & 加载 Extension
kMeans = load(
    name="kmeans_cuda",
    sources=["kmeans1d_bindings.cpp", "kmeans1d_cuda.cu"],
    extra_cflags=['-O3'],
    extra_cuda_cflags=["-O3", "-g", "-lineinfo", "-Xcompiler", "-rdynamic"],
    verbose=True,  # 可看到详细编译过程
)

# 示例：运行 K-means
N, D, K = 2**20, 128, 4
t = torch.randn(N, D, device='cuda', dtype=torch.float32)

torch.cuda.synchronize()
t0 = time.time()
centers, counts, stds = kMeans.kmeans1d(t, K, 3)
torch.cuda.synchronize()
t1 = time.time()

print(f"CUDA KMeans time: {t1 - t0:.4f} seconds")