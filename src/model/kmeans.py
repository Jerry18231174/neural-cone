import numpy as np
import torch
import torch.nn.functional as F

import os
from torch.utils.cpp_extension import load


class KMeans:
    """
    KMeans clustering algorithm.
    """
    def __init__(self, n_clusters: int, n_iter: int = 10, use_kernel: bool = False):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.use_kernel = use_kernel

        if use_kernel:
            self.set_kernel()
        
    def load_kernel(self):
        self.use_kernel = True

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cuda_kernel = load(
            name="kmeans_cuda",
            sources=[
                os.path.join(current_dir, "kmeans_cuda", "kmeans1d_bindings.cpp"),
                os.path.join(current_dir, "kmeans_cuda", "kmeans1d_cuda.cu")],
            extra_cflags=[
                '-O3',
                "-DN_CLUSTERS={}".format(self.n_clusters)
            ],
            extra_cuda_cflags=[
                "-O3", "-g", "-lineinfo", "-Xcompiler", "-rdynamic",
                "-DN_CLUSTERS={}".format(self.n_clusters)
            ],
            verbose=True,
        )

    def fit(self, t: torch.Tensor, precision=torch.float32):
        """
        Perform K-means clustering on the input samples.

        Args:
            t (torch.Tensor): Input samples of shape (N, D),
            where N is the number of samples and D is the dimensionality.

        Returns:
            torch.Tensor: Cluster centers of shape (N, n_clusters).
            torch.Tensor: Number of points in each cluster of shape (N, n_clusters).
            torch.Tensor: Standard deviation of each cluster of shape (N, n_clusters).
        """
        if self.use_kernel:
            return self.cuda_kernel.kmeans1d(t, self.n_iter)
        N = t.shape[0]
        inf_mask = torch.isinf(t)
        t_fill0 = t.masked_fill(inf_mask, 0)

        # Initialize cluster centers uniformly between the min and max of the samples
        t_min = torch.min(t.masked_fill(inf_mask, float('inf')), dim=1, keepdim=True).values
        t_max = torch.max(t.masked_fill(inf_mask, -float('inf')), dim=1, keepdim=True).values
        t_mu = torch.rand(N, self.n_clusters, device=t.device, dtype=precision) * (t_max - t_min) + t_min
        cluster_size = torch.zeros_like(t_mu)
        cluster_ids = torch.arange(self.n_clusters, device=t.device, dtype=precision)[None, :, None]  # [1, n_clusters, 1]
        
        for i in range(self.n_iter):
            # Compute distances from samples to cluster centers     [N, n_clusters, D]
            t_dist = torch.abs(t_fill0[:, None, :] - t_mu[:, :, None])
            # Find the closest cluster center for each sample       [N, D]
            cluster_idx = torch.min(t_dist, dim=1).indices
            # Set invalid samples' cluster to -1
            cluster_idx[inf_mask] = -1
            
            # mask = torch.zeros_like(t_dist, dtype=torch.bool)
            # for j in range(self.n_clusters):
            #     mask[:, j, :] = (cluster_idx == j)
            
            cluster_idx_exp = cluster_idx.unsqueeze(1)  # [N, 1, D]
            mask = (cluster_idx_exp == cluster_ids)     # [N, n_clusters, D]

            # Update cluster centers and cluster size               [N, n_clusters, D]
            cluster_size = torch.sum((mask + 1e-8), dim=-1, dtype=precision)
            t_mu = torch.sum(t_fill0[:, None, :] * (mask + 1e-8), dim=-1, dtype=precision) / cluster_size
        
        # Compute standard deviation
        t_dist = (t_fill0[:, None, :] - t_mu[:, :, None]) ** 2
        t_var = torch.sum(t_dist * (mask + 1e-8), dim=-1, dtype=precision) / cluster_size
        t_std = torch.sqrt(t_var)
        
        return t_mu, cluster_size, t_std

    def fit_3d(self, p: torch.Tensor, dir: torch.Tensor = None, precision=torch.float32):
        """
        Perform 3-dim K-means clustering on the input samples.

        Args:
            p (torch.Tensor): Input samples of shape (N, D, 3),
            dir (torch.Tensor): Input directions of shape (N, D, 3),
            where N is the number of samples and D is the dimensionality.

        Returns:
            torch.Tensor: Cluster centers of shape (N, n_clusters, 3).
            torch.Tensor: Number of points in each cluster of shape (N, n_clusters, 1).
            torch.Tensor: Isotropic standard deviation of each cluster of shape (N, n_clusters, 1).
            torch.Tensor: Direction of each cluster of shape (N, n_clusters, 3).
        """
        N = p.shape[0]
        inf_mask = torch.any(torch.isinf(p), dim=-1, keepdim=True)
        p_fill0 = p.masked_fill(inf_mask, 0)

        # Initialize cluster centers uniformly between the min and max of the samples
        # [N, 1, 3]
        p_min = torch.min(p.masked_fill(inf_mask, float('inf')), dim=1, keepdim=True).values
        p_max = torch.max(p.masked_fill(inf_mask, -float('inf')), dim=1, keepdim=True).values
        # [N, n_clusters, 3]
        p_mu = torch.rand(N, self.n_clusters, 3, device=p.device, dtype=precision) * (p_max - p_min) + p_min
        # [N, n_clusters, 1]
        cluster_size = torch.zeros_like(p_mu)[:, :, 0:1]
        # [1, n_clusters, 1]
        cluster_ids = torch.arange(self.n_clusters, device=p.device, dtype=precision)[None, :, None]
        
        for i in range(self.n_iter):
            # Compute distances from samples to cluster centers     [N, n_clusters, D, 3]
            p_dist = torch.abs(p_fill0[:, None, :, :] - p_mu[:, :, None, :])
            p_dist = torch.norm(p_dist, p=2, dim=-1)              # [N, n_clusters, D]
            # Find the closest cluster center for each sample       [N, D]
            cluster_idx = torch.min(p_dist, dim=1).indices
            # Set invalid samples' cluster to -1
            cluster_idx[inf_mask.squeeze()] = -1
            
            cluster_idx_exp = cluster_idx.unsqueeze(1)  # [N, 1, D]
            mask = (cluster_idx_exp == cluster_ids)     # [N, n_clusters, D]

            # Update cluster centers and cluster size
            # [N, n_clusters, 1]
            cluster_size = torch.sum((mask + 1e-8), dim=-1, keepdim=True, dtype=precision)
            # [N, n_clusters, 3]
            p_mu = torch.sum(p_fill0[:, None, :, :] * (mask + 1e-8)[..., None], dim=2, dtype=precision) / cluster_size
        
        # Compute standard deviation
        # [N, n_clusters, D]
        p_dist = torch.norm(p_fill0[:, None, :, :] - p_mu[:, :, None, :], p=2, dim=-1) ** 2
        # [N, n_clusters, 1]
        p_var = torch.sum(p_dist * (mask + 1e-8), dim=-1, keepdim=True, dtype=precision) / cluster_size
        p_std = torch.sqrt(p_var)
        
        # Compute direction
        # [N, n_clusters, 3]
        dir_mu = torch.sum(dir[:, None, :, :] * (mask + 1e-8)[..., None], dim=2, dtype=precision)
        dir_mu = dir_mu / torch.norm(dir_mu, p=2, dim=-1, keepdim=True)

        return p_mu, cluster_size, p_std, dir_mu