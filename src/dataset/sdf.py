import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import trimesh
import mesh2sdf
import pysdf

import mitsuba as mi
mi.set_variant("cuda_rgb")


class SDFDataset(Dataset):
    """
    Signed Distance Field Dataset
    """
    def __init__(
        self,
        mesh_path: str,
        size: int = 1,
        batch_size: int = 2 ** 18,
        clip_threshold: float = 10.0
    ):
        # Configs
        self.size = size
        self.batch_size = batch_size
        self.clip_threshold = clip_threshold
        self.perturb = 0.01

        # Load mesh
        self.mesh = trimesh.load(mesh_path)

        # Normalize mesh to [-1, 1]
        self.bbox = {
            "min": np.min(self.mesh.vertices, axis=0) - 1e-2,
            "max": np.max(self.mesh.vertices, axis=0) + 1e-2
        }
        self.scale = np.max(self.bbox["max"] - self.bbox["min"]) / 2
        self.center = (self.bbox["min"] + self.bbox["max"]) / 2
        self.mesh.vertices = (self.mesh.vertices - self.center) / self.scale
        
        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)

        # Samples components
        self.sample_ratio = {
            "surface": 0.2,
            "nearby": 0.5,
            "random": 0.3
        }
        self.surface_size = int(self.batch_size * self.sample_ratio["surface"])
        self.nearby_size = int(self.batch_size * self.sample_ratio["nearby"])
        self.random_size = self.batch_size - self.surface_size - self.nearby_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, _):
        # Sample points
        # Points on the surface
        points_surface = self.mesh.sample(self.surface_size)
        # Points nearby the surface
        points_nearby = self.mesh.sample(self.nearby_size) + np.random.randn(self.nearby_size, 3) * self.perturb
        # Random points inside the bounding box
        points_random = np.random.rand(self.random_size, 3) * 2 - 1
        points = np.concatenate([points_surface, points_nearby, points_random], axis=0).astype(np.float32)

        # Compute SDF
        sdf_surface = np.zeros((self.surface_size, 1), dtype=np.float32)
        sdf_other = self.sdf_fn(np.concatenate([points_nearby, points_random], axis=0).astype(np.float32))[:, None]
        sdf = np.concatenate([sdf_surface, sdf_other], axis=0).astype(np.float32)

        # Clip SDF
        if self.clip_threshold is not None:
            sdf = np.clip(sdf, -self.clip_threshold, self.clip_threshold)
        
        # Recover original scale
        points = points * self.scale + self.center
        sdf = sdf * self.scale
        
        return {
            "xyz": torch.from_numpy(points).to(dtype=torch.float32, device="cuda"),
            "sdf": torch.from_numpy(sdf).to(dtype=torch.float32, device="cuda")
        }
        

