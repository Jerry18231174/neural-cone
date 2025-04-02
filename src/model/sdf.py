import os
import time

import numpy as np
import torch
import torch.nn as nn

import trimesh
import mesh2sdf
import pysdf
from mesh_to_sdf import mesh_to_voxels

from src.module.basic import ShallowMLP
from src.module.hash_grid import MultiresHashGrid


def torch_bbox(vertices, margin=1e-3):
    bbox = [
        torch.from_numpy(np.min(vertices, axis=0) - margin).to(dtype=torch.float32, device="cuda"),
        torch.from_numpy(np.max(vertices, axis=0) + margin).to(dtype=torch.float32, device="cuda"),
    ]
    return bbox


class NGPSDF(nn.Module):
    """
    Instant-NGP based Signed Distance Field
    """

    def __init__(self, config: dict, mesh_path: str) -> None:
        super(NGPSDF, self).__init__()
        self.config = config
        mesh = trimesh.load(mesh_path)
        self.bbox = torch_bbox(mesh.vertices)

        self.hash_grid = MultiresHashGrid(config, self.bbox)

        self.mlp = ShallowMLP(
            in_channels=config["n_features_per_level"] * config["n_levels"],
            out_channels=1,
            hidden_layers=1,
            hidden_channels=config["n_features_per_level"] * config["n_levels"] // 2,
            activation=nn.ReLU(),
            output_activation=nn.Identity()
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        return self.mlp(self.hash_grid(self._normalize_pos(points)))
    
    def _normalize_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Normalize positions to [0, 1] range.
        """
        return (pos - self.bbox[0]) / (self.bbox[1] - self.bbox[0])


class GridSDF:
    """
    Dense Grid based Signed Distance Field
    """
    index_offset = torch.tensor(
        [[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]],
        dtype=torch.int64,
        device="cuda"
    )

    def __init__(self, config: dict, mesh_path: str) -> None:
        self.config = config

        self.mesh: trimesh.Trimesh = trimesh.load(mesh_path, process=False)
        self.bbox: torch.Tensor = torch_bbox(self.mesh.vertices, margin=1e-1)
        # Record bbox center and scale for random access
        self.center = (self.bbox[0] + self.bbox[1]) / 2
        self.scale = (self.bbox[1] - self.bbox[0]).max() / 2
        # Normalize mesh to [-1, 1]
        self.mesh.vertices = (self.mesh.vertices - self.center.cpu().numpy()) / self.scale.cpu().numpy()

        # SDF grid
        self.grid: torch.Tensor = None
    
    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Random access SDF values from grid.
        """
        # Normalize positions the same way as mesh
        pos = (pos - self.center) / self.scale
        # Resize to [0, 1]
        pos = (pos + 1) / 2

        resolution = self.config["resolution"]

        # Calculate base and offset
        # [N, 3]
        base = torch.floor(pos * resolution).to(torch.int64)
        # [N, 3]
        offset = pos * resolution - base
        comp_offset = 1 - offset
        # [N, 8, 3]
        index = base.unsqueeze(1) + GridSDF.index_offset
        # [8N, 3]
        index = index.reshape(-1, 3)

        # Calculate tri-lerp weight
        w0 = comp_offset[:, 0] * comp_offset[:, 1] * comp_offset[:, 2]
        w1 = offset[:, 0]      * comp_offset[:, 1] * comp_offset[:, 2]
        w2 = comp_offset[:, 0] * offset[:, 1]      * comp_offset[:, 2]
        w3 = offset[:, 0]      * offset[:, 1]      * comp_offset[:, 2]
        w4 = comp_offset[:, 0] * comp_offset[:, 1] * offset[:, 2]
        w5 = offset[:, 0]      * comp_offset[:, 1] * offset[:, 2]
        w6 = comp_offset[:, 0] * offset[:, 1]      * offset[:, 2]
        w7 = offset[:, 0]      * offset[:, 1]      * offset[:, 2]
        # [N, 8]
        weight = torch.stack([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)

        # Fetch corner sdf values
        assert (index >= 0).all() and (index <= resolution).all(), "Index out of range in SDF access!"
        # [8N]
        active = (index >= 0).all(dim=1) & (index < resolution).all(dim=1)
        sdf = torch.ones((index.shape[0],), dtype=torch.float32, device="cuda") * 1e10

        sdf[active] = self.grid[index[active, 0], index[active, 1], index[active, 2]]
        # [N, 8, 1]
        sdf = sdf.reshape(-1, 8, 1)

        # Interpolate
        # [N, 1]
        sdf = (sdf * weight.unsqueeze(-1)).sum(dim=1)
        return sdf
    
    def compute(self, cache_path: str) -> None:
        """
        Compute SDF from mesh.
        """
        # Grid size of SDF, resolution is the number of voxels per axis
        size = self.config["resolution"]

        if os.path.exists(cache_path):
            # with open(cache_path, "rb") as f:
            #     sdf = pickle.load(f)
            sdf = np.load(cache_path)
            print("SDF cache loaded from", cache_path)
            
            self.grid = torch.zeros(size + 1, size + 1, size + 1, dtype=torch.float32, device="cuda")
            self.grid[:-1, :-1, :-1] = torch.from_numpy(sdf).to(dtype=torch.float32, device="cuda")
            assert self.grid.shape == (size + 1, size + 1, size + 1), \
                f"SDF cache shape mismatch, expect {size + 1}^3 but got {self.grid.shape}"
            return

        # Grid SDF computation
        start_time = time.time()
        print("Generating SDF ...")

        if self.config["method"] == "mesh2sdf":
            # A grid only method (grid: [-1, 1)^3)
            sdf = mesh2sdf.compute(vertices=self.mesh.vertices, faces=self.mesh.faces, size=size)
        
        elif self.config["method"] == "mesh_to_sdf":
            # A grid only method
            sdf = mesh_to_voxels(self.mesh, size)

        elif self.config["method"] == "pysdf":
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            z = np.linspace(-1, 1, size)
            xx, yy, zz = np.meshgrid(x, y, z)
            xyz = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
            sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
            sdf = sdf_fn(xyz)
        else:
            raise ValueError("Unsupported SDF computing method:", self.config["method"])

        end_time = time.time()
        print(f"SDF generation complete: {end_time - start_time:.2f} seconds.")

        # Retrieve original scale
        sdf = sdf * self.scale.cpu().numpy()
        
        # Save cache
        # with open(cache_path, "wb") as f:
        #     pickle.dump(sdf, f)
        np.save(cache_path, sdf)
        print("SDF cache dumped to", cache_path)

        self.grid = torch.zeros(size + 1, size + 1, size + 1, dtype=torch.float32, device="cuda")
        self.grid[:-1, :-1, :-1] = torch.from_numpy(sdf).to(dtype=torch.float32, device="cuda")