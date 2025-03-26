import torch
import torch.nn as nn
import mitsuba as mi

from typing import List

mi.set_variant("cuda_rgb")


class TriMipPlanes(nn.Module):
    """
    Triple Mip-Maped Feature Planes
    """

    def __init__(self, config: dict, bbox: mi.BoundingBox3f) -> None:
        super(TriMipPlanes, self).__init__()
        self.config = config
        self.bbox = (
            bbox.min.torch().to(device="cuda") - 1e-3,
            bbox.max.torch().to(device="cuda") + 1e-3,
        )

        self.D = config["n_features_per_level"]

        self.base = nn.Parameter(torch.zeros(
            (3, config["base_resolution"], config["base_resolution"], config["n_features"]),
            dtype=torch.float32,
            device="cuda"
        ))

        self.mipmap: List[torch.Tensor] = []
        self.prefilter()
    
    def prefilter(self):
        """
        Prefilter feature planes.
        Should be called after parameter update.
        """
        pass