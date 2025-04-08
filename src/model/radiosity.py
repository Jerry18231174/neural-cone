import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.basic import ShallowMLP
from src.module.hash_grid import MultiresHashGrid
from src.sample.lhs_rhs import LHSRHS, extract_input, get_mc_itsc
from src.model.kmeans import KMeans

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def tan_ggx_lobe(alpha: torch.Tensor, k: float = 0.5) -> torch.Tensor:
    """
    Tangent of GGX lobe with roughness alpha and threshold k
    """
    sqrt_k = np.sqrt(k)
    sq_a = alpha ** 2

    assert (alpha >= 0).all(), "Roughness must be positive"
    assert (sq_a < sqrt_k).all(), "Roughness must be less than threshold"

    # Numerator
    numer = torch.sqrt((1 - sqrt_k) * sq_a * (sqrt_k - sq_a))
    # Denominator
    denom = sqrt_k + (sqrt_k - 2) * sq_a

    return numer / denom


class NeuralRadiosity(nn.Module):
    """
    Neural Radiosity model
    """

    def __init__(self, config: dict, bbox: torch.Tensor) -> None:
        super(NeuralRadiosity, self).__init__()
        self.config = config

        self.hash_grid = MultiresHashGrid(config, bbox, twosided=False)

        self.mlp = ShallowMLP(
            # encoding + pos + normal + wr + albedo + roughness
            in_channels=config["n_features_per_level"] * config["n_levels"] + 3 * 4 + 1,
            out_channels=3,
            hidden_layers=config["n_hidden_layers"],
            hidden_channels=config["n_hidden_dims"],
            activation=nn.ReLU(),
            output_activation=nn.Identity()
        )

    def query_model(self, si: mi.SurfaceInteraction3f) -> torch.Tensor:
        """
        Query the model with surface interaction
        """

        pos, normal, dir, albedo, roughness, active_side = extract_input(si)

        # Hash grid encoding
        enc = self.hash_grid(pos)

        # Concatenate encoding with wr_direction and roughness
        enc = torch.cat([enc, pos, dir, normal, albedo, roughness], dim=-1)

        # Pass through MLP
        color = torch.abs(self.mlp(enc))

        return color
    
    def forward(self, lhs_rhs: LHSRHS) -> torch.Tensor:
        """
        Query the model with lhs and rhs interactions
        """
        si_lhs = lhs_rhs.si_lhs
        si_rhs = lhs_rhs.si_bsdf

        lhs_color = self.query_model(si_lhs)
        rhs_color = self.query_model(si_rhs)

        # Render rhs color
        rhs_color = rhs_color.reshape(-1, lhs_rhs.dirs_per_point, 3)
        rhs_color = lhs_rhs.render(rhs_color, None)

        return {
            "lhs": lhs_color,
            "rhs": rhs_color
        }
    
    def render_lhs(self, si_lhs: mi.SurfaceInteraction3f, scene: mi.Scene):
        with torch.no_grad():
            lhs_color = self.query_model(si_lhs)
        
        return lhs_color
    
    def render_rhs(self, si_lhs: mi.SurfaceInteraction3f, scene: mi.Scene, spp: int = 1):
        with torch.no_grad():
            # Sample rhs interactions
            point_num = si_lhs.p.torch().shape[0]
            lhs_rhs = LHSRHS(
                scene=scene,
                point_num=point_num,
                dirs_per_point=spp
            )
            lhs_rhs.sample(seed=np.random.randint(0, 1000000), si_lhs=si_lhs)
            si_rhs = lhs_rhs.si_bsdf

            rhs_color = self.query_model(si_rhs)

            # Render rhs color
            rhs_color = rhs_color.reshape(-1, spp, 3)
            rhs_color = lhs_rhs.render(rhs_color, None)

        return rhs_color


def get_ncr_bbox(scene: mi.Scene) -> torch.Tensor:
    """
    Get the bounding box of the scene for Neural Cone Radiosity
    """
    bbox = scene.bbox()
    bbox = torch.tensor([bbox.min - 1e-1, bbox.max + 1e-1], dtype=torch.float32, device="cuda")
    return bbox

class NeuralConeRadiosity(NeuralRadiosity):
    """
    Neural Cone Radiosity model
    """

    def __init__(self, config: dict, bbox: torch.Tensor) -> None:
        super(NeuralConeRadiosity, self).__init__(config["ray"], bbox)
        self.config = config["cone"]
        self.k = config["cone_threshold"]
        self.n_glossy_rhs = config["n_glossy_rhs"]
        self.n_glossy_samples = config["n_glossy_max_samples"]
        
        self.kMeans = KMeans(
            n_clusters=self.n_glossy_samples,
            n_iter=10
        )

        self.pfilt_grid = MultiresHashGrid(self.config, bbox, twosided=False)
        
        self.cone_mlp = ShallowMLP(
            # encoding + pos + normal + wr + albedo + roughness
            in_channels=self.config["n_features_per_level"] + 3 * 2 + 1,
            out_channels=3,
            hidden_layers=self.config["n_hidden_layers"],
            hidden_channels=self.config["n_hidden_dims"],
            activation=nn.ReLU(),
            output_activation=nn.Identity()
        )

        self.merge_mlp = ShallowMLP(
            in_channels=6+1,
            out_channels=3,
            hidden_layers=1,
            hidden_channels=32,
            activation=nn.ReLU(),
            output_activation=nn.Softplus()
        )

    def query_model(
        self,
        si: mi.SurfaceInteraction3f,
        scene: mi.Scene,
        seed: int = np.random.randint(0, 1000000)
    ) -> torch.Tensor:
        """
        Query the model with surface interaction
        """

        t0 = time.time()
        nr_color = super().query_model(si)

        t1 = time.time()
        pos, normal, dir, albedo, roughness, active_side = extract_input(si)

        # Mask & indices for glossy materials
        glossy_mask = ((roughness < 0.5) & (roughness > 0.01)).squeeze()

        # Get RHS interaction distance from Monte Carlo sampling
        t2 = time.time()
        dr.sync_device()
        si_glo_rhs, _, _ = get_mc_itsc(si, scene, glossy_mask, self.n_glossy_rhs, seed=seed)
        dr.sync_device()
        t_mc = si_glo_rhs.t.torch().reshape(-1, self.n_glossy_rhs)
        dr.sync_device()
        t3 = time.time()
        # t_far = ~si_glo_rhs.is_valid().torch().bool().reshape(-1, self.n_glossy_rhs)

        # Aggregate MC points into fixed number of gaussians
        t_fix, n_fix, var_fix = self.kMeans.fit(t_mc)
        t4 = time.time()

        # Compute query size
        tan_lobe = tan_ggx_lobe(roughness[glossy_mask], self.k)
        
        # Glossy model inference
        cone_color = torch.zeros_like(nr_color[glossy_mask])

        for i in range(self.n_glossy_samples):
            active = n_fix[:, i] >= 1
            pos_march = (pos[glossy_mask] + t_fix[:, i:i+1] * dir[glossy_mask])[active]
            radius = (t_fix[:, i:i+1] * tan_lobe + var_fix[:, i:i+1])[active] / 2
        
            pfilt_enc = self.pfilt_grid.forward_layer_interp(pos_march, point_size=radius)
            pfilt_enc = torch.cat([pfilt_enc, pos_march, -dir[glossy_mask][active], radius], dim=-1)
            march_color = torch.abs(self.cone_mlp(pfilt_enc))

            # Update color
            cone_color[active] += march_color * (n_fix[:, i:i+1] / self.n_glossy_rhs)[active]

        # Merge with neural radiosity
        t5 = time.time()
        color = nr_color.clone()
        color[glossy_mask] = self.merge_mlp(torch.cat(
            [nr_color[glossy_mask], cone_color, roughness[glossy_mask]], dim=-1))
        t6 = time.time()
        # print("######################")
        # print("NR time:\t", t1-t0)
        # print("Extract input:\t", t2-t1)
        # print("MC sampling:\t", t3-t2)
        # print("KMeans:\t\t", t4-t3)
        # print("Model:\t\t", t5-t4)
        # print("Merge:\t\t", t6-t5)

        return color
    
    def forward(self, lhs_rhs: LHSRHS) -> torch.Tensor:
        """
        Query the model with lhs and rhs interactions
        """
        si_lhs = lhs_rhs.si_lhs
        si_rhs = lhs_rhs.si_bsdf

        lhs_color = self.query_model(si_lhs, lhs_rhs.scene)
        rhs_color = self.query_model(si_rhs, lhs_rhs.scene)

        # Render rhs color
        rhs_color = rhs_color.reshape(-1, lhs_rhs.dirs_per_point, 3)
        rhs_color = lhs_rhs.render(rhs_color, None)

        return {
            "lhs": lhs_color,
            "rhs": rhs_color
        }
    
    def render_lhs(self, si_lhs: mi.SurfaceInteraction3f, scene: mi.Scene):
        with torch.no_grad():
            lhs_color = self.query_model(si_lhs, scene)
        
        return lhs_color
    
    def render_rhs(self, si_lhs: mi.SurfaceInteraction3f, scene: mi.Scene, spp: int = 1):
        with torch.no_grad():
            # Sample rhs interactions
            point_num = si_lhs.p.torch().shape[0]
            lhs_rhs = LHSRHS(
                scene=scene,
                point_num=point_num,
                dirs_per_point=spp
            )
            lhs_rhs.sample(seed=np.random.randint(0, 1000000), si_lhs=si_lhs)
            si_rhs = lhs_rhs.si_bsdf

            rhs_color = self.query_model(si_rhs, scene)

            # Render rhs color
            rhs_color = rhs_color.reshape(-1, spp, 3)
            rhs_color = lhs_rhs.render(rhs_color, None)

        return rhs_color
    