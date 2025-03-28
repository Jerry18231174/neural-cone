import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.basic import ShallowMLP
from src.module.hash_grid import MultiresHashGrid
from src.sample.lhs_rhs import LHSRHS, extract_input, get_wr_itsc
from src.model.sdf import GridSDF

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
    

class NeuralConeRadiosity(NeuralRadiosity):
    """
    Neural Cone Radiosity model
    """

    def __init__(self, config: dict, bbox: torch.Tensor, sdf_model: GridSDF) -> None:
        super(NeuralConeRadiosity, self).__init__(config["ray"], bbox)
        self.config = config["cone"]
        self.k = config["cone_threshold"]
        self.march_min = config["min_march_distance"]
        self.march_steps = config["max_march_steps"]

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
        
        self.sdf_model = sdf_model

    def query_model(
        self,
        si: mi.SurfaceInteraction3f,
        scene: mi.Scene,
    ) -> torch.Tensor:
        """
        Query the model with surface interaction
        """

        nr_color = super().query_model(si)

        pos, normal, dir, albedo, roughness, active_side = extract_input(si)

        # Mask & indices for glossy materials
        glossy_mask = ((roughness < 0.5) & (roughness > 0.01)).squeeze()
        glossy_ind = torch.nonzero(glossy_mask).squeeze()
        glossy_ind = mi.Int(glossy_ind.to(dtype=torch.int32))

        # Gather glossy interactions
        si_glo = dr.gather(mi.SurfaceInteraction3f, si, glossy_ind)
        si_wr = get_wr_itsc(si_glo, scene)
        t_max = si_wr.t.torch().unsqueeze(-1)

        # Compute query size
        tan_lobe = tan_ggx_lobe(roughness[glossy_mask], self.k)
        
        # SDF cone marching
        t = torch.ones_like(t_max) * self.march_min
        transmittance = torch.ones_like(t_max)
        cone_color = torch.zeros_like(nr_color[glossy_mask])

        for i in range(self.march_steps - 1):
            # Get radius and sdf
            radius = t * tan_lobe
            pos_march = pos[glossy_mask] + t * dir[glossy_mask]
            active_t = (t < t_max).squeeze() & si_wr.is_valid().torch().bool()
            sdf = 100000 * torch.ones_like(radius)
            sdf[active_t] = self.sdf_model(pos_march[active_t])
            # print(i, "t:", t.mean(), t_max.mean())
            # print(i, "sdf:", sdf[active_t].mean())

            # Prefiltered model inference
            active = active_t & (sdf < radius).squeeze()

            pfilt_enc = self.pfilt_grid.forward_layer_interp(pos_march[active], point_size=radius[active])
            pfilt_enc = torch.cat([pfilt_enc, pos_march[active], -dir[glossy_mask][active], radius[active]], dim=-1)
            march_color = torch.abs(self.cone_mlp(pfilt_enc))

            # Update color
            opacity = ((radius - sdf.abs()) / radius)[active]
            cone_color[active] += transmittance[active] * opacity * march_color
            
            # Update t & transmittance
            transmittance[active] *= 1 - torch.clamp(opacity, 0, 1)
            t[active_t] = t[active_t] + sdf[active_t]
        
        # Query model at glossy interactions
        radius = t_max * tan_lobe
        pos_march = pos[glossy_mask] + t_max * dir[glossy_mask]
        active = si_wr.is_valid().torch().bool()
        
        pfilt_enc = self.pfilt_grid.forward_layer_interp(pos_march[active], point_size=radius[active])
        pfilt_enc = torch.cat([pfilt_enc, pos_march[active], -dir[glossy_mask][active], radius[active]], dim=-1)
        march_color = torch.abs(self.cone_mlp(pfilt_enc))

        # Update color
        cone_color[active] += transmittance[active] * march_color

        # Merge with neural radiosity
        color = nr_color.clone()
        color[glossy_mask] = self.merge_mlp(torch.cat(
            [nr_color[glossy_mask], cone_color, roughness[glossy_mask]], dim=-1))

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
    