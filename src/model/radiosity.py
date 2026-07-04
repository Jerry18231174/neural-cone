import os
import time
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import tinycudann as tcnn

from src.module.basic import ShallowMLP
from src.module.hash_grid import MultiresHashGrid
from src.sample.lhs_rhs import LHSRHS, extract_input, get_mc_itsc
from src.model.kmeans import KMeans
from src.util.tan_lobe import LobeLUT

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def get_model_bbox(scene: mi.Scene) -> torch.Tensor:
    """
    Get the bounding box of the scene for Neural Cone Radiosity
    """
    bbox = scene.bbox()
    bbox = torch.tensor([bbox.min - 1e-1, bbox.max + 1e-1], dtype=torch.float32)
    return bbox


def get_time():
    dr.sync_device()
    torch.cuda.synchronize()
    return time.time()


class RadiosityPipeline(L.LightningModule):
    """
    Radiosity pipeline
    """
    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super(RadiosityPipeline, self).__init__()
        self.pipeline_config = pipeline_config
        self.scene = scene
        self.register_buffer("bbox", get_model_bbox(scene))

    def training_step(self, batch, batch_idx):
        """
        Training step for the model
        """
        seed = self.global_step * self.trainer.world_size + self.global_rank

        # Adaptive RHS and epsilon
        ad_ratio = 2 ** int(4 * (self.global_step / self.trainer.max_steps))
        point_num = self.pipeline_config["sample"]["n_points"] // ad_ratio
        dirs_per_point = self.pipeline_config["sample"]["n_dirs_per_point"] * ad_ratio
        eps = 1e-1 / ad_ratio

        # Sample
        lhs_rhs = LHSRHS(
            scene=self.scene,
            point_num=point_num,
            dirs_per_point=dirs_per_point,
        )
        if self.pipeline_config["sample"]["from_poses"]:
            lhs_rhs.sample(seed=seed, pose=batch)
        else:
            lhs_rhs.sample(seed=seed)

        # Forward pass
        result = self(lhs_rhs)
        lhs_color = result["lhs"]
        rhs_color = result["rhs"].detach()

        # Compute loss
        nr_norm = (rhs_color + lhs_color).detach() / 2 + eps
        loss = torch.mean(((rhs_color - lhs_color) / nr_norm) ** 2) / ad_ratio

        # Tone loss
        lhs_tone_norm = torch.norm(lhs_color, dim=-1, keepdim=True)
        rhs_tone_norm = torch.norm(rhs_color, dim=-1, keepdim=True)
        lhs_tone = lhs_color / (lhs_tone_norm + eps)
        rhs_tone = rhs_color / (rhs_tone_norm + eps)
        tone_loss = torch.mean((lhs_tone - rhs_tone) ** 2) / ad_ratio

        loss += tone_loss * self.pipeline_config["train"]["tone_loss_weight"]

        # Logging
        self.log("loss", loss.item(), prog_bar=True)
        self.log("real_step", self.global_step, prog_bar=True)

        if (self.global_step + 1) % 200 == 0 and self.global_rank == 0:
            print(f"##### Step {self.global_step + 1} #####")
            print("lhs color", lhs_color[:3])
            print("rhs color", rhs_color[:3])

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.pipeline_config["train"]["learning_rate"])
    
    def on_before_optimizer_step(self, optimizer):
        clip_val = float(self.pipeline_config["train"].get("gradient_clip_val", 1.0))
        has_nonfinite_grad = False

        for param in self.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                has_nonfinite_grad = True
                param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        if clip_val > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_val)
            if isinstance(grad_norm, torch.Tensor):
                self.log("grad_norm", grad_norm.item(), prog_bar=False)

        if has_nonfinite_grad:
            self.log("nonfinite_grad", 1.0, prog_bar=True)
    
    @abstractmethod
    def query_model(self, si: mi.SurfaceInteraction3f, gbuf: tuple, precision=torch.float32) -> torch.Tensor:
        """
        Query the model with surface interaction
        """
        pass
    
    def forward(self, lhs_rhs: LHSRHS) -> torch.Tensor:
        """
        Query the model with lhs and rhs interactions
        """
        si_lhs = lhs_rhs.si_lhs
        si_rhs = lhs_rhs.si_bsdf

        dr.eval(si_lhs)
        dr.eval(si_rhs)

        gbuf_lhs = extract_input(si_lhs)
        gbuf_rhs = extract_input(si_rhs)

        lhs_color = self.query_model(si_lhs, gbuf_lhs)
        with torch.no_grad():
            rhs_color = self.query_model(si_rhs, gbuf_rhs)

        # Render rhs color
        rhs_color = rhs_color.reshape(-1, lhs_rhs.dirs_per_point, 3)
        rhs_color = lhs_rhs.render(rhs_color, None)

        return {
            "lhs": lhs_color,
            "rhs": rhs_color
        }
    
    def render_lhs(self, si_lhs: mi.SurfaceInteraction3f, gbuf: tuple, precision=torch.float32):
        with torch.no_grad():
            lhs_color = self.query_model(si_lhs, gbuf, precision=precision)
        
        return lhs_color
    
    def render_rhs(self, si_lhs: mi.SurfaceInteraction3f, spp: int = 1, precision=torch.float32):
        with torch.no_grad():
            dr.eval(si_lhs)
            point_num = dr.width(si_lhs.p)

            # Sample rhs interactions
            lhs_rhs = LHSRHS(
                scene=self.scene,
                point_num=point_num,
                dirs_per_point=spp
            )
            lhs_rhs.sample(seed=np.random.randint(0, 1000000), si_lhs=si_lhs)
            si_rhs = lhs_rhs.si_bsdf
            dr.eval(si_rhs)
            pos, normal, direction, albedo, roughness, active_side = extract_input(si_rhs)
            gbuf = (pos, normal, direction, albedo, roughness, active_side)

            rhs_color = self.query_model(si_rhs, gbuf, precision=precision)

            # Render rhs color
            rhs_color = rhs_color.reshape(-1, spp, 3)
            rhs_color = lhs_rhs.render(rhs_color, None)

        return rhs_color
    
    def render_deferred(self, si_lhs: mi.SurfaceInteraction3f, spp: int = 1, precision=torch.float32):
        with torch.no_grad():
            t0 = get_time()
            dr.eval(si_lhs)
            point_num = dr.width(si_lhs.p)

            glossy_mask = si_lhs.bsdf().eval_roughness(si_lhs) < 0.5
            diff_idx = dr.compress(~glossy_mask)
            glo_idx = dr.compress(glossy_mask)
            si_diff = dr.gather(mi.SurfaceInteraction3f, si_lhs, diff_idx)
            si_glo = dr.gather(mi.SurfaceInteraction3f, si_lhs, glo_idx)
            dr.eval(si_diff)
            dr.eval(si_glo)

            # Sample rhs interactions
            lhs_rhs = LHSRHS(
                scene=self.scene,
                point_num=dr.width(glo_idx),
                dirs_per_point=spp
            )
            t1 = get_time()
            lhs_rhs.sample(seed=np.random.randint(0, 1000000), si_lhs=si_glo)
            t2 = get_time()
            si_glo_rhs = lhs_rhs.si_bsdf
            dr.eval(si_glo_rhs)
            pos, normal, direction, albedo, roughness, active_side = extract_input(si_glo_rhs)
            gbuf = (pos, normal, direction, albedo, roughness, active_side)
            t3 = get_time()

            rhs_color = self.query_model(si_glo_rhs, gbuf, precision=precision)
            t4 = get_time()

            # Render glossy rhs color
            rhs_color = rhs_color.reshape(-1, spp, 3)
            glo_color = lhs_rhs.render(rhs_color, None)
            t5 = get_time()

            pos, normal, direction, albedo, roughness, active_side = extract_input(si_diff)
            gbuf = (pos, normal, direction, albedo, roughness, active_side)
            diff_color = self.query_model(si_diff, gbuf, precision=precision)
            t6 = get_time()

            glossy_mask = glossy_mask.torch().bool()
            color = torch.zeros((point_num, 3), device="cuda", dtype=precision)
            color[glossy_mask] = glo_color
            color[~glossy_mask] = diff_color
            t7 = get_time()

            # print("######################")
            # print("Preproc:\t", t1-t0)
            # print("Sample:\t\t", t2-t1)
            # print("Extract:\t", t3-t2)
            # print("Glossy model:\t", t4-t3)
            # print("Render:\t\t", t5-t4)
            # print("Diffuse model:\t", t6-t5)
            # print("Merge:\t\t", t7-t6)
            # print("Total:\t\t", t7-t0)

        return color
    

class NeuralRadiosity(RadiosityPipeline):
    """
    Neural Radiosity model
    """

    def __init__(self, config: dict, pipeline_config: dict, scene: mi.Scene) -> None:
        super(NeuralRadiosity, self).__init__(pipeline_config, scene)

        self.hash_grid = MultiresHashGrid(config, self.bbox, twosided=False)
        self.hash_grid.load_kernel("NR_hash_grid_cuda")

        if config["use_tcnn"]:
            network_config = {
                "otype": "FullyFusedMLP" if config["n_hidden_dims"] <= 128 else "CutlassMLP",
                "n_hidden_layers": config["n_hidden_layers"],
                "n_neurons": config["n_hidden_dims"],
                "activation": "ReLU",
                "output_activation": config["output_activation"],
            }
            self.mlp = tcnn.Network(
                n_input_dims=config["n_features_per_level"] * config["n_levels"] + 3 * 4 + 1,
                n_output_dims=3,
                network_config=network_config
            )
        else:
            self.mlp = ShallowMLP(
                # encoding + pos + normal + wr + albedo + roughness
                in_channels=config["n_features_per_level"] * config["n_levels"] + 3 * 4 + 1,
                out_channels=3,
                hidden_layers=config["n_hidden_layers"],
                hidden_channels=config["n_hidden_dims"],
                activation=nn.ReLU(),
                output_activation=nn.Identity() if config["output_activation"] == "None" else nn.Softplus()
            )

    def query_model(self, si: mi.SurfaceInteraction3f, gbuf: tuple, precision=torch.float32) -> torch.Tensor:
        """
        Query the model with surface interaction
        """
        t0 = get_time()

        pos, normal, dir, albedo, roughness, active_side = gbuf

        t1 = get_time()

        # Hash grid encoding
        enc = self.hash_grid(pos)

        t2 = get_time()

        # Concatenate encoding with wr_direction and roughness
        enc = torch.cat([enc, pos, dir, normal, albedo, roughness], dim=-1)

        # Pass through MLP
        color = torch.abs(self.mlp(enc)).to(dtype=precision)

        t3 = get_time()

        # print("\tNR_Preproc:\t", t1-t0)
        # print("\tNR_HashGrid:\t", t2-t1)
        # print("\tNR_MLP:\t\t", t3-t2)


        return color


class NeuralConeRadiosity(NeuralRadiosity):
    """
    Neural Cone Radiosity model
    """

    def __init__(self, config: dict, pipeline_config: dict, scene: mi.Scene) -> None:
        super(NeuralConeRadiosity, self).__init__(config["ray"], pipeline_config, scene)
        self.config = config["cone"]
        self.n_glossy_rhs = config["n_glossy_rhs"]
        self.n_glossy_samples = config["n_glossy_max_samples"]

        self.tan_lobe_lut = LobeLUT(
            alpha=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            cone_threshold=config["cone_threshold"],
            integrand_type="GGX"
        )
        
        self.kMeans = KMeans(
            n_clusters=self.n_glossy_samples,
            n_iter=config["n_kmeans_iter"]
        )

        self.pfilt_grid = MultiresHashGrid(self.config, self.bbox, twosided=False)

        self.kMeans.load_kernel()
        self.pfilt_grid.load_kernel("NCR_hash_grid_cuda")
        
        if self.config["use_tcnn"]:
            network_config = {
                "otype": "FullyFusedMLP" if self.config["n_hidden_dims"] <= 128 else "CutlassMLP",
                "n_hidden_layers": self.config["n_hidden_layers"],
                "n_neurons": self.config["n_hidden_dims"],
                "activation": "ReLU",
                "output_activation": self.config["output_activation"],
            }
            self.cone_mlp = tcnn.Network(
                n_input_dims=self.config["n_features_per_level"] + 3 * 2 + 1,
                n_output_dims=3,
                network_config=network_config
            )
        else:
            self.cone_mlp = ShallowMLP(
                # encoding + pos + normal + wr + albedo + roughness
                in_channels=self.config["n_features_per_level"] + 3 * 2 + 1,
                out_channels=3,
                hidden_layers=self.config["n_hidden_layers"],
                hidden_channels=self.config["n_hidden_dims"],
                activation=nn.ReLU(),
                output_activation=nn.Identity() if self.config["output_activation"] == "None" else nn.Softplus()
            )

        self.merge_mlp = ShallowMLP(
            in_channels=6+1+3,
            out_channels=3,
            hidden_layers=1,
            hidden_channels=32,
            activation=nn.ReLU(),
            output_activation=nn.Softplus()
        )

    def query_model(
        self,
        si: mi.SurfaceInteraction3f,
        gbuf: tuple,
        precision=torch.float32,
    ) -> torch.Tensor:
        """
        Query the model with surface interaction
        """
        seed = np.random.randint(0, 10000000)
        dr.eval(si)
        t0 = get_time()

        # Get color from neural radiosity
        color = super().query_model(si, gbuf, precision=precision)

        t1 = get_time()
        pos, normal, dir, albedo, roughness, active_side = gbuf

        # Mask & indices for glossy materials
        glossy_mask = (roughness < 0.5).squeeze()

        if not glossy_mask.any():
            return color

        # Get RHS interaction distance from Monte Carlo sampling
        t2 = get_time()
        si_glo_rhs, _, _ = get_mc_itsc(si, self.scene, glossy_mask, self.n_glossy_rhs, seed=seed)
        t21 = get_time()
        t_mc = si_glo_rhs.t.torch().to(device=self.device, dtype=precision).reshape(-1, self.n_glossy_rhs)
        t3 = get_time()

        # Aggregate MC points into fixed number of gaussians
        t_fix, n_fix, std_fix = self.kMeans.fit(t_mc, precision=precision)
        t4 = get_time()

        # Compute query size
        tan_lobe = self.tan_lobe_lut(roughness[glossy_mask])
        
        t5 = get_time()

        # Glossy model inference
        N_glossy = t_fix.shape[0]
        cone_color = torch.zeros(N_glossy, self.n_glossy_samples, 3, device=self.device, dtype=precision)
        # [N, n_clusters]
        active = n_fix >= 1
        radius = (t_fix * tan_lobe + std_fix)[active][:, None] / 2
        # radius = (t_fix * tan_lobe + var_fix).reshape(-1)[:, None] / 2
        # [N, n_clusters, 3: xyz]
        pos_march = (pos[glossy_mask][:, None, :] + t_fix[:, :, None] * dir[glossy_mask][:, None, :])[active]
        # pos_march = (pos[glossy_mask][:, None, :] + t_fix[:, :, None] * dir[glossy_mask][:, None, :]).reshape(-1, 3)
        t51 = get_time()

        pfilt_enc = self.pfilt_grid.forward_layer_interp(pos_march, radius)
        t52 = get_time()
        pfilt_enc = torch.cat([
            pfilt_enc,
            pos_march,
            -dir[glossy_mask][:, None, :].repeat(1, self.n_glossy_samples, 1)[active],
            # -dir[glossy_mask].repeat(self.n_glossy_samples, 1),
            radius
        ], dim=-1)
        march_color = torch.abs(self.cone_mlp(pfilt_enc))

        cone_color[active] = march_color * (n_fix / self.n_glossy_rhs)[active][:, None]
        # cone_color = march_color.reshape(-1, self.n_glossy_samples, 3) * (n_fix / self.n_glossy_rhs)[..., None]
        cone_color = torch.sum(cone_color, dim=1)

        # Merge with neural radiosity
        t6 = get_time()
        color[glossy_mask] = self.merge_mlp(torch.cat(
            [color[glossy_mask], cone_color, roughness[glossy_mask], albedo[glossy_mask]], dim=-1))
        t7 = get_time()
        # print("######################")
        # print("Glossy size:\t", glossy_mask.sum())
        # print("NR time:\t", t1-t0)
        # print("Extract input:\t", t2-t1)
        # print("MC sampling:\t", t21-t2)
        # print("MC torch:\t", t3-t21)
        # print("KMeans:\t\t", t4-t3)
        # print("TanLobe:\t", t5-t4)
        # print("Model:\t\t", t6-t5)
        # print("\tPreproc:\t", t51-t5)
        # print("\tHashGrid:\t", t52-t51)
        # print("\tMLP:\t\t", t6-t52)
        # print("Merge:\t\t", t7-t6)
        # print("Total:\t\t", t7-t0)

        return color
    
    def visualize_glo(self,
        si: mi.SurfaceInteraction3f,
        scene: mi.Scene,
        precision=torch.float32,
        radius_selection: int = 1
    ) -> torch.Tensor:
        """
        Visualize the glossy model
        """

        radii = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

        dr.eval(si)

        pos, normal, dir, albedo, roughness, active_side = extract_input(si, device=self.device, dtype=precision)

        radius = roughness * 0.0 + radii[radius_selection - 1]

        pfilt_enc = self.pfilt_grid.forward_layer_interp(pos, radius)
        
        pfilt_enc = torch.cat([pfilt_enc, pos, -dir, radius], dim=-1)
        color = torch.abs(self.cone_mlp(pfilt_enc))

        return color

    def visualize(self,
        si: mi.SurfaceInteraction3f,
        scene: mi.Scene,
        precision=torch.float32,
        radius_selection: int = 1
    ) -> torch.Tensor:
        """
        Visualize the glossy model
        """
        seed = np.random.randint(0, 10000000)
        dr.eval(si)

        # Get color from neural radiosity
        color = super().query_model(si, scene, precision=precision)

        pos, normal, dir, albedo, roughness, active_side = extract_input(si, device=self.device, dtype=precision)

        # Mask & indices for glossy materials
        glossy_mask = (roughness < 0.5).squeeze()

        if not glossy_mask.any():
            return color

        # Get RHS interaction distance from Monte Carlo sampling
        si_glo_rhs, _, _ = get_mc_itsc(si, scene, glossy_mask, self.n_glossy_rhs, seed=seed)
        t_mc = si_glo_rhs.t.torch().to(device=self.device, dtype=precision).reshape(-1, self.n_glossy_rhs)
        
        t_valid = ~torch.isnan(t_mc) & ~torch.isinf(t_mc)
        t_mc = torch.nan_to_num(t_mc, nan=0.0, posinf=0.0, neginf=0.0)

        t_mean = torch.sum(t_mc * t_valid, dim=-1, keepdim=True) / (torch.sum(t_valid, dim=-1, keepdim=True) + 1e-6)
        t_var = torch.sum((t_mc - t_mean) ** 2 * t_valid, dim=-1, keepdim=True) / (torch.sum(t_valid, dim=-1, keepdim=True) + 1e-6)
        t_std = torch.sqrt(t_var)

        # Aggregate MC points into fixed number of gaussians
        t_fix, n_fix, std_fix = self.kMeans.fit(t_mc, precision=precision)
        # Compute query size
        tan_lobe = self.tan_lobe_lut(roughness[glossy_mask])

        result = torch.zeros_like(color)
        if radius_selection == 1:
            # Coefficient of Variation of all
            result[glossy_mask] = t_std / (t_mean + 1e-6)
        elif radius_selection == 2:
            # Average CV within each cluster
            result[glossy_mask] = torch.sum(std_fix / (t_fix + 1e-6) * n_fix, dim=-1, keepdim=True) / self.n_glossy_rhs
        elif radius_selection == 3:
            # Radius of all
            result[glossy_mask] = (t_mean * tan_lobe + t_std) / 2
        elif radius_selection == 4:
            # Average cluster radius
            cradius = (t_fix * tan_lobe + std_fix) / 2
            result[glossy_mask] = torch.sum(cradius * n_fix, dim=-1, keepdim=True) / self.n_glossy_rhs

        return result
    

class NeuralConeRadiosity2(RadiosityPipeline):
    """
    Neural Cone Radiosity model
    """

    def __init__(self, config: dict, pipeline_config: dict, scene: mi.Scene) -> None:
        super(NeuralConeRadiosity2, self).__init__(pipeline_config, scene)
        self.n_glossy_rhs = config["n_glossy_rhs"]
        self.n_glossy_samples = config["n_glossy_max_samples"]

        self.tan_lobe_lut = LobeLUT(
            alpha=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            cone_threshold=config["cone_threshold"],
            integrand_type="GGX"
        )
        
        self.kMeans = KMeans(
            n_clusters=self.n_glossy_samples,
            n_iter=config["n_kmeans_iter"]
        )

        self.diff_grid = MultiresHashGrid(config["diff"], self.bbox, twosided=False)
        self.spec_grid = MultiresHashGrid(config["spec"], self.bbox, twosided=False)

        self.kMeans.load_kernel()
        self.diff_grid.load_kernel("NCR2_diff_grid_cuda")
        self.spec_grid.load_kernel("NCR2_spec_grid_cuda")
        
        if config["use_tcnn"]:
            diff_mlp_config = {
                "otype": "FullyFusedMLP" if config["diff"]["n_hidden_dims"] <= 128 else "CutlassMLP",
                "n_hidden_layers": config["diff"]["n_hidden_layers"],
                "n_neurons": config["diff"]["n_hidden_dims"],
                "activation": "ReLU",
                "output_activation": config["diff"]["output_activation"],
            }
            self.diff_mlp = tcnn.Network(
                n_input_dims=config["diff"]["n_features_per_level"] * config["diff"]["n_levels"] + 3 * 4 + 1,
                n_output_dims=3,
                network_config=diff_mlp_config
            )
            spec_mlp_config = {
                "otype": "FullyFusedMLP" if config["spec"]["n_hidden_dims"] <= 128 else "CutlassMLP",
                "n_hidden_layers": config["spec"]["n_hidden_layers"],
                "n_neurons": config["spec"]["n_hidden_dims"],
                "activation": "ReLU",
                "output_activation": config["spec"]["output_activation"],
            }
            self.spec_mlp = tcnn.Network(
                n_input_dims=config["diff"]["n_features_per_level"] * config["diff"]["n_levels"] + 3 * 4 + 1 \
                            + (config["spec"]["n_features_per_level"] + 3 * 2 + 1 + 1) * self.n_glossy_samples,
                n_output_dims=3,
                network_config=spec_mlp_config
            )
        else:
            self.diff_mlp = ShallowMLP(
                # encoding + pos + normal + wr + albedo + roughness
                in_channels=config["diff"]["n_features_per_level"] * config["diff"]["n_levels"] + 3 * 4 + 1,
                out_channels=3,
                hidden_layers=config["diff"]["n_hidden_layers"],
                hidden_channels=config["diff"]["n_hidden_dims"],
                activation=nn.ReLU(),
                output_activation=nn.Identity() if config["diff"]["output_activation"] == "None" else nn.Softplus()
            )
            self.spec_mlp = ShallowMLP(
                in_channels=config["diff"]["n_features_per_level"] * config["diff"]["n_levels"] + 3 * 4 + 1 \
                           + (config["spec"]["n_features_per_level"] + 3 * 2 + 1 + 1) * self.n_glossy_samples,
                out_channels=3,
                hidden_layers=config["spec"]["n_hidden_layers"],
                hidden_channels=config["spec"]["n_hidden_dims"],
                activation=nn.ReLU(),
                output_activation=nn.Identity() if config["spec"]["output_activation"] == "None" else nn.Softplus()
            )

    def query_model(
        self,
        si: mi.SurfaceInteraction3f,
        gbuf: tuple,
        precision=torch.float32,
    ) -> torch.Tensor:
        """
        Query the model with surface interaction
        """
        t00 = get_time()
        seed = np.random.randint(0, 10000000)
        dr.eval(si)
        t0 = get_time()

        # Get primary encoding from neural radiosity
        pos, normal, dir, albedo, roughness, active_side = gbuf
        gbuf = torch.cat([pos, dir, normal, albedo, roughness], dim=-1)
        color = torch.zeros_like(pos)

        t1 = get_time()

        # Mask & indices for glossy materials
        glossy_mask = (roughness < 0.5).squeeze()

        # Primary intersection encoding
        prim_enc = self.diff_grid(pos)
        t2 = get_time()

        if (~glossy_mask).any():
            diff_color = self.diff_mlp(torch.cat([prim_enc, gbuf], dim=-1)[~glossy_mask])
            color[~glossy_mask] = diff_color.to(torch.float32)

        if not glossy_mask.any():
            return color

        # Get RHS interaction distance from Monte Carlo sampling
        t3 = get_time()
        si_glo_rhs, _, _ = get_mc_itsc(si, self.scene, glossy_mask, self.n_glossy_rhs, seed=seed)
        t_mc = si_glo_rhs.t.torch().to(device=self.device, dtype=precision).reshape(-1, self.n_glossy_rhs)
        t4 = get_time()

        # Aggregate MC points into fixed number of gaussians
        t_cls, n_cls, std_cls = self.kMeans.fit(t_mc, precision=precision)
        t5 = get_time()

        # Compute query size
        tan_lobe = self.tan_lobe_lut(roughness[glossy_mask])

        # [N, n_clusters, 1]
        scale_cls = (t_cls * tan_lobe + std_cls)[..., None] / 2
        weight_cls = (n_cls / self.n_glossy_rhs)[..., None]

        # [N, n_clusters, 3: xyz]
        pos_cls = (pos[glossy_mask][:, None, :] + t_cls[:, :, None] * dir[glossy_mask][:, None, :])
        dir_cls = -dir[glossy_mask][:, None, :].repeat(1, self.n_glossy_samples, 1)
        t6 = get_time()

        refl_enc = self.spec_grid.forward_layer_interp(
            pos_cls.reshape(-1, 3), scale_cls.reshape(-1, 1)
        ).reshape(pos_cls.size(0), self.n_glossy_samples, -1)
        t7 = get_time()

        refl_enc = torch.cat([
            refl_enc * weight_cls,
            pos_cls,
            dir_cls,
            scale_cls,
            weight_cls
        ], dim=-1).reshape(pos_cls.size(0), -1)

        spec_color = self.spec_mlp(torch.cat([prim_enc[glossy_mask], refl_enc, gbuf[glossy_mask]], dim=-1))
        color[glossy_mask] = spec_color.to(torch.float32)

        t8 = get_time()
        
        # print("######################")
        # print("Glossy size:\t", glossy_mask.sum())
        # print("dr.eval():\t", t0-t00)
        # print("Extract input:\t", t1-t0)
        # print("Diff encoding:\t", t2-t1)
        # print("Diff MLP:\t", t3-t2)
        # print("Ray Trace:\t", t4-t3)
        # print("KMeans:\t\t", t5-t4)
        # print("TanLobe:\t", t6-t5)
        # print("Spec encoding:\t", t7-t6)
        # print("Spec MLP:\t", t8-t7)
        # print("Total: \t\t", t8-t0)

        return color