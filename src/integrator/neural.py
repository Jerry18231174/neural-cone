import torch
import time

from src.model.radiosity import NeuralRadiosity
from src.sample.lhs_rhs import first_smooth, first_smooth_dnr, extract_input
from src.module.ncr_filter import NCRFilter

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


class RadiosityIntegrator(mi.SamplingIntegrator):
    def __init__(
        self,
        model: NeuralRadiosity,
        render_mode: str = "LHS",
        width: int = 800,
        height: int = 600,
        spp: int = 1,
        use_filter: bool = True,
        precision=torch.float32
    ) -> None:
        props = mi.Properties()
        # Set ray direction to pixel center (Implemented in Mitsuba3)
        props["pixel_center"] = True

        super().__init__(props)

        self.model = model
        self.render_mode = render_mode
        self.spp = spp
        self.max_spp = 16
        self.precision = precision
        self.width = width
        self.height = height
        self.use_filter = use_filter
        self.ncr_filter = NCRFilter(
            width, height, kernel_size = 9,
            fxaa=True, bilateral=True,
            bilateral_sigma_spatial=4, bilateral_sigma_range=0.005, bilateral_sigma_color=2.0,
            roughness_threshold=0.5, use_kernel=True
        )

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, bool, list[float]]:
        
        # si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
        si, throughput, emission, _ = first_smooth(scene, sampler, ray, active)

        with torch.no_grad():
        
            if self.render_mode == "LHS":
                dr.eval(si)
                pos, normal, direction, albedo, roughness, active_side = extract_input(si)
                gbuf = (pos, normal, direction, albedo, roughness, active_side)
                color = self.model.render_lhs(si, gbuf, precision=self.precision)
            elif self.render_mode == "RHS":
                # Unfold spp into multiple iterations to avoid OOM
                point_num = dr.width(si.p)
                color = torch.zeros((point_num, 3), device="cuda")
                render_iter = (self.spp + self.max_spp - 1) // self.max_spp
                for i in range(render_iter):
                    iter_spp = min(self.max_spp, self.spp - i * self.max_spp)
                    iter_color = self.model.render_rhs(si, spp=iter_spp, precision=self.precision)
                    color += iter_color * (iter_spp / self.spp)
            elif self.render_mode == "Deferred":
                color = self.model.render_deferred(si, spp=self.spp, precision=self.precision)
            elif self.render_mode == "visualize":
                color = self.model.visualize(si, scene, radius_selection=self.spp, precision=self.precision)
            else:
                raise ValueError("Invalid render mode:", self.render_mode)
        
        if self.use_filter:
            color = color.reshape(self.height, self.width, 3).contiguous()
            pos = pos.reshape(self.height, self.width, 3).contiguous()
            normal = normal.reshape(self.height, self.width, 3).contiguous()
            albedo = albedo.reshape(self.height, self.width, 3).contiguous()
            roughness = roughness.reshape(self.height, self.width, 1).contiguous()

            color = self.ncr_filter.apply(
                color,
                position=pos,
                normal=normal,
                albedo=albedo,
                roughness=roughness,
            ).reshape(-1, 3)

        result = mi.Color3f(color.to(torch.float32)) * throughput + emission
        
        dr.sync_device()
        torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        
        return result, si.is_valid(), []
