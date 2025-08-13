import torch

from src.model.radiosity import NeuralRadiosity
from src.sample.lhs_rhs import first_smooth

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


class RadiosityIntegrator(mi.SamplingIntegrator):
    def __init__(
        self,
        model: NeuralRadiosity,
        render_mode: str = "LHS",
        spp: int = 1,
        precision=torch.float32
    ) -> None:
        props = mi.Properties()
        # Set ray direction to pixel center (Implemented in Mitsuba3)
        props["pixel_center"] = True

        super().__init__(props)

        self.model = model
        self.render_mode = render_mode
        self.spp = spp
        self.precision = precision

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
                color = self.model.render_lhs(si, scene, precision=self.precision)
            elif self.render_mode == "RHS":
                color = self.model.render_rhs(si, scene, spp=self.spp, precision=self.precision)
            elif self.render_mode == "visualize":
                color = self.model.visualize(si, scene, radius_selection=self.spp, precision=self.precision)
            else:
                raise ValueError("Invalid render mode:", self.render_mode)

        result = mi.Color3f(color.to(torch.float32)) * throughput + emission
        
        dr.sync_device()
        torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        
        return result, si.is_valid(), []

