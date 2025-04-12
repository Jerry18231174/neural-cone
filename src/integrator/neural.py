import torch

from src.model.radiosity import NeuralRadiosity

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
        super().__init__(mi.Properties())

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
        
        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)

        with torch.no_grad():
        
            if self.render_mode == "LHS":
                color = self.model.render_lhs(si, scene, precision=self.precision)
            elif self.render_mode == "RHS":
                color = self.model.render_rhs(si, scene, spp=self.spp, precision=self.precision)
            else:
                raise ValueError("Invalid render mode:", self.render_mode)

        result = mi.Color3f(color.to(torch.float32))
        
        dr.sync_device()
        torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        
        return result, si.is_valid(), []

