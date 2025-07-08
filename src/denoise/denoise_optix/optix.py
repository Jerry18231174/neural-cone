import setup_optix_example as optix
import mitsuba as mi
import imgui
import torch
from integrators.mi_albedo_normal import BAlbedoNormalIntegrator
from typing import Tuple


class OptixDenoiser:
    def name(self):
        return "[optix]"

    def __init__(self, scene: mi.Scene):
        self.use_builtin_aov = False
        self.aux_integrator = None

        self.scene: mi.Scene = scene
        self.sensor: mi.Sensor = scene.sensors()[0]

        self.module = optix
        self.aux = True
        self.temporal = False

    def denoise(self, noisy: mi.TensorXf):
        noisy = noisy.torch()
        if ((not self.aux) or self.use_builtin_aov):
            img = self.module.denoise(noisy, self.aux, self.temporal)
        else:
            if (self.aux_integrator is None):
                self.aux_integrator = mi.load_dict({'type': 'b_albedo_normal'})
            aux_tensor: torch.Tensor = mi.render(
                scene=self.scene, spp=1, seed=0, sensor=self.sensor, integrator=self.aux_integrator
            ).torch()
            img = self.module.denoise_separate(noisy, aux_tensor, self.aux, self.temporal)
        return img

    def free(self):
        self.module.free_denoiser()
        torch.cuda.empty_cache()

    def render_ui(self, integrator: mi.Integrator) -> Tuple[bool, mi.Integrator]:
        value_changed = False
        vc, self.aux = imgui.checkbox(
            "{} Use Albedo and Normal".format(self.name()), self.aux)
        value_changed = value_changed or vc

        vc, self.temporal = imgui.checkbox(
            "{} Use Temporal".format(self.name()), self.temporal)
        value_changed = value_changed or vc

        if (self.aux):
            vc, self.use_builtin_aov = imgui.checkbox(
                "{} Use Built-in Interator(slow)".format(self.name()), self.use_builtin_aov)
            value_changed = value_changed or vc

            if (self.use_builtin_aov):
                # we have test in oidn.py, even we record the integrator and do not reconstruct it,
                # the cost time is still the same, the biggest cost is the builtin aov integrator
                # (it is toooooooo slow!!!)
                integrator = mi.load_dict({
                    'type': 'aov',
                    'aovs': "albedo:albedo,sh_normal:sh_normal",
                    'integrator': integrator
                })

        return value_changed, integrator
