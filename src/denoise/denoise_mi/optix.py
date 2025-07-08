import mitsuba as mi
import drjit as dr
import torch
import imgui
from typing import Tuple


class OptixDenoiserMI:
    def name(self):
        return "[optix-mi]"

    def __init__(self, scene: mi.Scene):
        self.use_builtin_aov = False
        self.aux_integrator = None

        self.size = (1, 1)
        self.aux = True
        self.temporal = False
        self.denoiser = mi.OptixDenoiser(
            self.size, self.aux, self.aux, self.temporal)
        self.img_pre: torch.Tensor = None
        self.scene: mi.Scene = scene
        self.sensor: mi.Sensor = scene.sensors()[0]

        self.zero_flow = None

    def resize(self, size, aux, temporal):
        changed = False
        if self.size != size:
            self.size = size
            changed = True
        if self.aux != aux:
            self.aux = aux
            changed = True
        if self.temporal != temporal:
            self.temporal = temporal
            if (self.temporal):
                print("\033[0;33m{}\033[0m optical flow is not supported yet, they will be set to all \033[0;31mzeros\033[0m".format(
                    self.name()))
            changed = True
        if changed:
            print("\033[0;33m{}\033[0m Resize, size = {}, albedo & normal = {}, temporal = {}"
                  .format(self.name(), self.size, self.aux, self.temporal))
            self.zero_flow = dr.zeros(mi.TensorXf, [size[0], size[1], 2])
            self.denoiser = mi.OptixDenoiser(
                self.size[::-1], self.aux, self.aux, self.temporal)

    def denoise_simple(self, img):
        self.resize(img.shape[0:2], self.aux, self.temporal)

        ret = None
        if (self.temporal):
            if (self.img_pre is None):
                self.img_pre = img
            ret = self.denoiser(
                img, previous_denoised=self.img_pre, flow=self.zero_flow)
        else:
            ret = self.denoiser(img)
        return ret

    def denoise_albedo_normal(self, img_color: mi.TensorXf, img_albedo: mi.TensorXf, img_normal: mi.TensorXf):
        self.resize(img_color.shape[0:2], self.aux, self.temporal)

        to_sensor: mi.Transform4f = self.sensor.world_transform().inverse()

        ret = None
        if (self.temporal):
            if (self.img_pre is None):
                self.img_pre = img_color
            ret = self.denoiser(img_color, True, img_albedo, img_normal,
                                to_sensor, previous_denoised=self.img_pre, flow=self.zero_flow)
        else:
            ret = self.denoiser(img_color, True, img_albedo, img_normal, to_sensor)
        return ret

    def denoise(self, noisy) -> torch.Tensor:
        ret = None
        if self.aux:
            if (self.use_builtin_aov):
                ret = self.denoise_albedo_normal(noisy[:, :, 0:3], noisy[:, :, 3:6], noisy[:, :, 6:9])
            else:
                if (self.aux_integrator is None):
                    self.aux_integrator = mi.load_dict({'type': 'b_albedo_normal'})
                aux_tensor: mi.TensorXf = mi.render(
                    scene=self.scene, spp=1, seed=0, sensor=self.sensor, integrator=self.aux_integrator
                )
                ret = self.denoise_albedo_normal(noisy[:, :, 0:3], aux_tensor[:, :, 0:3], aux_tensor[:, :, 3:6])
        else:
            ret = self.denoise_simple(noisy)
        ret = ret.torch()
        if self.temporal:
            self.img_pre = ret
        return ret

    def render_ui(self, integrator: mi.Integrator) -> Tuple[bool, mi.Integrator]:
        value_changed = False
        vc, aux = imgui.checkbox(
            "{} Use Albedo and Normal".format(self.name()), self.aux)
        value_changed = value_changed or vc

        temporal = self.temporal
        if aux:
            vc, temporal = imgui.checkbox("{} Use Temporal".format(self.name()), self.temporal)
            if (vc or (not temporal)):
                self.img_pre = None
        else:
            temporal = False

        self.resize(self.size, aux, temporal)

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
