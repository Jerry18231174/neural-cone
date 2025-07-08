import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CURRENT_DIR, "../"))
sys.path.append(CURRENT_DIR)

from utils.ui import UI
import mitsuba as mi

from utils.images import tonemap_aces
import imgui
from optix import OptixDenoiser
import time
from datetime import datetime
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_gpu_ui", action="store_true", help="force to use gpu ui")
    args = parser.parse_args()
    parser.print_help()
    print("\n")

    scene_file = os.path.join(
        CURRENT_DIR, "../../assets/ignore/scenes/veach-ajar/scene.xml")
    if (not os.path.exists(scene_file)):
        print(
            "\033[91mScene File Not Found, Please Run 'python prepare.py' in the root dir\033[0m")
        exit(-1)

    scene: mi.Scene = mi.load_file(scene_file)
    width, height = scene.sensors()[0].film().size()

    ui_gpu_on = False
    if (args.force_gpu_ui):
        ui_gpu_on = True
    else:
        if sys.platform == "win32":
            ui_gpu_on = True
        elif sys.platform == "linux":
            ui_gpu_on = False
    ui = UI(width, height, ui_gpu_on)

    optix_denoiser_on = False
    denoiser = None
    index = 0
    update_frame = True
    spp = 4

    use_same_seed = False
    same_seed = 0
    img = None
    acc = False
    img_acc = None
    num_acc = 0

    while not ui.should_close():
        if (not update_frame):
            time.sleep(1 / 60)
        ui.begin_frame()
        value_changed = False
        vc, update_frame = imgui.checkbox("Update Frame", update_frame)
        value_changed = value_changed or vc
        vc, spp = imgui.slider_int("spp", spp, 1, 16)
        vc, optix_denoiser_on = imgui.checkbox(
            "Optix Denoiser On", optix_denoiser_on)
        value_changed = value_changed or vc
        vc, acc = imgui.checkbox("Accumulate", acc)
        if (vc):
            num_acc = 0
            img_acc = None
        imgui.text_ansi("Accumulate Frames: {}".format(num_acc + 1))
        value_changed = value_changed or vc

        if (denoiser == None):
            denoiser = OptixDenoiser()

        integrator = scene.integrator()
        if (optix_denoiser_on):
            vc, integrator = denoiser.render_ui(integrator)

        vc, use_same_seed = imgui.checkbox("Use Same Seed", use_same_seed)
        seed = index
        if (vc):
            same_seed = seed
        if (use_same_seed):
            seed = same_seed
        value_changed = value_changed or vc

        if (value_changed or update_frame):
            img = mi.render(scene=scene, spp=spp, seed=seed,
                            integrator=integrator)

            img = img.torch()

            if (acc):
                if (img_acc == None):
                    img_acc = img[::, ::, 0:3:1]
                else:
                    img_acc = img_acc + img[::, ::, 0:3:1]
                num_acc += 1
                img[::, ::, 0:3:1] = img_acc / num_acc

            if (optix_denoiser_on):
                img = denoiser.denoise(img)

            img = tonemap_aces(img)

            # ui.write_texture_cpu(img.cpu().numpy())
            ui.write_texture_gpu(img)

        ui.end_frame()
        index += 1

    ui.close()

    # add these lines to avoid jit_shutdown() error
    # we should use the img variable to notifiy the jit compiler
    result_dir = os.path.join(CURRENT_DIR, "../../results")
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir)
    timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')
    mi.util.write_bitmap(os.path.join(
        result_dir, "output-{}.png".format(timestamp)), img)
