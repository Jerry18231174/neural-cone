import mitsuba as mi
import drjit as dr
import argparse
import torch

from src.denoise.denoiser_wrap import *
from src.denoise.simple_denoise.filter import FilterTasks
import time
from tqdm import trange
import glfw


def sync():
    torch.cuda.synchronize()
    dr.sync_device()
    torch.cuda.synchronize()
    dr.sync_device()
    return time.time()

def fxaa(input_file: str, output_file: str):
    if (output_file is None):
        segs = input_file.split(".")
        output_file = ".".join(segs[:-1]) + ".fxaa." + segs[-1]

    img_in = mi.Bitmap(input_file)
    scene: mi.Scene = mi.load_dict(mi.cornell_box())  # no use, just for convience
    fxaa_denoiser = FilterTasks(None, scene, *img_in.size(), img_in.size())

    img_in_tensor: torch.Tensor = mi.TensorXf(img_in).torch().cuda()

    # LHS with FXAA
    sync()
    img_out = fxaa_denoiser.fetch_denoised_result_headless(img_in_tensor, True)
    sync()
    mi.util.write_bitmap(output_file, img_out)


if __name__ == "__main__":
    # python scripts/fxaa.py -i screenshots/test.exr
    parser = argparse.ArgumentParser(description="FXAA")
    # input image
    parser.add_argument("-i", type=str, help="input image")
    parser.add_argument("-d", type=str, help="input image directory")
    # optional arguments, output image
    parser.add_argument("-o", type=str, help="output image")

    args = parser.parse_args()

    if args.i is None and args.d is None:
        print("Please provide input image(or image directory)")
        parser.print_help()
        exit()

    # fxaa
    # opengl init
    if not glfw.init():
        print("Failed to initialize GLFW")
        exit()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, False)  # headless
    window = glfw.create_window(1, 1, "Off-Screen", None, None)
    if not window:
        print("Failed to create window")
        glfw.terminate()
        exit()
    glfw.make_context_current(window)

    if(args.d is not None):
        import os
        for root, dirs, files in os.walk(args.d):
            for file in files:
                if file.endswith(".exr") or file.endswith(".png"):
                    fxaa(os.path.join(root, file), None)
    else:
        fxaa(args.i, args.o)
