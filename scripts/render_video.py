# Basics
import os
import json
import argparse
from tqdm import tqdm
import ffmpeg

# Computational
import numpy as np
import torch

# Custom
from src.viewer.camera import FPSCamera, MovingCamera
from render import load_render_vars

# Mitsuba
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def render_video(render_vars: dict, script: dict, args: argparse.Namespace) -> torch.Tensor:
    # Load render variables
    scene: mi.Scene = render_vars["scene"]
    params = mi.traverse(scene)
    
    fps = script["fps"]
    duration = script["duration"]
    spp = script["spp"]
    render_mode = script["render_mode"]

    if render_mode == "LHS":
        integrator = render_vars["integrators"][args.config]
        integrator.render_mode = "LHS"
        spp = 1
    elif render_mode == "RHS":
        integrator = render_vars["integrators"][args.config]
        integrator.render_mode = "RHS"
        integrator.spp = spp
        spp = 1
    elif render_mode == "PT":
        integrator = render_vars["integrators"]["path"]
    else:
        raise ValueError("Invalid render mode:", render_mode)
    
    # Load moving camera
    cameras = []
    for camera_path in script["cameras"]:
        with np.load(camera_path, allow_pickle=True) as data:
            extrinsics = data["extrinsics"]
            intrinsics = data["intrinsics"].item()
        cameras.append(FPSCamera(intrinsics, extrinsics, speed=1))
    cameras = MovingCamera(cameras)
    
    # Render
    cache_format = os.path.join("out", "video_cache", "{:02d}{:04d}.png")
    for section in range(len(script["cameras"])):
        for frame in tqdm(range(duration * fps)):
            cache_path = cache_format.format(section, frame)
            # Skip if already rendered
            if os.path.exists(cache_path):
                continue

            # Interpolate camera
            cam_v = frame / (duration * fps)
            camera = cameras.get_camera(section, cam_v)
            params['PerspectiveCamera.to_world'] = mi.Matrix4f(camera.get_transform()[None, ...])
            params['PerspectiveCamera.x_fov'] = mi.Float32(camera.get_x_fov()[None, ...])
            params.update()

            # Render img
            seed = np.random.randint(0, 1000000)
            img = mi.render(scene, integrator=integrator, seed=seed, spp=spp).torch()
            dr.sync_device()
            torch.cuda.synchronize()
            mi.util.write_bitmap(cache_path, img)
            # print("Image saved to", output_path)
    
    # Convert to video
    imgs_path = os.path.join("out", "video_cache", "*.png")
    video_path = os.path.join("out", "{}.mp4".format(script["name"]))
    ffmpeg.input(imgs_path, pattern_type='glob', framerate=fps).output(
        video_path,
        vcodec='libx264',
        pix_fmt='yuv420p',
    ).run()
    print("Video saved to {}".format(video_path))


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="visualize")
    parser.add_argument("-c", "--config", type=str, default="ncr")
    parser.add_argument("-s", "--scene", type=str, default="veach-ajar")
    parser.add_argument("-m", "--model_ckpt", type=str, default="20000")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    args = parse_args()
    
    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)
    
    # Load video script
    with open(os.path.join("configs", args.script + ".json"), "r") as f:
        script = json.load(f)
    
    # Render
    render_vars = load_render_vars(config, args)
    render_video(render_vars, script, args)
    