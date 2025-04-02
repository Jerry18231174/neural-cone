import numpy as np
import torch

import mcubes, trimesh
import mesh2sdf
from mesh_to_sdf import mesh_to_voxels
import pysdf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import os
import json
import time
import pickle

from src.model.sdf import NGPSDF, GridSDF


def normalize_mesh(mesh):
    """
    Normalize vertex coordinates between [-1, 1]
    """
    vertices = mesh.vertices
    min_xyz = np.min(vertices, axis=0) - 1e-2
    max_xyz = np.max(vertices, axis=0) + 1e-2
    center = (min_xyz + max_xyz) / 2
    scale = np.max(max_xyz - min_xyz) / 2
    vertices = (vertices - center) / scale
    mesh.vertices = vertices
    return mesh


def convert_mesh_to_sdf(mesh_path, size=256, cache_path="out/cache/sdf.pkl"):
    """
    Convert mesh to SDF.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache_info = pickle.load(f)
        print("SDF cache loaded.")
    else:
        grid_size = 2 / size
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        z = np.linspace(-1, 1, size)
        xx, yy, zz = np.meshgrid(x, y, z)
        xyz = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        start_time = time.time()
        print("Generating SDF ...")
        mesh = trimesh.load(mesh_path, process=False)
        mesh = normalize_mesh(mesh)
        sdf, mesh = mesh2sdf.compute(vertices=mesh.vertices, faces=mesh.faces, size=size, return_mesh=True)
        # sdf = mesh_to_voxels(mesh, size)
        # sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)
        # sdf = sdf_fn(xyz)
        end_time = time.time()
        print(f"SDF generation complete: {end_time - start_time:.2f} seconds.")
        # sdf: [size, size, size]

        sdf = sdf.reshape(-1, 1).astype(np.float32)

        tsdf = sdf.copy()   
        tsdf[sdf < -0.1] = -0.1
        tsdf[sdf > 0.1] = 0.1
        occ = np.zeros_like(sdf)
        occ[sdf < 0] = 1
        cache_info = {
            # "xyz": xyz,
            # "occ": occ,
            "sdf": sdf,
            # "tsdf": tsdf
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_info, f)
    
    return cache_info


def sdf_to_mesh(sdf_info, mesh_path="out/mesh/mcmesh.obj", size=256):
    sdf = sdf_info["sdf"]
    sdf = sdf.reshape(size, size, size)
    # sdf = sdf.transpose(1, 0, 2)
    sdf = mcubes.smooth(sdf)
    vertices, triangles = mcubes.marching_cubes(sdf, 0.0)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.export(mesh_path)
    return mesh


def get_sdf_from_model(config, scene, size=256):
    """
    Get SDF from model.
    """
    mesh_dir = os.path.join("scenes", scene, "raw_meshes", "merged.ply")
    ckpt_dir = os.path.join("out", scene, "sdf_cache.npy")

    model = GridSDF(config, mesh_dir)
    model.compute(ckpt_dir)

    x = np.linspace(-0.99, 0.99, size)
    y = np.linspace(-0.99, 0.99, size)
    z = np.linspace(-0.99, 0.99, size)
    xx, yy, zz = np.meshgrid(x, y, z)
    xyz = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    xyz = torch.from_numpy(xyz).to("cuda")
    xyz = xyz * model.scale + model.center
    
    with torch.no_grad():
        t1 = time.time()
        sdf = model(xyz.to(torch.float32)).cpu().numpy()
        t2 = time.time()
        print(f"SDF random access time: {t2 - t1:.2f} seconds.")
    sdf = sdf.reshape(-1, 1).astype(np.float32)
    return {"sdf": sdf}


def visualize_sdf(sdf_info, size=256):
    sdf = sdf_info["sdf"]
    sdf = sdf.reshape(size, size, size)
    sdf = sdf
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)
    
    slice_idx = size // 2
    img = ax.imshow(sdf[slice_idx], cmap="coolwarm", vmin=-0.1, vmax=0.1)
    ax.set_title(f"SDF Slice at Z = {slice_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(img, ax=ax, label="SDF Value")

    # Add slider bar
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Z Slice", 0, size-1, valinit=slice_idx, valstep=1)
    
    def update(val):
        slice_idx = int(slider.val)
        img.set_data(sdf[slice_idx])
        ax.set_title(f"SDF Slice at Z = {slice_idx}")
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    mesh_path = "scenes/cornell-box/raw_meshes/merged.ply"
    # mesh_path = "scenes/remeshed.ply"
    with open("configs/ncr.json", "r") as f:
        config = json.load(f)
    size = 100
    # sdf_info = convert_mesh_to_sdf(mesh_path, size=size, cache_path="out/cache/sdf_merged256.pkl")
    sdf_info = get_sdf_from_model(config["model"]["sdf"], "cornell-box", size=size)
    # mesh = sdf_to_mesh(sdf_info, mesh_path="out/mesh/mcmesh_merge256.obj", size=size)
    visualize_sdf(sdf_info, size=size)
    print("Done.")
