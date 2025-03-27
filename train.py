# Basics
import os
import json
import argparse
import trimesh
from tqdm import tqdm

# Computational
import numpy as np
import torch

# Mitsuba
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")

# Custom
from src.model.sdf import NGPSDF, GridSDF
from src.model.radiosity import NeuralRadiosity, NeuralConeRadiosity
from src.sample.lhs_rhs import LHSRHS
from src.dataset.sdf import SDFDataset


def train_sdf(config: dict, args: argparse.Namespace):
    """
    Train SDF model
    """
    # Load model
    # mesh_path = os.path.join("scenes", args.scene, "raw_meshes", "merged.ply")
    mesh_path = os.path.join("scenes", "remeshed.ply")
    model = NGPSDF(config["model"]["sdf"], mesh_path).to("cuda")
    model.train()

    # Load dataset
    dataset = SDFDataset(mesh_path, size=1, batch_size=config["sample"]["sdf"]["batch_size"])

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    # Set up training directory or load from checkpoint
    ckpt_steps = 0
    if not os.path.exists(os.path.join("out", args.scene)):
        os.makedirs(os.path.join("out", args.scene))
        os.makedirs(os.path.join("out", args.scene, "checkpoints"))
    elif args.model_ckpt is not None:
        ckpt_steps = int(args.model_ckpt)
        model.load_state_dict(torch.load(os.path.join(
            "out", args.scene, "checkpoints", args.model_ckpt + "_model.pth"
        )))

    # Train
    tqdm_iter = tqdm(range(config["train"]["epochs"] - ckpt_steps))
    for step in tqdm_iter:
        batch = dataset[step]
        points = batch["xyz"].to("cuda")
        sdf = model(points)
        
        # Compute loss
        rel_res = (sdf - batch["sdf"]) / (torch.abs(sdf) + torch.abs(batch["sdf"]) + 1e-3)
        loss = torch.mean(rel_res ** 2)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tqdm_iter.set_description("loss: {:.4e}".format(loss.item()))

        if (step + 1) % config["train"]["save_every"] == 0:
            torch.save(model.state_dict(), os.path.join(
                "out", args.scene, "checkpoints", f"{step + ckpt_steps + 1}_model.pth"
            ))


def train(config: dict, args: argparse.Namespace):
    """
    Train model
    """
    mesh_path = os.path.join("scenes", args.scene, "raw_meshes", "merged.ply")

    # Load SDF
    sdf_model = GridSDF(config["model"]["sdf"], mesh_path)
    sdf_cache_path = os.path.join("out", args.scene, "sdf_cache.npy")
    sdf_model.compute(sdf_cache_path)

    # Load scene
    scene = mi.load_file(os.path.join("scenes", args.scene, "scene.xml"))
    bbox = scene.bbox()
    bbox = torch.tensor([bbox.min - 1e-2, bbox.max + 1e-2], dtype=torch.float32, device="cuda")

    # Load model
    # model = NeuralRadiosity(config["model"]["ray"], bbox).to("cuda")
    model = NeuralConeRadiosity(config["model"], bbox, sdf_model).to("cuda")
    model.train()

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    # Set up training directory or load from checkpoint
    ckpt_steps = 0
    if not os.path.exists(os.path.join("out", args.scene)):
        os.makedirs(os.path.join("out", args.scene))
        os.makedirs(os.path.join("out", args.scene, "checkpoints"))
    elif args.model_ckpt is not None:
        ckpt_steps = int(args.model_ckpt)
        model.load_state_dict(torch.load(os.path.join(
            "out", args.scene, "checkpoints", args.model_ckpt + "_model.pth"
        )))
    
    # Train
    tqdm_iter = tqdm(range(config["train"]["epochs"] - ckpt_steps))
    for step in tqdm_iter:
        # Sample
        lhs_rhs = LHSRHS(
            scene=scene,
            point_num=config["sample"]["n_points"],
            dirs_per_point=config["sample"]["n_dirs_per_point"],
        )
        lhs_rhs.sample(seed=step)

        # Forward pass
        result = model(lhs_rhs)
        lhs_color = result["lhs"]
        rhs_color = result["rhs"].detach()

        # Compute loss
        nr_norm = (rhs_color + lhs_color).detach() / 2 + 1e-1
        loss = torch.mean(((rhs_color - lhs_color) / nr_norm) ** 2)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        dr.flush_malloc_cache()
        
        tqdm_iter.set_description("loss: {:.4e}".format(loss.item()))

        if (step + 1) % config["train"]["save_every"] == 0:
            torch.save(model.state_dict(), os.path.join(
                "out", args.scene, "checkpoints", f"{step + ckpt_steps + 1}_model.pth"
            ))
        if (step + 1) % 100 == 0:
            print("lhs color", lhs_color[:3])
            print("rhs color", rhs_color[:3])


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="grid")
    parser.add_argument("-s", "--scene", type=str, default="veach-ajar")
    parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    parser.add_argument("-v", "--viewer", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)
    
    # Train model
    train(config, args)
