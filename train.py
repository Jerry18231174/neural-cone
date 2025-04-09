# Basics
import os
import json
import argparse
import glob
from tqdm import tqdm

# Computational
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

# Mitsuba
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")

# Custom
from src.model.sdf import NGPSDF, GridSDF
from src.model.radiosity import NeuralRadiosity, NeuralConeRadiosity, get_model_bbox
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

    # Load scene
    scene = mi.load_file(os.path.join("scenes", args.scene, "scene.xml"))

    # Load model
    if config["model"]["name"] == "NR":
        model = NeuralRadiosity(config["model"]["ray"], config, scene)
    elif config["model"]["name"] == "NCR":
        # # Load SDF
        # mesh_path = os.path.join("scenes", args.scene, "raw_meshes", "merged.ply")
        # sdf_model = GridSDF(config["model"]["sdf"], mesh_path)
        # sdf_cache_path = os.path.join("out", args.scene, "sdf_cache.npy")
        # sdf_model.compute(sdf_cache_path)

        model = NeuralConeRadiosity(config["model"], config, scene)
    
    model.train()

    # Tensorboard logger
    logger = TensorBoardLogger(
        os.path.join("out", args.scene, "tb_logs"),
        name=args.scene + "_" + config["model"]["name"]
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=500,
        dirpath=os.path.join("out", args.scene, "checkpoints", config["model"]["name"]),
        filename="{step}_loss{loss:.3f}.pth"
    )

    # Lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        # precision=16,  # mixed precision
        max_epochs=-1,
        max_steps=config["train"]["epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, RichProgressBar()],
    )

    # Set up training directory or load from checkpoint
    if not os.path.exists(os.path.join("out", args.scene)):
        os.makedirs(os.path.join("out", args.scene, "checkpoints", config["model"]["name"]))
    
    # Train
    fake_loader = DataLoader(TensorDataset(torch.arange(1)))
    if args.model_ckpt is None:
        trainer.fit(model, train_dataloaders=fake_loader)
    else:
        ckpts = glob.glob(os.path.join(
            "out", args.scene, "checkpoints", config["model"]["name"], args.model_ckpt + "*.pth"
        ))
        if len(ckpts) == 0:
            raise ValueError(f"No checkpoint found for {args.model_ckpt}")
        
        # Train from the chosen checkpoint
        trainer.fit(model, ckpt_path=ckpts[0], train_dataloaders=fake_loader)
    

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="ncr")
    parser.add_argument("-s", "--scene", type=str, default="veach-ajar")
    parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)
    
    # Train model
    train(config, args)
