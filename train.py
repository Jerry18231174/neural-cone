# Basics
import os
import json
import argparse

# Computational
import numpy as np
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Mitsuba
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")

# Custom
from src.model.radiosity import NeuralRadiosity, NeuralConeRadiosity
from src.dataset.camera import CameraDataset
from src.util.progress_bar import StepRichProgressBar, find_best_ckpt


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))

# def train_sdf(config: dict, args: argparse.Namespace):
#     """
#     Train SDF model
#     """
#     # Load model
#     # mesh_path = os.path.join("scenes", args.scene, "raw_meshes", "merged.ply")
#     mesh_path = os.path.join("scenes", "remeshed.ply")
#     model = NGPSDF(config["model"]["sdf"], mesh_path).to("cuda")
#     model.train()

#     # Load dataset
#     dataset = SDFDataset(mesh_path, size=1, batch_size=config["sample"]["sdf"]["batch_size"])

#     # Load optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

#     # Set up training directory or load from checkpoint
#     ckpt_steps = 0
#     if not os.path.exists(os.path.join("out", args.scene)):
#         os.makedirs(os.path.join("out", args.scene))
#         os.makedirs(os.path.join("out", args.scene, "checkpoints"))
#     elif args.model_ckpt is not None:
#         ckpt_steps = int(args.model_ckpt)
#         model.load_state_dict(torch.load(os.path.join(
#             "out", args.scene, "checkpoints", args.model_ckpt + "_model.pth"
#         )))

#     # Train
#     tqdm_iter = tqdm(range(config["train"]["epochs"] - ckpt_steps))
#     for step in tqdm_iter:
#         batch = dataset[step]
#         points = batch["xyz"].to("cuda")
#         sdf = model(points)
        
#         # Compute loss
#         rel_res = (sdf - batch["sdf"]) / (torch.abs(sdf) + torch.abs(batch["sdf"]) + 1e-3)
#         loss = torch.mean(rel_res ** 2)
        
#         # Optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         tqdm_iter.set_description("loss: {:.4e}".format(loss.item()))

#         if (step + 1) % config["train"]["save_every"] == 0:
#             torch.save(model.state_dict(), os.path.join(
#                 "out", args.scene, "checkpoints", f"{step + ckpt_steps + 1}_model.pth"
#             ))


def train(config: dict, args: argparse.Namespace):
    """
    Train model
    """

    # Load scene
    scene = mi.load_file(os.path.join("scenes", args.scene, "scene.xml"))

    # Tensorboard logger
    logger = TensorBoardLogger(
        os.path.join("out", args.scene, "tb_logs"),
        name=config["model"]["name"]
    )

    # Set up training directory or load from checkpoint
    if not os.path.exists(os.path.join("out", args.scene, "checkpoints", config["model"]["name"])):
        os.makedirs(os.path.join("out", args.scene, "checkpoints", config["model"]["name"]))

    # Load checkpoint files, choose the best one, set corresponding step
    ckpt_dir = os.path.join("out", args.scene, "checkpoints", config["model"]["name"])
    ckpt_path, ckpt_step = find_best_ckpt(ckpt_dir, metric="loss")

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        mode="min",
        save_top_k=3,
        save_last=False,
        every_n_train_steps=500,
        dirpath=ckpt_dir,
        filename="{step}_{loss:.4f}"
    )

    # Lightning trainer
    max_steps = np.ceil(config["train"]["epochs"] / get_world_size())
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        # precision=16,  # mixed precision
        max_epochs=-1,
        max_steps=max_steps,
        logger=logger,
        callbacks=[checkpoint_callback, StepRichProgressBar(total_steps=max_steps)],
        log_every_n_steps=1,
    )
    
    # Train
    data_loader = DataLoader(
        CameraDataset(os.path.join("scenes", args.scene, "camera_poses")),
        collate_fn=lambda x: x[0],
        batch_size=1,
    )

    if ckpt_path is None:
        # Load model
        if config["model"]["name"] == "NR":
            model = NeuralRadiosity(config["model"]["ray"], config, scene)
        elif config["model"]["name"][:3] == "NCR":
            model = NeuralConeRadiosity(config["model"], config, scene)
        model.train()

        trainer.fit(model, train_dataloaders=data_loader)
    else:
        # Train from the chosen checkpoint
        if config["model"]["name"] == "NR":
            model = NeuralRadiosity.load_from_checkpoint(
                ckpt_path,
                config=config["model"]["ray"],
                pipeline_config=config,
                scene=scene
            )
        elif config["model"]["name"][:3] == "NCR":
            model = NeuralConeRadiosity.load_from_checkpoint(
                ckpt_path,
                config=config["model"],
                pipeline_config=config,
                scene=scene
            )
        
        trainer.fit(model, ckpt_path=ckpt_path, train_dataloaders=data_loader)
    

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="ncr")
    parser.add_argument("-s", "--scene", type=str, default="veach-ajar")
    # parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)
    
    # Train model
    train(config, args)
