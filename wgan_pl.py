import os  # Importing os for file path operations
from glob import glob  # Importing glob to search for file patterns
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for visualizations

import torch  # Importing PyTorch for deep learning operations
import torch.nn as nn  # Importing PyTorch's neural network module
import torch.optim as optim  # Importing PyTorch's optimization module

import pytorch_lightning as pl  # Importing PyTorch Lightning for training
from pytorch_lightning import Trainer  # Importing Trainer class for model training
from pytorch_lightning.loggers import TensorBoardLogger  # Importing TensorBoard logger

from monai.networks.nets import UNet  # Importing MONAI's UNet for 3D image generation
from monai.networks.nets import DenseNet  # Importing MONAI's DenseNet for critic
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandSpatialCropd,
    SpatialPadd,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader  # Importing MONAI CacheDataset and DataLoader
from monai.config import print_config  # Importing MONAI config printer

print_config()  # Print MONAI configuration for debugging

# Hyperparameter dictionary
hparams = {
    "learning_rate": 1e-4,  # Learning rate for optimizers
    "batch_size": 1,  # Number of samples per training batch
    "image_size": 64,  # Dimensions of input/output 3D images
    "channels_img": 1,  # Number of channels in images (grayscale)
    "num_epochs": 500,  # Number of training epochs
    "features_gen": (16, 32, 64, 128, 256),  # Generator layer features
    "critic_iterations": 5,  # Critic updates per generator update
    "lambda_gp": 10,  # Gradient penalty weight
    "beta1": 0.0,  # Beta1 for Adam optimizer
    "beta2": 0.9,  # Beta2 for Adam optimizer
    "num_workers": 0,  # DataLoader workers
}

# Dataset for managing data loading
class NiftiDataset:
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir  # Set data directory
        self.image_size = image_size  # Set target image size
        self.data_dicts = []  # Initialize list for data dictionaries
        cases = glob(os.path.join(self.data_dir, "*"))  # Find all case directories
        for case in cases:
            vnc_paths = glob(os.path.join(case, "*_VNC.nii.gz"))  # Find VNC files
            mix_paths = glob(os.path.join(case, "*_MIX.nii.gz"))  # Find MIX files
            if vnc_paths and mix_paths:  # Check if both file types exist
                vnc_path, mix_path = vnc_paths[0], mix_paths[0]  # Get file paths
                if os.path.exists(vnc_path) and os.path.exists(mix_path):
                    self.data_dicts.append({"VNC": vnc_path, "MIX": mix_path})
                else:
                    print(f"File not found: VNC or MIX in case: {case}")
            else:
                print(f"Missing VNC or MIX file in case: {case}")
        if not self.data_dicts:
            raise ValueError("No training data found. Please check your data directory.")
        self.transforms = Compose(
            [
                LoadImaged(keys=["VNC", "MIX"]),
                EnsureChannelFirstd(keys=["VNC", "MIX"]),
                ScaleIntensityd(keys=["VNC", "MIX"]),
                RandSpatialCropd(
                    keys=["VNC", "MIX"],
                    roi_size=(self.image_size, self.image_size, self.image_size),
                    random_size=False,
                ),
                SpatialPadd(keys=["VNC", "MIX"], spatial_size=(self.image_size, self.image_size, self.image_size)),
                RandRotate90d(keys=["VNC", "MIX"], prob=0.5, spatial_axes=(0, 1)),
                EnsureTyped(keys=["VNC", "MIX"]),
            ]
        )

    def get_cache_dataset(self):
        return CacheDataset(data=self.data_dicts, transform=self.transforms, cache_rate=1.0, num_workers=hparams["num_workers"])


# LightningModule for WGAN-GP
class WGAN_GP(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.example_input_array = torch.rand(
            1,
            hparams["channels_img"],
            hparams["image_size"],
            hparams["image_size"],
            hparams["image_size"],
        )
        self.automatic_optimization = False
        self.generator = UNet(
            spatial_dims=3,
            in_channels=hparams["channels_img"],
            out_channels=hparams["channels_img"],
            channels=hparams["features_gen"],
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
        )
        self.critic = DenseNet(
            spatial_dims=3,
            in_channels=hparams["channels_img"],
            out_channels=1,  # Single output for WGAN-GP critic
            init_features=32,
            growth_rate=32,
            block_config=(6, 12, 24, 16),  # Default DenseNet-121 configuration
            bn_size=4,
            drop_rate=0.0,
        )

    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)
        device = real.device
        epsilon = torch.rand(batch_size, 1, 1, 1, 1, device=device)
        interpolated_images = epsilon * real + (1 - epsilon) * fake
        interpolated_images.requires_grad_(True)
        mixed_scores = self.critic(interpolated_images)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.size(0), -1)
        gradient_norm = gradient.norm(2, dim=1)
        gp = torch.mean((gradient_norm - 1) ** 2)
        return gp

    def training_step(self, batch, batch_idx):
        vnc, mix = batch["VNC"].float(), batch["MIX"].float()
        opt_gen, opt_critic = self.optimizers()
        for _ in range(self.hparams["critic_iterations"]):
            fake = self.generator(vnc)
            critic_real, critic_fake = self.critic(mix), self.critic(fake.detach())
            gp = self.gradient_penalty(mix, fake.detach())
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + self.hparams["lambda_gp"] * gp
            )
            opt_critic.zero_grad()
            self.manual_backward(loss_critic, retain_graph=True)
            opt_critic.step()
        self.log("loss_critic", loss_critic, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        fake = self.generator(vnc)
        gen_loss = -torch.mean(self.critic(fake))
        recon_loss = nn.functional.mse_loss(fake, mix)  # Reconstruction loss
        total_gen_loss = gen_loss + recon_loss
        opt_gen.zero_grad()
        self.manual_backward(total_gen_loss)
        opt_gen.step()
        self.log("loss_gen", gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("gradient_penalty", gp, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        opt_gen = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams["learning_rate"],
            betas=(self.hparams["beta1"], self.hparams["beta2"]),
        )
        opt_critic = optim.Adam(
            self.critic.parameters(),
            lr=self.hparams["learning_rate"],
            betas=(self.hparams["beta1"], self.hparams["beta2"]),
        )
        return [opt_gen, opt_critic], []


# Main script execution
if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, "..", "data")
    dataset_class = NiftiDataset(
        data_dir=data_dir,
        image_size=hparams["image_size"],
    )
    dataset = dataset_class.get_cache_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
    )

    model = WGAN_GP(hparams)

    logger = TensorBoardLogger("tb_logs", name="WGAN_GP")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="WGAN_GP-{epoch:02d}-{loss_gen:.2f}-{loss_critic:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="loss_gen",
        mode="min",
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        max_epochs=hparams["num_epochs"],
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, dataloader)