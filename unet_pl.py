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
from monai.networks.nets.resnet import (
    ResNet,
    get_inplanes,
)  # Importing MONAI's ResNet for critic
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    EnsureTyped,
)

# Importing MONAI transforms for preprocessing medical images

from monai.data import (
    CacheDataset,
    DataLoader,
)  # Importing MONAI dataset and dataloader
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

# DataModule for managing data loading
class NiftiDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir, batch_size, image_size, num_workers=0
    ):  # Initialize with paths and settings
        super().__init__()  # Initialize LightningDataModule
        self.data_dir = data_dir  # Set data directory
        self.batch_size = batch_size  # Set batch size
        self.image_size = image_size  # Set target image size
        self.num_workers = num_workers  # Set number of worker processes

    def prepare_data(self):  # Placeholder for data preparation
        pass  # No data preparation required

    def setup(self, stage=None):  # Dataset setup
        cases = glob(os.path.join(self.data_dir, "*"))  # Find all case directories
        data_dicts = []  # Initialize list for data dictionaries
        for case in cases:  # Iterate through case directories
            vnc_paths = glob(os.path.join(case, "*_VNC.nii.gz"))  # Find VNC files
            mix_paths = glob(os.path.join(case, "*_MIX.nii.gz"))  # Find MIX files
            if vnc_paths and mix_paths:  # Check if both file types exist
                vnc_path, mix_path = vnc_paths[0], mix_paths[0]  # Get file paths
                if os.path.exists(vnc_path) and os.path.exists(
                    mix_path
                ):  # Validate file existence
                    data_dicts.append(
                        {"VNC": vnc_path, "MIX": mix_path}
                    )  # Add file paths to data_dict
                else:
                    print(
                        f"File not found: VNC or MIX in case: {case}"
                    )  # Log missing file
            else:
                print(f"Missing VNC or MIX file in case: {case}")  # Log incomplete case
        if not data_dicts:  # Validate data availability
            raise ValueError(
                "No training data found. Please check your data directory."
            )  # Raise error
        self.transforms = Compose(
            [  # Define preprocessing transforms
                LoadImaged(keys=["VNC", "MIX"]),  # Load NIfTI images
                EnsureChannelFirstd(keys=["VNC", "MIX"]),  # Ensure channel-first format
                ScaleIntensityd(keys=["VNC", "MIX"]),  # Normalize intensity values
                Resized(
                    keys=["VNC", "MIX"],
                    spatial_size=(self.image_size, self.image_size, self.image_size),
                ),  # Resize images
                EnsureTyped(keys=["VNC", "MIX"]),  # Convert to PyTorch tensors
            ]
        )
        self.train_dataset = CacheDataset(  # Create dataset with caching
            data=data_dicts,  # Provide data dictionaries
            transform=self.transforms,  # Apply preprocessing transforms
            cache_rate=1.0,  # Cache all data
            num_workers=self.num_workers,  # Use specified workers
        )

    def train_dataloader(self):  # Return DataLoader for training
        return DataLoader(
            self.train_dataset,  # Dataset to load
            batch_size=self.batch_size,  # Number of samples per batch
            shuffle=True,  # Enable shuffling
            num_workers=self.num_workers,  # Number of worker processes
        )


# LightningModule for WGAN-GP
class WGAN_GP(pl.LightningModule):
    def __init__(self, hparams):  # Initialize WGAN-GP
        super().__init__()  # Initialize LightningModule
        self.save_hyperparameters(hparams)  # Save hyperparameters for easy access
        self.example_input_array = torch.rand(  # Define example input for tracing
            1,
            hparams["channels_img"],
            hparams["image_size"],
            hparams["image_size"],
            hparams["image_size"],
        )
        self.automatic_optimization = False  # Disable automatic optimization
        self.generator = UNet(  # Define UNet generator for style transfer
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
        block_inplanes = get_inplanes()  # Get ResNet input planes
        self.critic = ResNet(  # Define ResNet critic
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=block_inplanes,
            spatial_dims=3,
            n_input_channels=hparams["channels_img"],
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type="B",
            widen_factor=1.0,
            num_classes=1,
            feed_forward=True,
            bias_downsample=True,
            act=("relu", {"inplace": True}),
            norm="batch",
        )

    def gradient_penalty(self, real, fake):  # Calculate gradient penalty for WGAN-GP
        batch_size = real.size(0)  # Batch size
        device = real.device  # Device (CPU/GPU)
        epsilon = torch.rand(
            batch_size, 1, 1, 1, 1, device=device
        )  # Random weights for interpolation
        interpolated_images = (
            epsilon * real + (1 - epsilon) * fake
        )  # Interpolate between real and fake
        interpolated_images.requires_grad_(True)  # Enable gradient tracking
        mixed_scores = self.critic(interpolated_images)  # Pass through critic
        gradient = torch.autograd.grad(  # Calculate gradients
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.size(0), -1)  # Flatten gradients
        gradient_norm = gradient.norm(2, dim=1)  # Compute L2 norm
        gp = torch.mean((gradient_norm - 1) ** 2)  # Calculate penalty
        return gp  # Return gradient penalty

    def training_step(self, batch, batch_idx):  # Training loop for WGAN-GP
        vnc, mix = batch["VNC"].float(), batch["MIX"].float()  # Get batch data
        opt_gen, opt_critic = self.optimizers()  # Get optimizers
        for _ in range(
            self.hparams["critic_iterations"]
        ):  # Update critic multiple times
            fake = self.generator(vnc)  # Generate fake images
            critic_real, critic_fake = self.critic(mix), self.critic(
                fake.detach()
            )  # Evaluate real and fake
            gp = self.gradient_penalty(mix, fake.detach())  # Calculate gradient penalty
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + self.hparams["lambda_gp"] * gp
            )  # Critic loss
            opt_critic.zero_grad()  # Reset gradients for critic
            self.manual_backward(loss_critic, retain_graph=True)  # Backpropagate
            opt_critic.step()  # Update critic weights
        self.log(
            "loss_critic",
            loss_critic,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )  # Log critic loss
        fake = self.generator(vnc)  # Generate new fake images
        gen_loss = -torch.mean(self.critic(fake))  # Generator loss
        opt_gen.zero_grad()  # Reset gradients for generator
        self.manual_backward(gen_loss)  # Backpropagate
        opt_gen.step()  # Update generator weights
        self.log(
            "loss_gen",
            gen_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )  # Log generator loss

    def configure_optimizers(self):  # Configure optimizers for generator and critic
        opt_gen = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams["learning_rate"],
            betas=(self.hparams["beta1"], self.hparams["beta2"]),
        )  # Generator optimizer

        opt_critic = optim.Adam(
            self.critic.parameters(),
            lr=self.hparams["learning_rate"],
            betas=(self.hparams["beta1"], self.hparams["beta2"]),
        )  # Critic optimizer
        
        return [opt_gen, opt_critic], []  # Return optimizers


# Main script execution
if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    data_dir = os.path.join(root_dir, "..", "data")  # Define data directory path
    data_module = NiftiDataModule(
        data_dir=data_dir,
        batch_size=hparams["batch_size"],
        image_size=hparams["image_size"],
        num_workers=hparams["num_workers"],
    )  # Initialize DataModule

    model = WGAN_GP(hparams)  # Initialize WGAN-GP model

    logger = TensorBoardLogger("tb_logs", name="WGAN_GP")  # Set up TensorBoard logger
    checkpoint_callback = pl.callbacks.ModelCheckpoint(  # Set up model checkpointing
        dirpath="checkpoints",
        filename="WGAN_GP-{epoch:02d}-{loss_gen:.2f}-{loss_critic:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="loss_gen",
        mode="min",
    )

    # Determine the accelerator and devices based on CUDA availability
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1  # You can set this to 'auto' or the number of GPUs you want to use
    else:
        accelerator = 'cpu'
        devices = 1  # Must be an int > 0 for CPUAccelerator
    # Set up the Trainer
        trainer = Trainer(
        max_epochs=hparams['num_epochs'],
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data_module)  # Start training