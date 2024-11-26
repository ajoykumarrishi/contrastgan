# unet_pl.py

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from monai.networks.nets import UNet
from monai.networks.nets.resnet import ResNet, get_inplanes  # Updated import
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader
from monai.config import print_config

# Print MONAI configuration
print_config()

# Hyperparameters and configurations
hparams = {
    'learning_rate': 1e-4,
    'batch_size': 1,
    'image_size': 64,
    'channels_img': 1,
    'num_epochs': 500,
    'features_gen': (16, 32, 64, 128, 256),
    'critic_iterations': 5,
    'lambda_gp': 10,
    'beta1': 0.0,
    'beta2': 0.9,
    'num_workers': 0,  # Adjust based on your system
}

# DataModule for handling data loading
class NiftiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def prepare_data(self):
        # No preparation needed
        pass

    def setup(self, stage=None):
        # Create dataset
        cases = glob(os.path.join(self.data_dir, "*"))
        data_dicts = []
        for case in cases:
            vnc_paths = glob(os.path.join(case, "*_VNC.nii.gz"))
            mix_paths = glob(os.path.join(case, "*_MIX.nii.gz"))

            if vnc_paths and mix_paths:
                vnc_path = vnc_paths[0]
                mix_path = mix_paths[0]
                if os.path.exists(vnc_path) and os.path.exists(mix_path):
                    data_dicts.append({"VNC": vnc_path, "MIX": mix_path})
                else:
                    print(f"File not found: VNC or MIX in case: {case}")
            else:
                print(f"Missing VNC or MIX file in case: {case}")

        if not data_dicts:
            raise ValueError("No training data found. Please check your data directory.")

        # Define transforms using dictionary-based transforms
        self.transforms = Compose([
            LoadImaged(keys=["VNC", "MIX"]),
            EnsureChannelFirstd(keys=["VNC", "MIX"]),
            ScaleIntensityd(keys=["VNC", "MIX"]),
            Resized(keys=["VNC", "MIX"], spatial_size=(self.image_size, self.image_size, self.image_size)),
            EnsureTyped(keys=["VNC", "MIX"]),
        ])

        self.train_dataset = CacheDataset(
            data=data_dicts,
            transform=self.transforms,
            cache_rate=1.0,  # Cache all data
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

# WGAN-GP model using PyTorch Lightning for Style Transfer
class WGAN_GP(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(hparams)
        self.example_input_array = torch.rand(
            1, hparams['channels_img'],
            hparams['image_size'], hparams['image_size'], hparams['image_size']
        )
        # Set automatic optimization to False for manual optimization
        self.automatic_optimization = False

        # Use MONAI's UNet for the generator (style transfer)
        self.generator = UNet(
            spatial_dims=3,
            in_channels=hparams['channels_img'],
            out_channels=hparams['channels_img'],
            channels=hparams['features_gen'],
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            act='PRELU',
            norm='INSTANCE',
            dropout=0.0,
        )

        # Initialize the critic using MONAI's ResNet
        block_inplanes = get_inplanes()
        self.critic = ResNet(
            block='basic',  # Use 'basic' for ResNetBlock
            layers=[2, 2, 2, 2],  # ResNet18 configuration
            block_inplanes=block_inplanes,
            spatial_dims=3,
            n_input_channels=hparams['channels_img'],
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type='B',
            widen_factor=1.0,
            num_classes=1,  # Output a single scalar
            feed_forward=True,
            bias_downsample=True,
            act=('relu', {'inplace': True}),
            norm='batch',
        )

    def forward(self, x):
        return self.generator(x)

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
        vnc = batch['VNC']
        mix = batch['MIX']
        vnc = vnc.float()
        mix = mix.float()
        opt_gen, opt_critic = self.optimizers()

        # Train Critic
        for _ in range(self.hparams['critic_iterations']):
            fake = self.generator(vnc)
            critic_real = self.critic(mix)
            critic_fake = self.critic(fake.detach())
            gp = self.gradient_penalty(mix, fake.detach())
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.hparams['lambda_gp'] * gp

            opt_critic.zero_grad()
            self.manual_backward(loss_critic, retain_graph=True)
            opt_critic.step()

        self.log('loss_critic', loss_critic, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Train Generator
        fake = self.generator(vnc)
        gen_loss = -torch.mean(self.critic(fake))

        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()

        self.log('loss_gen', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log images every 10 batches
        if batch_idx % 10 == 0:
            with torch.no_grad():
                fake = self.generator(vnc)
                # Assuming batch size is 1
                slice_idx = fake.shape[2] // 2
                vnc_slice = vnc[0, 0, slice_idx, :, :].cpu().numpy()
                mix_slice = mix[0, 0, slice_idx, :, :].cpu().numpy()
                fake_slice = fake[0, 0, slice_idx, :, :].cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(vnc_slice, cmap='gray')
                axes[0].set_title('VNC (Input)')
                axes[0].axis('off')
                axes[1].imshow(mix_slice, cmap='gray')
                axes[1].set_title('MIX (Real)')
                axes[1].axis('off')
                axes[2].imshow(fake_slice, cmap='gray')
                axes[2].set_title('Generated')
                axes[2].axis('off')
                plt.tight_layout()
                self.logger.experiment.add_figure('VNC vs MIX vs Generated', fig, global_step=self.global_step)
                plt.close(fig)

    def configure_optimizers(self):
        opt_gen = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams['learning_rate'],
            betas=(self.hparams['beta1'], self.hparams['beta2'])
        )
        opt_critic = optim.Adam(
            self.critic.parameters(),
            lr=self.hparams['learning_rate'],
            betas=(self.hparams['beta1'], self.hparams['beta2'])
        )
        return [opt_gen, opt_critic], []

# Main script
if __name__ == '__main__':
    # Set up directories
    root_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    data_dir = os.path.join(root_dir, '..', 'data')  # Adjust the path to your data directory

    # Instantiate the DataModule
    data_module = NiftiDataModule(
        data_dir=data_dir,
        batch_size=hparams['batch_size'],
        image_size=hparams['image_size'],
        num_workers=hparams['num_workers']
    )

    # Instantiate the model
    model = WGAN_GP(hparams)

    # Set up the logger
    logger = TensorBoardLogger("tb_logs", name="WGAN_GP")

    # Set up callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='WGAN_GP-{epoch:02d}-{loss_gen:.2f}-{loss_critic:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='loss_gen',
        mode='min',
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

    # Train the model
    trainer.fit(model, data_module)