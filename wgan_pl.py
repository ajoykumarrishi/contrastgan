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
from monai.networks.nets import DenseNet
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
from monai.data import CacheDataset, DataLoader
from monai.config import print_config

print_config()

# CHANGED: Updated hyperparameters
hparams = {
    "learning_rate": 1e-4,
    "batch_size": 2,  # CHANGED: Updated from 1 to 2
    "image_size": 64,
    "channels_img": 1,
    "num_epochs": 500,
    "features_gen": (16, 32, 64, 128, 256),
    "critic_iterations": 5,
    "lambda_gp": 10,
    "beta1": 0.0,
    "beta2": 0.9,
    "num_workers": 0,
    "save_image_interval": 50,  # NEW: Interval for saving generated images
}

# CHANGED: Updated dataset class to handle both training and validation data
class NiftiDataset:
    def __init__(self, data_dir, image_size, is_validation=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_validation = is_validation  # NEW: Flag for validation dataset
        self.data_dicts = []
        
        cases = glob(os.path.join(self.data_dir, "*"))
        for case in cases:
            vnc_paths = glob(os.path.join(case, "*_VNC.nii.gz"))
            mix_paths = glob(os.path.join(case, "*_MIX.nii.gz"))
            if vnc_paths and mix_paths:
                vnc_path, mix_path = vnc_paths[0], mix_paths[0]
                if os.path.exists(vnc_path) and os.path.exists(mix_path):
                    self.data_dicts.append({"VNC": vnc_path, "MIX": mix_path})
                else:
                    print(f"File not found: VNC or MIX in case: {case}")
            else:
                print(f"Missing VNC or MIX file in case: {case}")
                
        if not self.data_dicts:
            raise ValueError(f"No {'validation' if is_validation else 'training'} data found. Please check your data directory.")

        # CHANGED: Different transforms for validation (no augmentation)
        if is_validation:
            self.transforms = Compose([
                LoadImaged(keys=["VNC", "MIX"]),
                EnsureChannelFirstd(keys=["VNC", "MIX"]),
                ScaleIntensityd(keys=["VNC", "MIX"]),
                SpatialPadd(keys=["VNC", "MIX"], 
                           spatial_size=(self.image_size, self.image_size, self.image_size)),
                EnsureTyped(keys=["VNC", "MIX"]),
            ])
        else:
            self.transforms = Compose([
                LoadImaged(keys=["VNC", "MIX"]),
                EnsureChannelFirstd(keys=["VNC", "MIX"]),
                ScaleIntensityd(keys=["VNC", "MIX"]),
                RandSpatialCropd(
                    keys=["VNC", "MIX"],
                    roi_size=(self.image_size, self.image_size, self.image_size),
                    random_size=False,
                    random_center=True
                ),
                SpatialPadd(keys=["VNC", "MIX"], 
                           spatial_size=(self.image_size, self.image_size, self.image_size)),
                RandRotate90d(keys=["VNC", "MIX"], prob=0.5, spatial_axes=(0, 1)),
                EnsureTyped(keys=["VNC", "MIX"]),
            ])

    def get_cache_dataset(self):
        return CacheDataset(
            data=self.data_dicts, 
            transform=self.transforms, 
            cache_rate=1.0, 
            num_workers=hparams["num_workers"]
        )

# CHANGED: Updated WGAN_GP class with validation and image saving
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
            out_channels=1,
            init_features=32,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
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
        
        # Train Critic
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
            torch.cuda.empty_cache()
            
        self.log("loss_critic", loss_critic, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Train Generator
        fake = self.generator(vnc)
        gen_loss = -torch.mean(self.critic(fake))
        recon_loss = nn.functional.mse_loss(fake, mix)
        total_gen_loss = gen_loss + recon_loss
        
        opt_gen.zero_grad()
        self.manual_backward(total_gen_loss)
        opt_gen.step()
        
        self.log("loss_gen", gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("gradient_penalty", gp, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    # NEW: Validation step
    def validation_step(self, batch, batch_idx):
        vnc, mix = batch["VNC"].float(), batch["MIX"].float()
        fake = self.generator(vnc)
        val_recon_loss = nn.functional.mse_loss(fake, mix)
        self.log("val_recon_loss", val_recon_loss, on_epoch=True)

        # Save validation images
        if batch_idx == 0 and self.current_epoch % self.hparams["save_image_interval"] == 0:
            self.save_validation_image(vnc[0], mix[0], fake[0], self.current_epoch)

    # NEW: Image saving method
    def save_validation_image(self, vnc, mix, fake, epoch):
        # Create directory if it doesn't exist
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
        os.makedirs(save_dir, exist_ok=True)

        # Take central slices
        slice_idx = vnc.shape[-1] // 2
        vnc_slice = vnc[0, :, :, slice_idx].cpu().numpy()
        mix_slice = mix[0, :, :, slice_idx].cpu().numpy()
        fake_slice = fake[0, :, :, slice_idx].detach().cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(vnc_slice, cmap='gray')
        axes[0].set_title('VNC')
        axes[1].imshow(mix_slice, cmap='gray')
        axes[1].set_title('MIX (Target)')
        axes[2].imshow(fake_slice, cmap='gray')
        axes[2].set_title('Generated')
        
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_validation.png'))
        plt.close()

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

    # NEW: Save models at the end of training
    def on_train_end(self):
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.generator.state_dict(), 
                  os.path.join(save_dir, "generator_final.pth"))
        torch.save(self.critic.state_dict(), 
                  os.path.join(save_dir, "discriminator_final.pth"))


# CHANGED: Updated main execution
if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(root_dir)
    
    # Setup training dataset
    train_dir = os.path.join(parent_dir, "training")
    train_dataset = NiftiDataset(
        data_dir=train_dir,
        image_size=hparams["image_size"],
        is_validation=False
    )
    train_loader = DataLoader(
        train_dataset.get_cache_dataset(),
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
    )
    
    # Setup validation dataset
    val_dir = os.path.join(parent_dir, "validation")
    val_dataset = NiftiDataset(
        data_dir=val_dir,
        image_size=hparams["image_size"],
        is_validation=True
    )
    val_loader = DataLoader(
        val_dataset.get_cache_dataset(),
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
    )

    model = WGAN_GP(hparams)

    # Setup logging and checkpoints
    logger = TensorBoardLogger("tb_logs", name="WGAN_GP")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="WGAN_GP-{epoch:02d}-{loss_gen:.2f}-{loss_critic:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="loss_gen",
        mode="min",
    )

    # Setup training device
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

    # Start training
    trainer.fit(model, train_loader, val_loader)