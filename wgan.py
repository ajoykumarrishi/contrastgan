import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from glob import glob
from monai.data import Dataset
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, Compose, ToTensor
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Critic, Generator, initialize_weights

# Set up directories
root_dir = os.path.dirname(os.path.abspath(__file__))  # StyleGAN repo directory
data_dir = os.path.join(root_dir, '..', 'data')  # Move up one level to access data

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Transforms for data processing
transforms = Compose([
    LoadImage(image_only=True, reader="NiababelReader"),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)),
    ToTensor(),
])

# Dataset and dataloader setup
cases = glob(os.path.join(data_dir, "*"))
train_data = []
for case in cases:
    vnc_path = glob(os.path.join(case, "*_VNC.nii.gz"))[0]
    mix_path = glob(os.path.join(case, "*_MIX.nii.gz"))[0]
    if os.path.exists(vnc_path) and os.path.exists(mix_path):
        train_data.append({"VNC": vnc_path, "MIX": mix_path})
        print(f"VNC path: {vnc_path}, MIX: {mix_path}")

dataset = Dataset(train_data, transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models
gen = Generator(CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# Initialize optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# TensorBoard writers
writer_real = SummaryWriter(f"logs/3D_WGAN_GP/real")
writer_fake = SummaryWriter(f"logs/3D_WGAN_GP/fake")
step = 0

# Lists for tracking losses
gen_losses = []
critic_losses = []

gen.train()
critic.train()

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (vnc, mix) in enumerate(tqdm(loader)):
        vnc = vnc.to(device)
        mix = mix.to(device)
        cur_batch_size = vnc.shape[0]

        # Train Critic
        for _ in range(CRITIC_ITERATIONS):
            fake = gen(vnc)
            critic_real = critic(mix).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, mix, fake, device=device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Track critic loss
        critic_losses.append(loss_critic.item())

        # Train Generator
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Track generator loss
        gen_losses.append(loss_gen.item())

        # Print and log progress
        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} "
                f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(vnc[:1])
                img_grid_real = torchvision.utils.make_grid(mix[:1], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

    # Save models and outputs periodically
    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        save_checkpoint({'gen': gen.state_dict(), 'disc': critic.state_dict()}, filename=f"checkpoint_epoch_{epoch}.pth.tar")
        torch.save(fake, f"output_fake_epoch_{epoch}.pt")

# Plot and save loss curves
plt.figure()
plt.plot(gen_losses, label="Generator Loss")
plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curves.png")
plt.close()
