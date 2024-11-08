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
import nibabel as nib
from monai.transforms import Compose, ScaleIntensity, Resize, EnsureChannelFirst
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Critic, Generator, initialize_weights

# Set up directories
root_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
data_dir = os.path.join(root_dir, '..', 'data')  # Move up one level to access data

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 1  # For large 3D data, batch size of 1 is reasonable
IMAGE_SIZE = 64  # Adjust as needed
CHANNELS_IMG = 1  # Single-channel images (grayscale)
NUM_EPOCHS = 500
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Custom Dataset class using nibabel directly
class NiftiDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for loading NIfTI images using nibabel.
    """
    def __init__(self, data, transforms=None):
        self.data = data  # List of dictionaries with 'VNC' and 'MIX' keys
        self.transforms = transforms  # Transforms to apply

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vnc_path = self.data[idx]["VNC"]
        mix_path = self.data[idx]["MIX"]

        # Load images using nibabel
        vnc_img = nib.load(vnc_path).get_fdata()
        mix_img = nib.load(mix_path).get_fdata()

        # Apply transforms if provided
        if self.transforms:
            vnc_img = self.transforms(vnc_img)
            mix_img = self.transforms(mix_img)

        # After transforms, images are MetaTensors with shape (C, D, H, W)
        # Ensure they are float tensors
        vnc_tensor = vnc_img.float()
        mix_tensor = mix_img.float()

        return vnc_tensor, mix_tensor

# Transforms for data processing
transforms = Compose([
    EnsureChannelFirst(),  # Adds channel dimension as the first dimension
    ScaleIntensity(),      # Scales intensity to [0, 1]
    Resize((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)),  # Resize to desired dimensions
])

# Dataset and DataLoader setup
cases = glob(os.path.join(data_dir, "*"))
train_data = []
for case in cases:
    vnc_paths = glob(os.path.join(case, "*_VNC.nii.gz"))
    mix_paths = glob(os.path.join(case, "*_MIX.nii.gz"))
    
    if vnc_paths and mix_paths:
        vnc_path = vnc_paths[0]
        mix_path = mix_paths[0]
        if os.path.exists(vnc_path) and os.path.exists(mix_path):
            train_data.append({"VNC": vnc_path, "MIX": mix_path})
            print("VNC path: {}, MIX: {}".format(vnc_path, mix_path))
        else:
            print("File not found: VNC or MIX in case: {}".format(case))
    else:
        print("Missing VNC or MIX file in case: {}".format(case))

# Check if training data is loaded
if not train_data:
    raise ValueError("No training data found. Please check your data directory.")

dataset = NiftiDataset(train_data, transforms)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0  # num_workers=0 to avoid multiprocessing issues
)

# Initialize models
gen = Generator(CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# Initialize optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# TensorBoard writers
writer_real = SummaryWriter("logs/3D_WGAN_GP/real")
writer_fake = SummaryWriter("logs/3D_WGAN_GP/fake")
step = 0

# Lists for tracking losses
gen_losses = []
critic_losses = []

gen.train()
critic.train()

# Training loop
for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader, leave=True)
    for batch_idx, (vnc, mix) in enumerate(loop):
        vnc = vnc.to(device)
        mix = mix.to(device)
        cur_batch_size = vnc.shape[0]

        # Train Critic
        for _ in range(CRITIC_ITERATIONS):
            fake = gen(vnc)
            critic_real = critic(mix).reshape(-1)
            critic_fake = critic(fake.detach()).reshape(-1)
            gp = gradient_penalty(critic, mix, fake.detach(), device=device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # Track critic loss
        critic_losses.append(loss_critic.item())

        # Train Generator
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Track generator loss
        gen_losses.append(loss_gen.item())

        # Update progress bar
        loop.set_description("Epoch [{}/{}]".format(epoch+1, NUM_EPOCHS))
        loop.set_postfix(loss_critic=loss_critic.item(), loss_gen=loss_gen.item())

        # Print and log progress periodically
        if batch_idx % 10 == 0:
            print(
                "Epoch [{}/{}] Batch {}/{} Loss D: {:.4f}, Loss G: {:.4f}".format(
                    epoch+1, NUM_EPOCHS, batch_idx, len(loader), loss_critic.item(), loss_gen.item()
                )
            )

            with torch.no_grad():
                # Generate fake image for visualization
                fake = gen(vnc[:1])
                # Select a slice in the middle for visualization
                slice_idx = fake.shape[2] // 2
                real_slice = mix[0, 0, slice_idx, :, :].cpu().numpy()
                fake_slice = fake[0, 0, slice_idx, :, :].cpu().numpy()

                # Log images to TensorBoard
                writer_real.add_image("Real", real_slice, global_step=step, dataformats='HW')
                writer_fake.add_image("Fake", fake_slice, global_step=step, dataformats='HW')

            step += 1

    # Save models and outputs periodically
    if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
        save_checkpoint(
            {'gen': gen.state_dict(), 'disc': critic.state_dict()},
            filename="checkpoint_epoch_{}.pth.tar".format(epoch+1)
        )
        torch.save(fake, "output_fake_epoch_{}.pt".format(epoch+1))

# Plot and save loss curves
plt.figure()
plt.plot(gen_losses, label="Generator Loss")
plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curves.png")
plt.close()