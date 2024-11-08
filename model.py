import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act

# Critic class (Discriminator)
class Critic(nn.Module):
    """
    Critic class for WGAN, designed to process 3D volumes of size 64x64x64.
    """
    def __init__(self, channels_img, features_d):
        """
        Initializes the Critic model.

        Args:
            channels_img (int): Number of channels in the input images.
            features_d (int): Base number of features in the Critic.
        """
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            # Input: N x channels_img x 64 x 64 x 64
            self._block(channels_img, features_d, stride=2),       # -> 32x32x32
            self._block(features_d, features_d * 2, stride=2),     # -> 16x16x16
            self._block(features_d * 2, features_d * 4, stride=2), # -> 8x8x8
            self._block(features_d * 4, features_d * 8, stride=2), # -> 4x4x4
            nn.Conv3d(features_d * 8, 1, kernel_size=4),           # -> 1x1x1
        )

    def _block(self, in_channels, out_channels, stride):
        """
        Creates a convolutional block with Conv3d, InstanceNorm3d, and LeakyReLU layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution.
        """
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, padding=1, bias=False
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Forward pass of the Critic.

        Args:
            x (Tensor): Input tensor of shape (N, channels_img, 64, 64, 64).

        Returns:
            Tensor: Output tensor representing the critic's score.
        """
        return self.model(x).view(-1)

# Generator class using 3D U-Net architecture
class Generator(nn.Module):
    """
    Generator class for WGAN, implemented as a 3D U-Net.

    The U-Net architecture consists of an encoder and decoder path with skip connections.
    It takes a 3D volume as input and outputs a 3D volume of the same size.
    """
    def __init__(self, channels_img, features_g):
        """
        Initializes the Generator model.

        Args:
            channels_img (int): Number of channels in the input and output images.
            features_g (int): Base number of features in the Generator.
        """
        super(Generator, self).__init__()
        # Define the 3D U-Net architecture from MONAI
        self.unet = UNet(
            spatial_dims=3,
            in_channels=channels_img,
            out_channels=channels_img,
            channels=(features_g, features_g * 2, features_g * 4, features_g * 8),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
        )

    def forward(self, x):
        """
        Forward pass of the Generator.

        Args:
            x (Tensor): Input tensor of shape (N, channels_img, 64, 64, 64).

        Returns:
            Tensor: Output tensor of shape (N, channels_img, 64, 64, 64).
        """
        return self.unet(x)

def initialize_weights(model):
    """
    Initializes weights of the model using a normal distribution.

    Args:
        model (nn.Module): The model to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0)