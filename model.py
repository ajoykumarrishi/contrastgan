import torch
import torch.nn as nn

# Critic class (also known as Discriminator in traditional GANs)
class Critic(nn.Module):  # Renamed to Critic for WGAN
    def __init__(self, channels_img, features_d):
        """
        Initializes the Critic model.

        Args:
            channels_img (int): Number of channels in the input images (e.g., 1 for grayscale MRI scans).
            features_d (int): Base number of features (filters) in the critic.
        """
        super(Critic, self).__init__()
        # Define the Critic network using a sequential container
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64 x 64
            nn.Conv3d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            # LeakyReLU activation function with negative slope of 0.2
            nn.LeakyReLU(0.2),
            # Add convolutional blocks with increasing number of features
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # Final convolution to reduce to a single value
            nn.Conv3d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # Note: For WGAN, we do not apply a Sigmoid activation at the end
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a convolutional block with Conv3d, InstanceNorm3d, and LeakyReLU layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to all sides of the input.
        """
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # Instance normalization for stable training
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Forward pass of the Critic.

        Args:
            x (Tensor): Input tensor of shape (N, channels_img, D, H, W).

        Returns:
            Tensor: Output tensor of shape (N, 1, 1, 1, 1), representing the critic's score.
        """
        return self.disc(x)

# Generator class
class Generator(nn.Module):
    def __init__(self, channels_img, features_g):
        """
        Initializes the Generator model.

        Args:
            channels_img (int): Number of channels in the input images.
            features_g (int): Base number of features (filters) in the generator.
        """
        super(Generator, self).__init__()
        # Define the Generator network using a sequential container
        self.net = nn.Sequential(
            # Input: N x channels_img x initial size (e.g., 4x4x4)
            self._block(channels_img, features_g * 8, 4, 2, 1),   # Output: size doubles
            self._block(features_g * 8, features_g * 4, 4, 2, 1), # Output: size doubles
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # Output: size doubles
            self._block(features_g * 2, features_g, 4, 2, 1),     # Output: size doubles
            # Final transposed convolution to get desired output size
            nn.ConvTranspose3d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1
            ),  # Output: full volume size
            # Tanh activation to get output values between -1 and 1
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a transposed convolutional block with ConvTranspose3d, BatchNorm3d, and ReLU layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the transposed convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to all sides of the input.
        """
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # Batch normalization for stable training
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass of the Generator.

        Args:
            x (Tensor): Input tensor (image patch) of shape (N, channels_img, D, H, W).

        Returns:
            Tensor: Generated volumetric data of shape (N, channels_img, D_out, H_out, W_out).
        """
        return self.net(x)

def initialize_weights(model):
    """
    Initializes weights of the model using a normal distribution as per DCGAN guidelines.

    Args:
        model (nn.Module): The model to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)