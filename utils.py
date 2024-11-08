import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device="cpu"):
    """
    Calculate the gradient penalty for WGAN-GP.

    Args:
        critic (nn.Module): The critic (discriminator) network.
        real (torch.Tensor): Batch of real images (N, C, D, H, W).
        fake (torch.Tensor): Batch of generated images from the generator (N, C, D, H, W).
        device (str): Device to perform calculations on ("cpu" or "cuda").

    Returns:
        torch.Tensor: Scalar tensor representing the gradient penalty.
    """
    # Get the batch size and dimensions from the real images
    BATCH_SIZE, C, D, H, W = real.shape  # For 3D data

    # Generate random interpolation factor alpha
    # Shape: (BATCH_SIZE, 1, 1, 1, 1) to broadcast over C, D, H, W
    alpha = torch.rand(BATCH_SIZE, 1, 1, 1, 1, device=device)

    # Create interpolated images between real and fake images
    # interpolated_images will have the same shape as real and fake images
    interpolated_images = alpha * real + (1 - alpha) * fake
    interpolated_images.requires_grad_(True)  # Enable gradient computation

    # Pass interpolated images through the critic
    mixed_scores = critic(interpolated_images)  # Output shape depends on critic's output

    # Create a tensor of ones with the same shape as mixed_scores for gradient calculation
    grad_outputs = torch.ones_like(mixed_scores, device=device)

    # Compute gradients of the critic's scores with respect to interpolated images
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=grad_outputs,
        create_graph=True,  # Retain computation graph for higher-order derivative
        retain_graph=True,
    )[0]  # Extract the gradients (tuple output)

    # Reshape gradients to (BATCH_SIZE, -1) to compute the norm
    gradients = gradients.view(BATCH_SIZE, -1)

    # Compute the L2 norm of the gradients for each sample in the batch
    gradient_norm = gradients.norm(2, dim=1)  # Shape: (BATCH_SIZE,)

    # Compute the gradient penalty using the formula (||gradient||_2 - 1)^2
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    """
    Save the current model checkpoint.

    Args:
        state (dict): Dictionary containing model and optimizer states.
        filename (str): Filename for saving the checkpoint.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gen, disc):
    """
    Load a model checkpoint into the generator and critic.

    Args:
        checkpoint (dict): Loaded checkpoint containing model states.
        gen (nn.Module): Generator model to load the state into.
        disc (nn.Module): Critic model to load the state into.
    """
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
