import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device="cpu"):
    """
    Calculate the gradient penalty for WGAN-GP.
    """
    BATCH_SIZE, C, D, H, W = real.shape  # For 3D data
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1, 1), device=device)
    alpha = alpha.expand_as(real)
    interpolated_images = alpha * real + (1 - alpha) * fake

    # Calculate critic scores for interpolated images
    mixed_scores = critic(interpolated_images)

    # Compute gradients
    gradient = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Reshape gradients for norm calculation
    gradient = gradient.view(BATCH_SIZE, -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    """
    Save the current model checkpoint.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gen, disc):
    """
    Load a model checkpoint into the generator and critic.
    """
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
