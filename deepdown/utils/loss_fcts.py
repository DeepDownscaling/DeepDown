import torch
import torch.nn as nn
from deepdown.utils.utils import DEVICE

def generator_loss(gen_img, true_img, logits_fake, weight_param=1e-3):
    """
    Computes the generator loss.

    Parameters
    ----------
    gen_img: PyTorch Tensor of shape (N, C, H, W)
        The generated images.
    true_img: PyTorch Tensor of shape (N, C, H, W)
        The true images.
    logits_fake: PyTorch Tensor of shape (N, )
        Tensor giving scores for the fake data.
    weight_param: float
        The weight to put on the adversarial loss. Default: 1e-3.

    Returns
    -------
    total_loss: PyTorch Tensor containing (scalar) the loss for the generator.
    content_loss: PyTorch Tensor containing (scalar) the content loss for the generator.
    adversarial_loss: PyTorch Tensor containing (scalar) the adversarial loss for the generator.
    """
    # Content loss - MSE loss for now. Ludig paper also suggests using
    # Euclidean distance between feature vector of true image and generated image, 
    # where we get the feature vector from a pretrained VGGnet. Probably wouldn't
    # work for us (at least pretrained) because climate data looks so different
    # from normal pictures
    content_loss_func = nn.MSELoss()
    content_loss = content_loss_func(gen_img, true_img)

    n = logits_fake.shape[0]
    desired_labels = torch.ones(n, 1).to(device=DEVICE, dtype=torch.float32)
    bce_Loss = nn.BCELoss()
    adversarial_loss = bce_Loss(logits_fake, desired_labels)

    total_loss = content_loss + weight_param * adversarial_loss

    return total_loss, content_loss, adversarial_loss


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    Adapted from homework 3 of CS231n at Stanford, GAN notebook
    
    Parameters
    ----------
    logits_real: PyTorch Tensor of shape (N, )
        Tensor giving scores for the real data.
    logits_fake: PyTorch Tensor of shape (N, )
        Tensor giving scores for the fake data.

    Returns
    -------
    loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    l1: PyTorch Tensor containing (scalar) the loss for the real data.
    l2: PyTorch Tensor containing (scalar) the loss for the fake data.
    """
    # How often it mistakes real images for fake
    n = logits_real.shape[0]
    real_labels = torch.ones(n, 1).to(device=DEVICE, dtype=torch.float32)
    bce_loss = nn.BCELoss()
    l1 = bce_loss(logits_real, real_labels)

    # How often it gets fooled into thinking fake images are real
    fake_labels = torch.zeros(n, 1).to(device=DEVICE, dtype=torch.float32)
    l2 = bce_loss(logits_fake, fake_labels)

    loss = (l1 + l2)

    return loss, l1, l2
