import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


class MeanSquaredErrorNans(nn.Module):
    def __init__(self):
        super(MeanSquaredErrorNans, self).__init__()

    def forward(self, y_true, y_pred):
        nb_values = torch.where(torch.isnan(y_true),
                                torch.zeros_like(y_true),
                                torch.ones_like(y_true))
        nb_values = torch.sum(nb_values)
        y_true = torch.where(torch.isnan(y_true), y_pred, y_true)
        loss = torch.square(y_pred - y_true)
        loss_sum = torch.sum(loss)
        return loss_sum / nb_values


def generator_loss(gen_img, true_img, logits_fake, weight_param=1e-3):
    """
    Computes the generator loss described above.

    Inputs:
    - gen_img: (PyTorch tensor) shape N, C image generated by the Generator, so that we can calculate MSE
    - true_img: (PyTorch tensor) the true, high res image, so that we can calculate the MSE
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    - weight_param: how much to weight the adversarial loss by when summing the losses. Default in Ledig paper is 1e-3
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    # Content loss - MSE loss for now. Ludig paper also suggests using
    # Euclidean distance between feature vector of true image and generated image, 
    # where we get the feature vector from a pretrained VGGnet. Probably wouldn't
    # work for us (at least pretrained) because climate data looks so different from normal pictures
    content_loss_func = nn.MSELoss()
    content_loss = content_loss_func(gen_img, true_img)

    N = logits_fake.shape[0]
    desired_labels = torch.ones(N, 1).to(device=device, dtype=dtype)
    BCE_Loss = nn.BCELoss()
    adversarial_loss = BCE_Loss(logits_fake, desired_labels)

    total_loss = content_loss + weight_param * adversarial_loss
    #     print("Total loss: ", total_loss.cpu().detach().numpy())
    #     print("content loss: ", content_loss.cpu().detach().numpy())
    #     print("adversarial loss: ", adversarial_loss.cpu().detach().numpy())
    return total_loss, content_loss, adversarial_loss


def discriminator_loss(logits_real, logits_fake):
    """
    
    Adapted from homework 3 of CS231n at Stanford, GAN notebook
    
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data (real numbers). 
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data (real numbers).
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    # How often it mistakes real images for fake
    N = logits_real.shape[0]
    real_labels = torch.ones(N, 1).to(device=device, dtype=dtype)
    BCE_Loss = nn.BCELoss()
    L1 = BCE_Loss(logits_real, real_labels)

    # How often it gets fooled into thinking fake images are real
    fake_labels = torch.zeros(N, 1).to(device=device, dtype=dtype)
    L2 = BCE_Loss(logits_fake, fake_labels)

    #     print("L1 (how bad on real data): %f\t L2 (how bad on fake data): %f" % (L1, L2))

    loss = (L1 + L2)
    return loss, L1, L2


def generator_withNan_loss(gen_img, true_img, logits_fake, weight_param=1e-3):
    """
    Computes the generator loss described above.

    Inputs:
    - gen_img: (PyTorch tensor) shape N, C image generated by the Generator, so that we can calculate MSE
    - true_img: (PyTorch tensor) the true, high res image, so that we can calculate the MSE
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    - weight_param: how much to weight the adversarial loss by when summing the losses. Default in Ledig paper is 1e-3
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    #
    if torch.isnan(gen_img).any() or torch.isnan(true_img).any() or torch.isnan(
            logits_fake).any():
        # Handle NaN values here
        # Replace NaN values in gen_img and true_img with zeros
        gen_img = torch.where(torch.isnan(gen_img), torch.zeros_like(gen_img), gen_img)
        true_img = torch.where(torch.isnan(true_img), torch.zeros_like(true_img),
                               true_img)

    # Content loss - MSE loss
    content_loss_func = nn.MSELoss()
    content_loss = content_loss_func(gen_img, true_img)

    N = logits_fake.shape[0]
    desired_labels = torch.ones(N, 1).to(device=device, dtype=dtype)
    BCE_Loss = nn.BCELoss()
    adversarial_loss = BCE_Loss(logits_fake, desired_labels)

    total_loss = content_loss + weight_param * adversarial_loss

    return total_loss, content_loss, adversarial_loss


def discriminator_with_Nan_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data (real numbers). 
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data (real numbers).
    
    Returns:
    - loss: PyTorch Tensor containing the loss for the discriminator.
    """
    # Handle NaN values
    if torch.isnan(logits_real).any() or torch.isnan(logits_fake).any():
        # Replace NaN values with zeros
        logits_real = torch.where(torch.isnan(logits_real),
                                  torch.zeros_like(logits_real), logits_real)
        logits_fake = torch.where(torch.isnan(logits_fake),
                                  torch.zeros_like(logits_fake), logits_fake)

    N = logits_real.shape[0]
    real_labels = torch.ones(N, 1).to(device=logits_real.device,
                                      dtype=logits_real.dtype)
    fake_labels = torch.zeros(N, 1).to(device=logits_fake.device,
                                       dtype=logits_fake.dtype)

    BCE_Loss = nn.BCELoss()
    L1 = BCE_Loss(logits_real, real_labels)
    L2 = BCE_Loss(logits_fake, fake_labels)

    loss = L1 + L2
    return loss, L1, L2
