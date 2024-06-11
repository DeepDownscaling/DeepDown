import matplotlib.pyplot as plt
import numpy as np


# Helper functions for plotting
def plot_epoch(x, y_pred, y):
    """
    Plots the input, output and true precipitation fields.

    Parameters
    ----------
    x: torch.Tensor
        Input precipitation field
    y_pred: torch.Tensor
        Output precipitation field
    y: torch.Tensor
        True precipitation field
    """
    figsize = (9, 4)
    plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.imshow(x[0, 0, :, :].cpu().detach().numpy())
    plt.title("Input Precip")
    plt.subplot(1, 3, 2)
    plt.imshow(y_pred[0, 0, :, :].cpu().detach().numpy())
    plt.title("Output Precip")
    plt.subplot(1, 3, 3)
    plt.imshow(y[0, 0, :, :].cpu().detach().numpy())
    plt.title("True Precip")


def plot_loss(G_content, G_advers, D_real_L, D_fake_L, weight_param):
    """
    Plots the generator and discriminator loss.

    Parameters
    ----------
    G_content: np.array
        Generator content loss
    G_advers: np.array
        Generator adversarial loss
    D_real_L: np.array
        Discriminator loss for real images
    D_fake_L: np.array
        Discriminator loss for fake images
    weight_param: float
        Weighting put on adversarial loss
    """
    D_count = np.count_nonzero(D_real_L)
    G_count = np.count_nonzero(G_content)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(G_count), G_content[range(G_count)])
    plt.plot(range(G_count), G_advers[range(G_count)])
    plt.plot(range(G_count),
             G_content[range(G_count)] + weight_param * G_advers[range(G_count)])
    plt.legend(("Content", "Adversarial", "Total"))
    plt.title("Generator loss")
    plt.xlabel("Iteration")

    plt.subplot(1, 2, 2)
    plt.plot(range(D_count), D_real_L[range(D_count)])
    plt.plot(range(D_count), D_fake_L[range(D_count)])
    plt.plot(range(D_count), D_real_L[range(D_count)] + D_fake_L[range(D_count)])
    plt.legend(("Real Pic", "Fake Pic", "Total"))
    plt.title("Discriminator loss")
    plt.xlabel("Iteration")
    plt.show()
