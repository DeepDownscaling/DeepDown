# Import torch
import numpy as np
import torch
import time

from deepdown.utils.loss_fcts import *
from deepdown.utils.utils import DEVICE


def srgan_train(loader_train, D, G, D_solver, G_solver, discriminator_loss,
                generator_loss, G_iters=1, show_every=250, num_epochs=100):
    """
    Adapted from https://github.com/mantariksh/231n_downscaling/blob/master/SRGAN.ipynb

    Parameters
    ----------
    loader_train: DataLoader
        DataLoader for the training set
    D: Discriminator
        PyTorch Discriminator model
    G: Generator
        PyTorch Generator model
    D_solver: torch.optim
        Optimizer for the discriminator
    G_solver: torch.optim
        Optimizer for the generator
    discriminator_loss: function
        Function to compute the discriminator loss
    generator_loss: function
        Function to compute the generator loss
    G_iters: int
        Number of times to update the generator
    show_every: int
        Number of iterations to print the loss
    num_epochs: int
        Number of epochs to train the model
    """
    # Move the models to the correct device (GPU if GPU is available)
    D = D.to(device=DEVICE)
    G = G.to(device=DEVICE)

    # Put models in training mode
    D.train()
    G.train()

    G_content = np.zeros(len(loader_train) * num_epochs * G_iters + 1)
    G_advers = np.zeros(len(loader_train) * num_epochs * G_iters + 1)
    D_real_L = np.zeros(len(loader_train) * num_epochs + 1)
    D_fake_L = np.zeros(len(loader_train) * num_epochs + 1)

    iter_count = 0
    G_iter_count = 0
    tic = time.time()
    for epoch in range(num_epochs):

        for x, y in loader_train:
            high_res_imgs = y.to(device=DEVICE, dtype=torch.float32)
            logits_real = D(high_res_imgs)

            x.requires_grad_()
            low_res_imgs = x.to(device=DEVICE, dtype=torch.float32)
            fake_images = G(low_res_imgs)
            logits_fake = D(fake_images)

            # Update for the discriminator
            d_total_error, D_real_L[iter_count], D_fake_L[iter_count] = (
                discriminator_loss(logits_real, logits_fake))
            print('d_total_error:', d_total_error)
            print('D_real_L[iter_count]:', D_real_L[iter_count])
            print('D_fake_L[iter_count]:', D_fake_L[iter_count])
            D_solver.zero_grad()
            d_total_error.backward()
            D_solver.step()

            for i in range(G_iters):
                # Update for the generator
                fake_images = G(low_res_imgs)
                logits_fake = D(fake_images)
                gen_logits_fake = D(fake_images)
                weight_param = 1e-1  # Weighting put on adversarial loss
                g_error, G_content[G_iter_count], G_advers[G_iter_count] = (
                    generator_loss(fake_images, high_res_imgs, gen_logits_fake,
                                   weight_param=weight_param))

                G_solver.zero_grad()
                g_error.backward()
                G_solver.step()
                G_iter_count += 1

            if (iter_count % show_every == 0):
                toc = time.time()
                print(f'Epoch: {epoch}, Iter: {iter_count}, '
                      f'D: {d_total_error.item():.4}, G: {g_error.item():.4}, '
                      f'Time since last print (min): {(toc - tic) / 60:.4}')
                tic = time.time()
                # plot_epoch(x, fake_images, y)
                # plot_loss(G_content, G_advers, D_real_L, D_fake_L, weight_param)
                print()
            iter_count += 1

        torch.save(D.cpu().state_dict(), 'GAN_D_checkpoint.pt')
        torch.save(G.cpu().state_dict(), 'GAN_G_checkpoint.pt')

        D = D.to(device=DEVICE)
        G = G.to(device=DEVICE)

        # Put models in training mode
        D.train()
        G.train()
