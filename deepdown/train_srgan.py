# Common imports
import argparse
import time
import numpy as np

# Import torch
from torch.utils.data import Dataset

# Utils
from deepdown.utils.data_loader import load_target_data, load_input_data
from deepdown.utils.loss_fcts import *
from deepdown.utils.data_generators import DataGenerator
from deepdown.utils.helpers import print_cuda_availability, DEVICE
from deepdown.models.srgan import Generator, Discriminator
from deepdown.config import Config

# Allowed arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")

# To make this notebook's output stable across runs
np.random.seed(42)

# Check if CUDA is available
print_cuda_availability()


def main(conf):
    # Data options
    date_start = conf.get('date_start', '1979-01-01')
    date_end = conf.get('date_end', '2021-12-31')
    years_train = conf.get('years_train', [1979, 2015])
    years_valid = conf.get('years_valid', [2015, 2018])
    years_test = conf.get('years_test', [2019, 2021])
    levels = conf.get('levels', [850, 1000])
    resol_low = conf.get('resol_low', 0.25)
    input_variables = conf.get('input_variables', ['tp', 't'])
    input_paths = [conf.get('path_era5_025') + '/precipitation',
                   conf.get('path_era5_025') + '/temperature']

    # Crop on a smaller region
    do_crop = conf.get('do_crop', False)
    crop_x = conf.get('crop_x', [2700000, 2760000])
    crop_y = conf.get('crop_y', [1190000, 1260000])

    # Hyperparameters
    lr = conf.get('lr', 0.0002)
    batch_size = conf.get('batch_size', 32)
    num_epochs = conf.get('num_epochs', 100)

    # Load target data
    target = load_target_data(date_start, date_end, conf.get('path_mch'),
                              path_tmp=conf.get('path_tmp'))

    # Extract the axes of the final target domain based on temperature 
    x_axis = target.TabsD.x
    y_axis = target.TabsD.y

    input_data = load_input_data(date_start, date_end, conf.get('pat_dem'),
                                 input_variables, input_paths,
                                 levels, resol_low, x_axis, y_axis,
                                 path_tmp=conf.get('path_tmp'))

    if do_crop:
        input_data = input_data.sel(x=slice(min(crop_x), max(crop_x)),
                                    y=slice(max(crop_y), min(crop_y)))
        target = target.sel(x=slice(min(crop_x), max(crop_x)),
                            y=slice(max(crop_y), min(crop_y)))

    # Split the data
    x_train = input_data.sel(time=slice(years_train[0], years_train[1]))
    x_valid = input_data.sel(time=slice(years_valid[0], years_valid[1]))
    x_test = input_data.sel(time=slice(years_test[0], years_test[1]))

    y_train = target.sel(time=slice(years_train[0], years_train[1]))
    y_valid = target.sel(time=slice(years_valid[0], years_valid[1]))
    y_test = target.sel(time=slice(years_test[0], years_test[1]))

    # Select the variables to use as input and output
    input_vars = {'topo': None, 'tp': None, 't': levels}
    output_vars = ['RhiresD', 'TabsD']  # ['RhiresD', 'TabsD', 'TmaxD', 'TminD']

    # Create the data generators
    training_set = DataGenerator(x_train, y_train, input_vars, output_vars)
    loader_train = torch.utils.data.DataLoader(training_set, batch_size=batch_size)
    valid_set = DataGenerator(x_valid, y_valid, input_vars, output_vars, shuffle=False,
                              mean=training_set.mean, std=training_set.std)
    loader_val = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    test_set = DataGenerator(x_test, y_test, input_vars, output_vars, shuffle=False,
                             mean=training_set.mean, std=training_set.std)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    # Check to make sure the range on the input and output images is correct,
    # and they're the correct shape
    test_x, test_y = training_set.__getitem__(3)
    print("x shape: ", test_x.shape)
    print("y shape: ", test_y.shape)
    print("x min: ", torch.min(test_x))
    print("x max: ", torch.max(test_x))
    print("y min: ", torch.min(test_y))
    print("y max: ",torch.max(test_y))
    torch.cuda.empty_cache()

    h, w = test_y.shape
    lowres_shape = test_x.shape
    num_channels_in = test_x.shape[0]
    num_channels_out = test_y.shape[0]

    D = Discriminator(num_channels=num_channels_out, H=h, W=w)
    G = Generator(num_channels_in, input_size=lowres_shape,
                  output_channels=num_channels_out)

    # Define optimizer for discriminator
    D_solver = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # Define optimizer for generator
    G_solver = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    print('train the SRGAN')
    train_srgan(loader_train, D, G, D_solver, G_solver, discriminator_loss,
                generator_loss, G_iters=1, show_every=250, num_epochs=num_epochs)


def train_srgan(loader_train, D, G, D_solver, G_solver, discriminator_loss,
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
    tic = time()
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
                toc = time()
                print(f'Epoch: {epoch}, Iter: {iter_count}, '
                      f'D: {d_total_error.item():.4}, G: {g_error.item():.4}, '
                      f'Time since last print (min): {(toc - tic) / 60:.4}')
                tic = time()
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


if __name__ == "__main__":
    args = argParser.parse_args()
    config = Config(args)
    # config.print()

    main(config.config)
