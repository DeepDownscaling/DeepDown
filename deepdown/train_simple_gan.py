# Common imports
import argparse

# Import torch
from torch.utils.data import Dataset

# Utils
from .utils.data_loader import *
from .utils.utils_loss import *
from .utils.data_generators import *
from .utils.helpers import print_cuda_availability
from .models.SRGAN import *
from deepdown.config import Config


argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")

print_cuda_availability()


def main(conf):
    # Paths
    PATH_DEM = config['PATH_DEM']
    PATH_ERA5_025 = config['PATH_ERA5_025']  # Original ERA5 0.25°
    PATH_ERA5_100 = config['PATH_ERA5_100']  # ERA5 1°
    PATH_MCH = config['PATH_MeteoSwiss']

    NUM_CHANNELS_IN = config['NUM_CHANNELS_IN']
    NUM_CHANNELS_OUT = config['NUM_CHANNELS_OUT']
    lr = config['lr']

    # Data options
    DATE_START = config['DATE_START']
    DATE_END = config['DATE_END']
    YY_TRAIN = config['YY_TRAIN']
    YY_TEST = config['YY_TEST']
    LEVELS = config['LEVELS']
    RESOL_LOW = config['RESOL_LOW']
    INPUT_VARIABLES = config['INPUT_VARIABLES']
    INPUT_PATHS = [PATH_ERA5_025 + '/precipitation', PATH_ERA5_025 + '/temperature']
    DUMP_DATA_TO_PICKLE = config['DUMP_DATA_TO_PICKLE']

    # Crop on a smaller region
    DO_CROP = config['DO_CROP']
    # I reduce the area of crop now, to avoid NA
    CROP_X = [2700000, 2760000]  # with NAN: [2720000, 2770000]
    CROP_Y = [1190000, 1260000]  # with NAN: [1290000, 1320000]

    device = config['device']
    dtype = config['dtype']
    # Hyperparameters
    BATCH_SIZE = 32
    h, w = config['lowres_shape']

    # Load target data
    target = load_target_data(DATE_START, DATE_END, PATH_MCH)

    # Extract the axes of the final target domain based on temperature 
    x_axis = target.TabsD.x
    y_axis = target.TabsD.y

    input_data = load_input_data(DATE_START, DATE_END, PATH_DEM, INPUT_VARIABLES,
                                 INPUT_PATHS,
                                 LEVELS, RESOL_LOW, x_axis, y_axis)

    if DO_CROP:
        input_data = input_data.sel(x=slice(min(CROP_X), max(CROP_X)),
                                    y=slice(max(CROP_Y), min(CROP_Y)))
        target = target.sel(x=slice(min(CROP_X), max(CROP_X)),
                            y=slice(max(CROP_Y), min(CROP_Y)))

    # Split the data
    x_train = input_data.sel(time=slice('1999', '2011'))
    x_valid = input_data.sel(time=slice('2012', '2015'))
    x_test = input_data.sel(time=slice('2016', '2021'))

    y_train = target.sel(time=slice('1999', '2011'))
    y_valid = target.sel(time=slice('2012', '2005'))
    y_test = target.sel(time=slice('2006', '2011'))

    # Select the variables to use as input and output
    input_vars = {'topo': None, 'tp': None, 't': LEVELS}
    output_vars = ['RhiresD', 'TabsD']  # ['RhiresD', 'TabsD', 'TmaxD', 'TminD']

    training_set = DataGenerator(x_train, y_train, input_vars, output_vars)
    loader_train = torch.utils.data.DataLoader(training_set, batch_size=32)

    # Validation
    valid_set = DataGenerator(x_valid, y_valid, input_vars, output_vars, shuffle=False,
                              mean=training_set.mean, std=training_set.std)
    loader_val = torch.utils.data.DataLoader(valid_set, batch_size=32)

    # Test
    test_set = DataGenerator(x_test, y_test, input_vars, output_vars, shuffle=False,
                             mean=training_set.mean, std=training_set.std)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=32)

    # Check to make sure the range on the input and output images is correct, and they're the correct shape
    testx, testy = training_set.__getitem__(3)
    print("x shape: ", testx.shape)
    print("y shape: ", testy.shape)
    torch.cuda.empty_cache()

    D = Discriminator(num_channels=NUM_CHANNELS_OUT, H=h, W=w)
    G = Generator(NUM_CHANNELS_IN, input_size=config['lowres_shape'],
                  output_channels=NUM_CHANNELS_OUT)

    # Define optimizer for discriminator
    D_solver = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # Define optimizer for generator
    G_solver = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    num_epochs = config['num_epochs']
    G_iters = config['G_iters']
    dtype = torch.float32

    print('train the SRGAN')
    train_srgan(loader_train, D, G, D_solver, G_solver, discriminator_loss,
                generator_loss, device, dtype, G_iters=1, show_every=250, num_epochs=5)


def train_srgan(loader_train, D, G, D_solver, G_solver, discriminator_loss,
                generator_loss, device, dtype, G_iters=1, show_every=250, num_epochs=5):
    """
    Adapted from CS231

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    # Move the models to the correct device (GPU if GPU is available)
    D = D.to(device=device)
    G = G.to(device=device)

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
            high_res_imgs = y.to(device=device, dtype=dtype)
            logits_real = D(high_res_imgs)

            x.requires_grad_()
            low_res_imgs = x.to(device=device, dtype=dtype)
            fake_images = G(low_res_imgs)
            logits_fake = D(fake_images)

            # Update for the discriminator
            # d_total_error, D_real_L[iter_count], D_fake_L[iter_count] = discriminator_with_Nan_loss(logits_real, logits_fake)
            d_total_error, D_real_L[iter_count], D_fake_L[
                iter_count] = discriminator_loss(logits_real, logits_fake)
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
                g_error, G_content[G_iter_count], G_advers[
                    G_iter_count] = generator_loss(fake_images, high_res_imgs,
                                                   gen_logits_fake,
                                                   weight_param=weight_param)
                # g_error, G_content[G_iter_count], G_advers[G_iter_count] = generator_withNan_loss(fake_images, high_res_imgs, gen_logits_fake, weight_param=weight_param)

                G_solver.zero_grad()
                g_error.backward()
                G_solver.step()
                G_iter_count += 1

            if (iter_count % show_every == 0):
                toc = time()
                print(
                    'Epoch: {}, Iter: {}, D: {:.4}, G: {:.4}, Time since last print (min): {:.4}'.format(
                        epoch, iter_count, d_total_error.item(), g_error.item(),
                        (toc - tic) / 60))
                tic = time()
                # plot_epoch(x, fake_images, y)
                # plot_loss(G_content, G_advers, D_real_L, D_fake_L, weight_param)
                print()
            iter_count += 1

        torch.save(D.cpu().state_dict(),
                   'GAN_Discriminator_checkpoint_adversWP_1e-1.pt')
        torch.save(G.cpu().state_dict(), 'GAN_Generator_checkpoint_adversWP_1e-1.pt')

        D = D.to(device=device)
        G = G.to(device=device)
        # Put models in training mode
        D.train()
        G.train()


if __name__ == "__main__":
    args = argParser.parse_args()
    config = Config(args)
    config.print()

    main(config.config)
