import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_cuda_availability():
    """Prints whether cuda is available and the device being used."""
    print("Cuda available :", torch.cuda.is_available())
    print(DEVICE)


def test_discriminator(training_set, discriminator, num_channels_out, h, w):
    """
    Test the discriminator model with a sample from the training set.

    Parameters
    ----------
    training_set: DataLoader
        DataLoader for the training set
    discriminator: class
        PyTorch Discriminator model
    num_channels_out: int
        Number of output channels
    h: int
        Height of the image
    w: int
        Width of the image
    """
    x, y = (training_set.__getitem__(3))
    y = y.unsqueeze(0)
    print("y: ", y.shape)
    model = discriminator(num_channels=num_channels_out, H=h, W=w)
    output = model(y)
    print(output.size())
    print(output)


def check_generator_accuracy(loader, model):
    # credits to: https://github.com/mantariksh/231n_downscaling/blob/master/SRGAN.ipynb

    model.eval()  # set model to evaluation mode
    count, rmse_precip_ypred, rmse_precip_x = 0, 0, 0
    # rmse_temp_ypred, rmse_temp_x = 0, 0
    with torch.no_grad():
        for x, y in loader:
            model = model.to(device=DEVICE)
            y = y.to(device=DEVICE, dtype=dtype)

            # Normalize x to be in -1 to 1 for purpose of comparing with high res data in same range
            # Turn it into a numpy array
            x_np = x.numpy()
            x_min = np.amin(x_np, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
            x_max = np.amax(x_np, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
            is_nan = np.int((x_min == x_max).any())
            eps = 1e-9
            x_norm_np = (x_np - x_min) / ((x_max - x_min + is_nan * eps) / 2) - 1

            x_norm = torch.from_numpy(x_norm_np)
            x_norm = x_norm.to(device=DEVICE, dtype=dtype)
            x = x.to(device=DEVICE, dtype=dtype)

            y_predicted = model(x)
            rmse_precip_ypred += torch.sqrt(
                torch.mean((y_predicted[:, 0, :, :] - y[:, 0, :, :]).pow(2)))
            rmse_precip_x += torch.sqrt(
                torch.mean((x_norm[:, 0, :, :] - y[:, 0, :, :]).pow(2)))
            # rmse_temp_ypred += torch.sqrt(torch.mean((y_predicted[:,1,:,:]-y[:,1,:,:]).pow(2)))
            # rmse_temp_x += torch.sqrt(torch.mean((x_norm[:,1,:,:]-y[:,1,:,:]).pow(2)))
            count += 1

        rmse_precip_ypred /= count
        rmse_precip_x /= count
        print('RMSEs: \tInput precip: %.3f\n\tOutput precip: %.3f\n\t' %
              (rmse_precip_x, rmse_precip_ypred))
    #  rmse_temp_ypred /= count
    #  rmse_temp_x /= count
    # print('RMSEs: \tInput precip: %.3f\n\tOutput precip: %.3f\n\tInput temp: %.3f\n\tOutput temp: %.3f\n\t' %
    #      (rmse_precip_x, rmse_precip_ypred, rmse_temp_x, rmse_temp_ypred))


def check_discriminator_accuracy(loader, D, G):
    D = D.to(device=DEVICE)
    G = G.to(device=DEVICE)
    D.eval()  # set model to evaluation mode
    G.eval()

    count, avg_true_pred, avg_fake_pred = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=DEVICE, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=DEVICE, dtype=dtype)

            true_pred = D(y)
            avg_true_pred += true_pred.sum()
            count += len(true_pred)

            fake_imgs = G(x)
            fake_pred = D(fake_imgs)
            avg_fake_pred += fake_pred.sum()

        avg_true_pred /= count
        avg_fake_pred /= count
        print("Average prediction score on real data: %f" % (avg_true_pred))
        print("Average prediction score on fake data: %f" % (avg_fake_pred))


def check_generator_with_nan_accuracy(loader, model):
    model.eval()
    count, rmse_precip_ypred, rmse_precip_x = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            model = model.to(device=DEVICE)
            y = y.to(device=DEVICE, dtype=dtype)

            x_np = x.numpy()
            x_min = np.amin(x_np, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
            x_max = np.amax(x_np, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
            is_nan = np.int((x_min == x_max).any())
            eps = 1e-9
            x_norm_np = (x_np - x_min) / ((x_max - x_min + is_nan * eps) / 2) - 1
            x_norm_np[np.isnan(x_norm_np)] = 0  # Replace NaN values with zeros

            x_norm = torch.from_numpy(x_norm_np)
            x_norm = x_norm.to(device=DEVICE, dtype=dtype)
            x = x.to(device=DEVICE, dtype=dtype)

            y_predicted = model(x)
            y_predicted[np.isnan(y_predicted)] = 0  # Replace NaN values with zeros

            rmse_precip_ypred += torch.sqrt(
                torch.mean((y_predicted[:, 0, :, :] - y[:, 0, :, :]).pow(2)))
            rmse_precip_x += torch.sqrt(
                torch.mean((x_norm[:, 0, :, :] - y[:, 0, :, :]).pow(2)))
            count += 1

        rmse_precip_ypred /= count
        rmse_precip_x /= count
        print('RMSEs: \tInput precip: %.3f\n\tOutput precip: %.3f\n\t' %
              (rmse_precip_x, rmse_precip_ypred))


def check_discriminator_with_nan_accuracy(loader, D, G):
    D = D.to(device=DEVICE)
    G = G.to(device=DEVICE)
    D.eval()
    G.eval()

    count, avg_true_pred, avg_fake_pred = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=DEVICE, dtype=dtype)
            y = y.to(device=DEVICE, dtype=dtype)

            true_pred = D(y)
            true_pred[np.isnan(true_pred)] = 0  # Replace NaN values with zeros
            avg_true_pred += true_pred.sum()
            count += len(true_pred)

            fake_imgs = G(x)
            fake_pred = D(fake_imgs)
            fake_pred[np.isnan(fake_pred)] = 0  # Replace NaN values with zeros
            avg_fake_pred += fake_pred.sum()

        avg_true_pred /= count
        avg_fake_pred /= count
        print("Average prediction score on real data: %f" % (avg_true_pred))
        print("Average prediction score on fake data: %f" % (avg_fake_pred))


def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)
    out = F.pad(x, pads, "constant", 0)

    return out, pads
