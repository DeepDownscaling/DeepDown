import torch
import numpy as np
import torch.nn as nn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    desired_labels = torch.ones(N,1).to(device=device, dtype=dtype)
    BCE_Loss = nn.BCELoss()
    adversarial_loss = BCE_Loss(logits_fake, desired_labels)
    
    total_loss = content_loss + weight_param*adversarial_loss
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
    real_labels = torch.ones(N,1).to(device=device, dtype=dtype)
    BCE_Loss = nn.BCELoss()
    L1 = BCE_Loss(logits_real, real_labels)
        
    # How often it gets fooled into thinking fake images are real
    fake_labels = torch.zeros(N,1).to(device=device, dtype=dtype)
    L2 = BCE_Loss(logits_fake, fake_labels)
    
#     print("L1 (how bad on real data): %f\t L2 (how bad on fake data): %f" % (L1, L2))
    
    loss = (L1 + L2)
    return loss, L1, L2


# functions to plot and train
def check_generator_accuracy(loader, model):
    #credits to: https://github.com/mantariksh/231n_downscaling/blob/master/SRGAN.ipynb
    
    model.eval() # set model to evaluation mode
    count, rmse_precip_ypred, rmse_precip_x = 0, 0, 0
    #rmse_temp_ypred, rmse_temp_x = 0, 0
    with torch.no_grad():
        for x, y in loader:
            model = model.to(device=device)
            y = y.to(device=device, dtype=dtype)
            
            # Normalize x to be in -1 to 1 for purpose of comparing with high res data in same range
            # Turn it into a numpy array
            x_np = x.numpy()
            x_min = np.amin(x_np, axis=(2,3))[:, :, np.newaxis, np.newaxis]
            x_max = np.amax(x_np, axis=(2,3))[:, :, np.newaxis, np.newaxis]
            is_nan = np.int((x_min == x_max).any())
            eps = 1e-9
            x_norm_np = (x_np - x_min) / ((x_max - x_min + is_nan*eps) / 2) - 1
            
            x_norm = torch.from_numpy(x_norm_np)
            x_norm = x_norm.to(device=device, dtype=dtype)
            x = x.to(device=device, dtype=dtype)
            
            y_predicted = model(x)
            rmse_precip_ypred += torch.sqrt(torch.mean((y_predicted[:,0,:,:]-y[:,0,:,:]).pow(2)))
            rmse_precip_x += torch.sqrt(torch.mean((x_norm[:,0,:,:]-y[:,0,:,:]).pow(2)))
            #rmse_temp_ypred += torch.sqrt(torch.mean((y_predicted[:,1,:,:]-y[:,1,:,:]).pow(2)))
            #rmse_temp_x += torch.sqrt(torch.mean((x_norm[:,1,:,:]-y[:,1,:,:]).pow(2)))
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
    D = D.to(device=device)
    G = G.to(device=device)
    D.eval() # set model to evaluation mode
    G.eval()
    
    count, avg_true_pred, avg_fake_pred = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype) # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

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