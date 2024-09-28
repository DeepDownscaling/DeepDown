import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


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



def plot_grid_points(input_data, grid_points, variables,target_coarser =None):
    """
    Plot specified variables and their debiased versions for multiple grid points.

    Parameters:
    input_data (dict): Dictionary containing the data arrays.
    grid_points (list of tuples): List of tuples where each tuple contains (row, col) coordinates.
    variables (list of str): List of variable names to plot.
    """
    num_rows = len(grid_points)
    num_cols = len(variables)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16*num_cols, 12*num_rows), squeeze=False)
    
    for i, (row, col) in enumerate(grid_points):
        for j, var in enumerate(variables):
            ax = axes[i, j]
            var_deb = f"{var}_deb"
            if var in input_data:
                input_data[var][:, row, col].plot(ax=ax,linestyle='dashed',color='red', label=f'{var} ({row},{col})')
            if var_deb in input_data:
                input_data[var_deb][:, row, col].plot(ax=ax, color='black',label=f'{var_deb} ({row},{col})')
            if target_coarser is not None:
                target_coarser[var][:, row, col].plot(ax=ax, color='blue',label=f'MCH{var} ({row},{col})')

                
            
            ax.set_xlabel('Time', fontsize=22)
            ax.set_ylabel(var, fontsize=22)
            ax.legend(prop=dict(size=22))

    
    plt.tight_layout()
    plt.show()


def plot_maps(input_data, variables, target_coarser=None, date =None, title=None, fig_name=None, crop=True):
    """
    Plot mean maps for specified variables and their debiased versions.

    Parameters:
    input_data (dict): Dictionary containing the data arrays.
    variables (list of str): List of variable names to plot.
    """
    num_vars = len(variables)
    num_rows = num_vars
    num_cols = 3  # One for the original variable, one for the debiased variable, one for the coarser version
    
    if crop:
        input_data = input_data.sel(x=slice(target_coarser.x.min(), target_coarser.x.max()), y=slice(target_coarser.y.max(),target_coarser.y.min()))
        
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows), subplot_kw={'projection': ccrs.PlateCarree()})
    
    for i, var in enumerate(variables):
        var_deb = f"{var}_deb"
        ax_original = axes[i][0] if num_vars > 1 else axes[0]
        ax_debiased = axes[i][1] if num_vars > 1 else axes[1]
        ax_coarser = axes[i][2] if num_vars > 1 else axes[2]
     
        
        if var in input_data:
            if date is not None:
                var_data = input_data[var].isel(time=date)
            else:
                var_data = input_data[var]
            var_data.plot(ax=ax_original, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'shrink': 0.5})
            ax_original.add_feature(cfeature.COASTLINE)
            ax_original.add_feature(cfeature.BORDERS)
            # ax_original.set_title(f'{var}', fontsize=14)
        
        if var_deb in input_data:
            if date is not None:
                var_deb = input_data[var_deb].isel(time=date)
            else:
                var_deb = input_data[var_deb]
            var_deb.plot(ax=ax_debiased, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'shrink': 0.5})
            ax_debiased.add_feature(cfeature.COASTLINE)
            ax_debiased.add_feature(cfeature.BORDERS)
            # ax_debiased.set_title(f'{var_deb}', fontsize=14)

        if target_coarser is not None and var in target_coarser:
            if date is not None:
                var_ch_coarser = target_coarser[var].isel(time=date)
            else:
                var_ch_coarser = target_coarser[var]

            var_ch_coarser.plot(ax=ax_coarser, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'shrink': 0.5})
            ax_coarser.add_feature(cfeature.COASTLINE)
            ax_coarser.add_feature(cfeature.BORDERS)
            # ax_coarser.set_title(f'{var} Coarser', fontsize=14)

         # Add the title to the figure if provided
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if fig_name:
            # Create the directory if it doesn't exist
            dir_name = os.path.dirname(fig_name)
            if dir_name:  # Only create the directory if it's not an empty string (i.e., a path is provided)
                os.makedirs(dir_name, exist_ok=True)
            
            fig.savefig(fig_name, bbox_inches='tight')
            print(f"Figure saved as {fig_name}")
        
    plt.show()
  


def get_extreme_date(data,var):
    # Get the date with the max number of grid points with pr > threshold
    if var == 't':
        x = data[var].mean(dim=['y', 'x'])
    else:
        x = data[var].sum(dim=['y', 'x'])
    return x.argmax(dim='time')