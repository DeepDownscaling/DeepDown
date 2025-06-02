import os
import bottleneck
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from deepdown.config import Config


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")


def covariance_gufunc(x, y):
    x_mean = np.nanmean(x, axis=-1, keepdims=True)
    y_mean = np.nanmean(y, axis=-1, keepdims=True)
    return np.nanmean((x - x_mean) * (y - y_mean), axis=-1)


def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (
        np.nanstd(x, axis=-1) * np.nanstd(y, axis=-1)
    )


def spearman_correlation_gufunc(x, y):
    x_ranks = bottleneck.nanrankdata(x, axis=-1)
    y_ranks = bottleneck.nanrankdata(y, axis=-1)
    return pearson_correlation_gufunc(x_ranks, y_ranks)


def spearman_correlation(x, y, dim):
    return xr.apply_ufunc(
        spearman_correlation_gufunc,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        dask="parallelized",
        output_dtypes=[float],
    )

def plot_maps_for_bc_method(conf, files, method, var):
    models = conf.RCMs
    path_output = conf.path_output

    # Create figure with subplots for all models and data types
    fig, axes = plt.subplots(len(models), len(files),
                             figsize=(25, len(models) * 3.3),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    v_min = None
    v_max = None

    for row, model in enumerate(models):
        data_path = os.path.join(path_output, method, model)
        for col, data_type in enumerate(files):
            file_path = os.path.join(data_path, data_type)

            if os.path.exists(file_path):
                print(f"Loading: {file_path}")
                ds = xr.open_dataset(file_path).load()

                if var in ds:
                    data = ds[var].mean(dim="time")  # Compute time mean

                    if v_min is None and col == 0:
                        v_min = data.min() * 0.8
                        v_max = data.max() * 1.2

                    # Plot on corresponding subplot
                    ax = axes[row, col]
                    add_colorbar = (col == len(files) - 1)
                    add_colorbar = False
                    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                              add_colorbar=add_colorbar, vmin=v_min, vmax=v_max)

                    ax.set_title(f"{model}\n{data_type.split('.')[0]}", fontsize=10)
                    ax.add_feature(cfeature.BORDERS)

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(path_output, f"maps_for_bc_method_{method}_{var}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(path_output, f"maps_for_bc_method_{method}_{var}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {path_output}")


def plot_maps_for_rcm(conf, files, model, var):
    methods = conf.bc_methods
    path_output = conf.path_output

    # Create figure with subplots for all methods and data types
    fig, axes = plt.subplots(len(methods), len(files),
                             figsize=(25, len(methods) * 3.3),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    v_min = None
    v_max = None

    for row, method in enumerate(methods):
        data_path = os.path.join(path_output, method, model)
        for col, data_type in enumerate(files):
            file_path = os.path.join(data_path, data_type)

            if os.path.exists(file_path):
                print(f"Loading: {file_path}")
                ds = xr.open_dataset(file_path).load()

                if var in ds:
                    data = ds[var].mean(dim="time")  # Compute time mean

                    if v_min is None and col == 0:
                        v_min = data.min() * 0.8
                        v_max = data.max() * 1.2

                    # Plot on corresponding subplot
                    ax = axes[row, col]
                    add_colorbar = (col == len(files) - 1)
                    add_colorbar = False
                    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                              add_colorbar=add_colorbar, vmin=v_min, vmax=v_max)

                    ax.set_title(f"{method}\n{data_type.split('.')[0]}", fontsize=10)
                    ax.add_feature(cfeature.BORDERS)

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(path_output, f"maps_for_rcm_{model}_{var}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(path_output, f"maps_for_rcm_{model}_{var}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {path_output}")


def plot_maps_correl_for_rcm(conf, files, model, plot_diff=False):
    methods = conf.bc_methods
    vars = conf.input_vars
    path_output = conf.path_output

    # Define the color limits for the correlation maps
    v_min = -0.5
    v_max = 0.5

    assert len(vars) == 2

    # Loop through the seasons to make 1 plot for each
    for season in ['DJF', 'MAM', 'JJA', 'SON']:

        swiss_crs = ccrs.epsg(2056)

        # Create figure with subplots for all methods and data types
        fig = plt.figure(figsize=(10, len(methods) * 3.3))
        gs = gridspec.GridSpec(len(methods) + 2, 2, figure=fig)

        # Load the reference data
        file_path = os.path.join(path_output, methods[0], model, files[0])
        ds = xr.open_dataset(file_path)
        ds = ds.where(ds.time.dt.season == season, drop=True).load()

        # Compute the spearman's rank correlation
        correl_ref = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")

        ax_big = fig.add_subplot(gs[0, :], projection=swiss_crs)

        # Plot on corresponding subplot
        ax = ax_big
        correl_ref.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                        vmin=v_min, vmax=v_max, add_colorbar=True)
        ax.set_title(f"Reference", fontsize=10)
        ax.add_feature(cfeature.BORDERS)

        # Load the historical model data
        file_path = os.path.join(path_output, methods[0], model, files[1])
        ds = xr.open_dataset(file_path).load()
        ds = ds.where(ds.time.dt.season == season, drop=True)

        # Compute the spearman's rank correlation
        correl_model_hist = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")
        if plot_diff:
            correl_diff = correl_model_hist - correl_ref
            t_prefix = "Difference in Spearman correlation"
        else:
            correl_diff = correl_model_hist
            t_prefix = "Spearman correlation"

        # Plot on corresponding subplot
        ax = fig.add_subplot(gs[1, 0], projection=swiss_crs)
        correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                         vmin=v_min, vmax=v_max, add_colorbar=False)
        ax.set_title(f"{t_prefix} for RCM (control)", fontsize=10)
        ax.add_feature(cfeature.BORDERS)

        # Load the historical model data
        file_path = os.path.join(path_output, methods[0], model, files[3])
        ds = xr.open_dataset(file_path).load()
        ds = ds.where(ds.time.dt.season == season, drop=True)

        # Compute the spearman's rank correlation
        correl_model_proj = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")
        if plot_diff:
            correl_diff = correl_model_proj - correl_ref
        else:
            correl_diff = correl_model_proj

        # Plot on corresponding subplot
        ax = fig.add_subplot(gs[1, 1], projection=swiss_crs)
        correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                         vmin=v_min, vmax=v_max, add_colorbar=False)

        ax.set_title(f"{t_prefix} for RCM (future)", fontsize=10)
        ax.add_feature(cfeature.BORDERS)

        # Keep only the 3rd and 5th files
        files_deb = [files[2], files[4]]

        # Loop through methods
        for row, method in enumerate(methods):
            data_path = os.path.join(path_output, method, model)
            for col, data_type in enumerate(files_deb):
                file_path = os.path.join(data_path, data_type)

                if os.path.exists(file_path):
                    print(f"Loading: {file_path}")
                    ds = xr.open_dataset(file_path).load()
                    ds = ds.where(ds.time.dt.season == season, drop=True)

                    # Compute the spearman's rank correlation
                    correl = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")

                    if plot_diff:
                        correl_diff = correl - correl_ref
                    else:
                        correl_diff = correl

                    # Plot on corresponding subplot
                    ax = fig.add_subplot(gs[row + 2, col], projection=swiss_crs)
                    correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                                     add_colorbar=False)

                    subtitle = 'control' if col == 0 else 'future'
                    ax.set_title(f"{t_prefix} for {method} ({subtitle})", fontsize=10)
                    ax.add_feature(cfeature.BORDERS)

        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(path_output, f"maps_correl_for_rcm_{model}_{season}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(path_output, f"maps_correl_for_rcm_{model}_{season}.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved in {path_output}")


def main():
    # Load the configuration
    conf = Config().get()
    models = conf.RCMs
    methods = conf.bc_methods

    # plots = ['maps_for_bc_method', 'maps_for_rcm', 'maps_correl_for_rcm']
    plots = ['maps_correl_for_rcm']

    files = [
        "target_data_hist_original.nc",
        "input_data_hist_original.nc",
        "input_data_hist_debiased.nc",
        "input_data_proj_original.nc",
        "input_data_proj_debiased.nc"
    ]

    if 'maps_for_bc_method' in plots:
        for method in methods:
            for var in conf.input_vars:
                plot_maps_for_bc_method(conf, files, method, var)

    if 'maps_for_rcm' in plots:
        for model in models:
            for var in conf.input_vars:
                plot_maps_for_rcm(conf, files, model, var)

    if 'maps_correl_for_rcm' in plots:
        for model in models:
            plot_maps_correl_for_rcm(conf, files, model)


if __name__ == "__main__":
    main()
