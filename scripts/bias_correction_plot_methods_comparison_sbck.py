import os
import xarray as xr
import xskillscore as xs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from deepdown.config import Config


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
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=":")

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
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=":")

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

        # Create figure with subplots for all methods and data types
        fig, axes = plt.subplots(len(methods) + 2, 2,
                                 figsize=(10, len(methods) * 3.3),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

        # Load the reference data
        file_path = os.path.join(path_output, methods[0], model, files[0])
        ds = xr.open_dataset(file_path)
        ds = ds.where(ds.time.dt.season == season, drop=True).load()

        # Compute the spearman's rank correlation
        #correl_ref = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")
        correl_ref = xs.spearman_r(ds[vars[0]], ds[vars[1]], dim="time", skipna=True)

        # Plot on corresponding subplot
        ax = axes[0, 0]
        correl_ref.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                        vmin=v_min, vmax=v_max, add_colorbar=False)
        ax.set_title(f"Reference", fontsize=10)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        # Add the colorbar in axes[0, 1]
        cbar_ax = fig.add_axes([0.65, 0.9, 0.02, 0.07])  # [left, bottom, width, height]
        correl_ref.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                        vmin=v_min, vmax=v_max, cbar_ax=cbar_ax)
        # Remove the box for axes[0,1]
        for spine in axes[0, 1].spines.values():
            spine.set_visible(False)

        # Load the historical model data
        file_path = os.path.join(path_output, methods[0], model, files[1])
        ds = xr.open_dataset(file_path).load()
        ds = ds.where(ds.time.dt.season == season, drop=True)

        # Compute the spearman's rank correlation
        correl_model_hist = xs.spearman_r(ds[vars[0]], ds[vars[1]], dim="time", skipna=True)
        if plot_diff:
            correl_diff = correl_model_hist - correl_ref
        else:
            correl_diff = correl_model_hist

        # Plot on corresponding subplot
        ax = axes[1, 0]
        correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                         vmin=v_min, vmax=v_max, add_colorbar=False)
        ax.set_title(f"Difference for model (control)", fontsize=10)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        # Load the historical model data
        file_path = os.path.join(path_output, methods[0], model, files[3])
        ds = xr.open_dataset(file_path).load()
        ds = ds.where(ds.time.dt.season == season, drop=True)

        # Compute the spearman's rank correlation
        correl_model_proj = xs.spearman_r(ds[vars[0]], ds[vars[1]], dim="time", skipna=True)
        if plot_diff:
            correl_diff = correl_model_proj - correl_ref
        else:
            correl_diff = correl_model_proj

        # Plot on corresponding subplot
        ax = axes[1, 1]
        correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                         vmin=v_min, vmax=v_max, add_colorbar=False)
        ax.set_title(f"Difference for model (future)", fontsize=10)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        # Keep only the 3rd and 5th files
        files = [files[2], files[4]]

        # Loop through methods
        for row, method in enumerate(methods):
            data_path = os.path.join(path_output, method, model)
            for col, data_type in enumerate(files):
                file_path = os.path.join(data_path, data_type)

                if os.path.exists(file_path):
                    print(f"Loading: {file_path}")
                    ds = xr.open_dataset(file_path).load()
                    ds = ds.where(ds.time.dt.season == season, drop=True)

                    # Compute the spearman's rank correlation
                    correl = xs.spearman_r(ds[vars[0]], ds[vars[1]], dim="time", skipna=True)

                    if plot_diff:
                        correl_diff = correl - correl_ref
                    else:
                        correl_diff = correl

                    # Plot on corresponding subplot
                    ax = axes[row + 2, col]
                    correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                                     add_colorbar=False)

                    t_prefix = "Spearman correlation"
                    if plot_diff:
                        t_prefix = "Difference in Spearman correlation"

                    subtitle = 'control' if col == 0 else 'future'
                    ax.set_title(f"{t_prefix} for {method} ({subtitle})", fontsize=10)
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=":")

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
