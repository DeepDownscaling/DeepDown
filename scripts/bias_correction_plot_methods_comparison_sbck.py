# Script to plot bias correction methods comparison performed with the script
# bias_correction_compare_methods_sbck.py

import warnings
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from deepdown.config import Config
from deepdown.utils.utils import spearman_correlation

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice")

PLOTS = ['maps_for_bc_method', 'maps_for_rcm', 'maps_correl_for_rcm'] # List of plots
CORREL_VARS = ['tp', 't']  # Variables to compute correlation for
CORREL_MIN = -0.5  # Minimum value for correlation plots
CORREL_MAX = 0.5  # Maximum value for correlation plots


def plot_maps_stat_for_bc_method(conf, files, titles, method, var, stat, epsg=2056):
    models = conf.RCMs
    path_output = Path(conf.path_output)
    crs = ccrs.epsg(epsg)

    # Create figure with subplots for all models
    fig = plt.figure(figsize=(17, len(models) * 3.3))
    gs = gridspec.GridSpec(len(models) + 2, len(files), figure=fig)

    v_min = None
    v_max = None

    for row, model in enumerate(models):
        data_path = path_output / method / model
        for col, data_type in enumerate(files):
            file_path = data_path / data_type

            if file_path.exists():
                ds = xr.open_dataset(file_path).load()

                if var in ds:
                    data = _compute_stat(ds, stat, var)
                    if data is None:
                        continue

                    v_min, v_max = _get_min_max(data, v_min, v_max, var, stat)

                    # Plot on corresponding subplot
                    ax = fig.add_subplot(gs[row, col], projection=crs)
                    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=_get_cmap(var),
                              add_colorbar=False, vmin=v_min, vmax=v_max)
                    ax.set_title(f"{model}\n{titles[data_type]}", fontsize=12)
                    ax.add_feature(cfeature.BORDERS)

    plt.tight_layout()

    # Save the figure
    plt.savefig(path_output / f"maps_{stat}_for_bc_method_{method}_{var}.png",
                dpi=300, bbox_inches="tight")
    plt.savefig(path_output / f"maps_{stat}_for_bc_method_{method}_{var}.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {path_output}")


def plot_maps_stat_for_rcm(conf, files, titles, model, var, stat, epsg=2056):
    methods = conf.bc_methods
    path_output = Path(conf.path_output)
    crs = ccrs.epsg(epsg)

    # Create figure with subplots for all methods
    fig = plt.figure(figsize=(17, len(methods) * 3.3))
    gs = gridspec.GridSpec(len(methods) + 2, len(files), figure=fig)

    v_min = None
    v_max = None

    for row, method in enumerate(methods):
        data_path = path_output / method / model
        for col, data_type in enumerate(files):
            file_path = data_path / data_type

            if file_path.exists():
                ds = xr.open_dataset(file_path).load()

                if var in ds:
                    data = _compute_stat(ds, stat, var)
                    if data is None:
                        continue

                    v_min, v_max = _get_min_max(data, v_min, v_max, var, stat)

                    # Plot on corresponding subplot
                    ax = fig.add_subplot(gs[row, col], projection=crs)
                    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=_get_cmap(var),
                              add_colorbar=False, vmin=v_min, vmax=v_max)
                    ax.set_title(f"{method}\n{titles[data_type]}", fontsize=12)
                    ax.add_feature(cfeature.BORDERS)

    plt.tight_layout()

    # Save the figure
    plt.savefig(path_output / f"maps_{stat}_for_rcm_{model}_{var}.png", dpi=300,
                bbox_inches="tight")
    plt.savefig(path_output / f"maps_{stat}_for_rcm_{model}_{var}.pdf", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {path_output}")


def plot_maps_correl_for_rcm(conf, files, model, plot_diff=False, epsg=2056):
    methods = conf.bc_methods
    vars = CORREL_VARS
    path_output = Path(conf.path_output)
    crs = ccrs.epsg(epsg)
    v_min = CORREL_MIN
    v_max = CORREL_MAX

    assert len(vars) == 2

    if plot_diff:
        t_prefix = "Diff. rank correl."
    else:
        t_prefix = "Rank correlation"

    # Loop through the seasons to make 1 plot for each
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        fig = plt.figure(figsize=(7, len(methods) * 3.3))
        gs = gridspec.GridSpec(len(methods) + 2, 2, figure=fig)

        # Load the reference data
        ds = xr.open_dataset(path_output / methods[0] / model / files[0]).load()
        ds = ds.where(ds.time.dt.season == season, drop=True)

        # Compute the spearman's rank correlation
        correl_ref = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")
        ax = fig.add_subplot(gs[0, :], projection=crs)
        correl_ref.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                        vmin=v_min, vmax=v_max, add_colorbar=True)
        ax.set_title(f"Reference", fontsize=12)
        ax.add_feature(cfeature.BORDERS)

        # Load the RCM data for the control period
        ds = xr.open_dataset(path_output / methods[0] / model / files[1]).load()
        ds = ds.where(ds.time.dt.season == season, drop=True)

        # Compute the spearman's rank correlation
        correl_model_hist = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")
        if plot_diff:
            correl_diff = correl_model_hist - correl_ref
        else:
            correl_diff = correl_model_hist
        ax = fig.add_subplot(gs[1, 0], projection=crs)
        correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                         vmin=v_min, vmax=v_max, add_colorbar=False)
        ax.set_title(f"{t_prefix} for RCM (control)", fontsize=12)
        ax.add_feature(cfeature.BORDERS)

        # Load the RCM data for the future period
        ds = xr.open_dataset(path_output / methods[0] / model / files[3]).load()
        ds = ds.where(ds.time.dt.season == season, drop=True)

        # Compute the spearman's rank correlation
        correl_model_proj = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")
        if plot_diff:
            correl_diff = correl_model_proj - correl_ref
        else:
            correl_diff = correl_model_proj
        ax = fig.add_subplot(gs[1, 1], projection=crs)
        correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm",
                         vmin=v_min, vmax=v_max, add_colorbar=False)
        ax.set_title(f"{t_prefix} for RCM (future)", fontsize=12)
        ax.add_feature(cfeature.BORDERS)

        # Keep only the debiased files
        files_deb = [files[2], files[4]]

        # Loop through methods
        for row, method in enumerate(methods):
            data_path = path_output / method / model
            for col, data_type in enumerate(files_deb):
                file_path = data_path / data_type

                if file_path.exists():
                    ds = xr.open_dataset(file_path).load()
                    ds = ds.where(ds.time.dt.season == season, drop=True)

                    # Compute the spearman's rank correlation
                    correl = spearman_correlation(ds[vars[0]], ds[vars[1]], dim="time")
                    if plot_diff:
                        correl_diff = correl - correl_ref
                    else:
                        correl_diff = correl
                    ax = fig.add_subplot(gs[row + 2, col], projection=crs)
                    correl_diff.plot(ax=ax, transform=ccrs.PlateCarree(),
                                     cmap="coolwarm", add_colorbar=False)

                    subtitle = 'control' if col == 0 else 'future'
                    ax.set_title(f"{t_prefix} for {method} ({subtitle})",
                                 fontsize=12)
                    ax.add_feature(cfeature.BORDERS)

        plt.tight_layout()

        # Save the figure
        if plot_diff:
            file_name = f"maps_correl_diff_for_rcm_{model}_{season}"
        else:
            file_name = f"maps_correl_for_rcm_{model}_{season}"

        plt.savefig(path_output / f"{file_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(path_output / f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved in {path_output}")


def _get_cmap(var):
    """Get the colormap based on the variable."""
    if var == 'tp':
        return 'Blues'
    elif var in ['t', 't_max', 't_min']:
        return 'turbo'
    else:
        return 'viridis'


def _compute_stat(ds, stat, var):
    if stat == 'mean':
        data = ds[var].mean(dim="time")
    elif stat == 'q99':
        data = ds[var].quantile(0.99, dim="time")
    elif stat == 'f0':
        if var != 'tp':
            return None
        data = (ds[var].where(ds[var] > 0.1).count(dim="time") /
                ds[var].count(dim="time"))
    else:
        raise ValueError(f"Unknown statistic: {stat}")

    return data


def _get_min_max(data, v_min, v_max, var, stat):
    if v_min is None:
        if var == 'tp':
            v_min = 0
            v_max = data.max() * 1.1
            if stat == 'f0':
                v_max = 1.0
        else:
            v_min = data.min() * 0.8
            v_max = data.max() * 1.2

    return v_min, v_max


def main():
    # Load the configuration
    conf = Config().get()
    models = conf.RCMs
    methods = conf.bc_methods

    files = [
        "target_data_hist_original.nc",
        "input_data_hist_original.nc",
        "input_data_hist_debiased.nc",
        "input_data_proj_original.nc",
        "input_data_proj_debiased.nc"
    ]
    titles = {
        'target_data_hist_original.nc': 'Target data (control period)',
        'input_data_hist_original.nc': 'RCM (control period, original)',
        'input_data_hist_debiased.nc': 'RCM (control period, debiased)',
        'input_data_proj_original.nc': 'RCM (future, original)',
        'input_data_proj_debiased.nc': 'RCM (future, debiased)'
    }
    stats = ['mean', 'q99', 'f0']  # Statistics to compute

    if 'maps_correl_for_rcm' in PLOTS:
        for model in models:
            plot_maps_correl_for_rcm(conf, files, model)

    if 'maps_for_bc_method' in PLOTS:
        for method in methods:
            for var in conf.input_vars:
                for stat in stats:
                    plot_maps_stat_for_bc_method(conf, files, titles, method, var, stat)

    if 'maps_for_rcm' in PLOTS:
        for model in models:
            for var in conf.input_vars:
                for stat in stats:
                    plot_maps_stat_for_rcm(conf, files, titles, model, var, stat)

    print("All plots generated successfully.")


if __name__ == "__main__":
    main()
