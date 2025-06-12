import bottleneck
import numpy as np
import xarray as xr


def covariance_gufunc(x, y):
    x_mean = np.nanmean(x, axis=-1, keepdims=True)
    y_mean = np.nanmean(y, axis=-1, keepdims=True)
    return np.nanmean((x - x_mean) * (y - y_mean), axis=-1)


def pearson_correlation_gufunc(x, y):
    with np.errstate(invalid='ignore', divide='ignore'):
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
