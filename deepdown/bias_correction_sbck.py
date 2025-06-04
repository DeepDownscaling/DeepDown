import argparse
import logging
import copy
import numpy as np
import xarray as xr
from pathlib import Path

from deepdown.utils.debiaser_utils import extract_for_sbck, debias_with_sbck
from deepdown.utils.data_loader import DataLoader
from deepdown.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_bias_correction(conf, method=None, preload_data=True, **kwargs):
    """
    Correct bias in the input data using the target data.

    Parameters
    ----------
    conf : dict
        Configuration dictionary. Contains the bias correction method and paths to the
        input and target data. bc_method should be one of the following.

        Univariate methods:
        - 'QM': Quantile Mapping method.
        - 'QDM' [1]: Quantile delta mapping method.
        - 'CDFt' [2]: Cumulative Distribution Function transfer. Quantile Mapping bias
           corrector, taking account of an evolution of the distribution.

        Multivariate methods:
        - 'MRec' [3]: Matrix Recorrelation method. Perform a multivariate bias
           correction with Gaussian assumption.
        - 'ECBC' [4]: Empirical Copula Bias Correction. Use Schaake shuffle.
        - 'MBCn' [5]: MBCn Bias correction method
        - 'QMrs' [6]: Quantile Mapping bias corrector with multivariate rank shuffle
        - 'R2D2' [6]: Rank Resampling for Distributions and Dependences method. Non
           stationnary Quantile Mapping bias corrector with multivariate rankshuffle.
        - 'AR2D2' [7]: Analogues Rank Resampling for Distributions and Dependences.
           Multivariate bias correction with quantiles shuffle.
        - 'OTC' [8]: Optimal Transport bias Corrector
        - 'dOTC' [8]: Dynamical Optimal Transport bias Corrector, taking account of an
           evolution of the distribution
        - 'TSMBC' [9]: Time Shifted Multivariate Bias Correction. Correct
           auto-correlation with a shift approach.
        - 'dTSMBC' [9]: Time Shifted Multivariate Bias Correction where observations
           are unknown. Perform a bias correction of auto-correlation.

        Only for comparison:
        - 'RBC': Random Bias Correction
        - 'IdBC': Identity Bias Correction

        In addition, the following configuration parameters are used:
        bc_config:
            dims: '2d'  # '2d': MBC method is applied independently at each grid cell
                        #       but jointly corrects both temperature and precipitation
                        #       time series.
                        # 'full': all time series are corrected jointly over the entire
                        #       grid for both temperature and precipitation.

    method : str, optional
        The bias correction method to use. If None, the method from the configuration is used.
    preload_data: boolean
        Whether to preload the data in memory or not.
    kwargs : additional arguments for SBCK methods
        Additional arguments for the bias correction methods. See SBCK documentation for
        details on the available arguments for each method.

    References
	----------
    [1] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias Correction of GCM
        Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in
        Quantiles and Extremes? Journal of Climate, 28(17), 6938–6959.
        https://doi.org/10.1175/JCLI-D-14-00754.1
	[2] Michelangeli, P. ‐A., Vrac, M., & Loukos, H. (2009). Probabilistic downscaling
	    approaches: Application to wind cumulative distribution functions. Geophysical
	    Research Letters, 36(11), 2009GL038401. https://doi.org/10.1029/2009GL038401
    [3] Bárdossy, A., & Pegram, G. (2012). Multiscale spatial recorrelation of RCM
        precipitation to produce unbiased climate change scenarios over large areas
        and small. Water Resources Research, 48(9), 2011WR011524.
        https://doi.org/10.1029/2011WR011524
    [4] Vrac, M., & Friederichs, P. (2015). Multivariate—Intervariable, Spatial, and
        Temporal—Bias Correction. Journal of Climate, 28(1), 218–237.
        https://doi.org/10.1175/JCLI-D-14-00059.1
    [5] Cannon, A. J. (2018). Multivariate quantile mapping bias correction: An
        N-dimensional probability density function transform for climate model
        simulations of multiple variables. Climate Dynamics, 50(1–2), 31–49.
        https://doi.org/10.1007/s00382-017-3580-6
    [6] Vrac, M. (2018). Multivariate bias adjustment of high-dimensional climate
        simulations: The Rank Resampling for Distributions and Dependences (R2D2)
        bias correction. Hydrology and Earth System Science, 22, 3175–3196.
        https://doi.org/10.5194/hess-22-3175-2018
	[7] Vrac, M., & Thao, S. (2020). R2 D2 v2.0: Accounting for temporal dependences in
	    multivariate bias correction via analogue rank resampling. Geoscientific Model
	    Development, 13(11), 5367–5387. https://doi.org/10.5194/gmd-13-5367-2020
    [8] Robin, Y., Vrac, M., Naveau, P., & Yiou, P. (2019). Multivariate stochastic
        bias corrections with optimal transport. Hydrology and Earth System Sciences,
        23(2), 773–786. https://doi.org/10.5194/hess-23-773-2019
    [9] Robin, Y., & Vrac, M. (2021). Is time a variable like the others in
        multivariate statistical downscaling and bias correction? Earth System Dynamics,
        12, 1253–1273. https://doi.org/10.5194/esd-12-1253-2021
    """
    logger.info("Loading input and targets data")

    # Get the bias correction method from the configuration if not provided
    if method is None:
        method = conf.bc_method

    # Check that the method is valid
    if method not in conf.bc_methods:
        raise ValueError(f"Invalid bias correction method: {method}")

    # Load target data for the historical period
    target_data_hist = DataLoader(path_tmp=conf.path_tmp)
    target_data_hist.load(conf.period_hist_start, conf.period_hist_end,
                          conf.path_targets)

    # Load input data (e.g. climate model) for both historical and future periods
    input_data = DataLoader(path_tmp=conf.path_tmp)
    input_data.load(conf.period_hist_start, conf.period_proj_end, conf.path_inputs)

    # Get the extent of the domain of interest
    x_min, x_max, y_min, y_max = input_data.get_extent_from(
        target_data_hist, from_proj='CH1903+', to_proj='WGS84')

    # Select the domain of interest
    input_data.select_domain(x_min, x_max, y_min, y_max)
    input_data_hist = input_data
    input_data_proj = copy.deepcopy(input_data)
    input_data_hist.select_period(conf.period_hist_start, conf.period_hist_end)
    input_data_proj.select_period(conf.period_proj_start, conf.period_proj_end)
    del input_data

    # Coarsen the target data to the resolution of the input data
    target_data_hist.coarsen(
        x_axis=input_data_hist.data.x, y_axis=input_data_hist.data.y,
        from_proj='CH1903_LV95', to_proj='WGS84')

    if preload_data:
        logger.info(f"Preloading data in memory.")
        input_data_hist.data.load()
        input_data_proj.data.load()
        target_data_hist.data.load()

    # If the time dimension length differs between the input and target data (e.g.,
    # due to ignored leap years), adjust the time dimension of the target data to match
    # the input data
    if input_data_hist.data.time.size != target_data_hist.data.time.size:
        logger.info(f"Adjusting time dimension of target data to match input data.")

        # Extract date (YYYY-MM-DD) from both indexes
        input_dates = np.array([str(t)[:10] for t in input_data_hist.data.time.values])
        targ_dates = np.array([str(t)[:10] for t in target_data_hist.data.time.values])

        # Find common dates
        common_dates = np.intersect1d(input_dates, targ_dates)

        # Select by matching string times
        target_data_hist.data = target_data_hist.data.sel(
            time=[t for t, d in zip(target_data_hist.data.time.values, targ_dates) if
                  d in common_dates]
        )

    # Create the output data arrays
    output_array_hist = []
    output_array_proj = []
    for i, var_out in enumerate(conf.target_vars):
        output_array_hist.append(np.ones(input_data_hist.data[var_out].shape) * np.nan)
        output_array_proj.append(np.ones(input_data_proj.data[var_out].shape) * np.nan)

    if conf.bc_config['dims'] == '2d':

        # Proceed to the point-wise bias correction
        x_axis = input_data_hist.data.x
        y_axis = input_data_hist.data.y
        for x_idx in range(x_axis.size):
            for y_idx in range(y_axis.size):
                x = float(x_axis[x_idx])
                y = float(y_axis[y_idx])

                # Prepare the data for SBCK
                target_array_hist = []
                input_array_hist = []
                input_array_proj = []
                for var_target, var_input in zip(conf.target_vars, conf.input_vars):
                    # Convert and extract the values as numpy arrays
                    target_array_hist_v = extract_for_sbck(target_data_hist, var_target, x, y)
                    if target_array_hist_v is None:
                        logger.debug(f"Skipping point ({x:.2f}, {y:.2f}) for {var_target}")
                        break

                    input_array_hist_v = extract_for_sbck(input_data_hist, var_input, x, y)
                    input_array_proj_v = extract_for_sbck(input_data_proj, var_input, x, y)

                    assert input_array_hist_v is not None, f"Missing data for {var_input} at ({x:.2f}, {y:.2f})"
                    assert input_array_proj_v is not None, f"Missing data for {var_input} at ({x:.2f}, {y:.2f})"

                    # Append to the list
                    target_array_hist.append(target_array_hist_v)
                    input_array_hist.append(input_array_hist_v)
                    input_array_proj.append(input_array_proj_v)

                else:  # Only proceed if the point has valid data for all variables
                    # Stack the data
                    target_array_hist = np.stack(target_array_hist, axis=1)
                    input_array_hist = np.stack(input_array_hist, axis=1)
                    input_array_proj = np.stack(input_array_proj, axis=1)

                    # Bias correct all variables simultaneously
                    debiased_proj_ts, debiased_hist_ts = debias_with_sbck(
                        method, input_array_proj, input_array_hist,
                        target_array_hist, **kwargs)

                    # Store the debiased time series
                    for i, var_out in enumerate(conf.target_vars):
                        output_array_hist[i][:, y_idx, x_idx] = debiased_hist_ts[:, conf.target_vars.index(var_out)]
                        output_array_proj[i][:, y_idx, x_idx] = debiased_proj_ts[:, conf.target_vars.index(var_out)]

    elif conf.bc_config['dims'] == 'full':

        # Prepare the data for SBCK
        mask = None
        target_array_hist = []
        input_array_hist = []
        input_array_proj = []
        for var_target, var_input in zip(conf.target_vars, conf.input_vars):
            # Convert and extract the values as numpy arrays
            target_array_hist_v = extract_for_sbck(target_data_hist, var_target)
            input_array_hist_v = extract_for_sbck(input_data_hist, var_input)
            input_array_proj_v = extract_for_sbck(input_data_proj, var_input)

            # Flatten spatial dimensions
            target_array_hist_v = target_array_hist_v.reshape(
                target_array_hist_v.shape[0], -1)
            input_array_hist_v = input_array_hist_v.reshape(
                input_array_hist_v.shape[0], -1)
            input_array_proj_v = input_array_proj_v.reshape(
                input_array_proj_v.shape[0], -1)

            # Append to the list
            target_array_hist.append(target_array_hist_v)
            input_array_hist.append(input_array_hist_v)
            input_array_proj.append(input_array_proj_v)

            # Get the mask of non nan or inf values in the target data
            mask_v = np.isfinite(target_array_hist_v)
            mask_v = np.all(mask_v, axis=0)
            if mask is None:
                mask = mask_v
            else:
                mask = np.logical_and(mask, mask_v)

        # Stack the data
        target_array_hist = np.hstack(target_array_hist)
        input_array_hist = np.hstack(input_array_hist)
        input_array_proj = np.hstack(input_array_proj)

        # Copy the mask to match the number of variables
        mask_full = np.tile(mask, len(conf.target_vars))
        mask_2d = mask.reshape(output_array_hist[0].shape[1:])

        # Extract the pixels where the mask is True and stack them into a 2D array (time, pixels)
        target_array_hist = target_array_hist[:, mask_full]
        input_array_hist = input_array_hist[:, mask_full]
        input_array_proj = input_array_proj[:, mask_full]

        # Bias correct all variables simultaneously
        debiased_proj_ts, debiased_hist_ts = debias_with_sbck(
            method, input_array_proj, input_array_hist,
            target_array_hist, **kwargs)

        # Store the debiased time series
        for i, var_out in enumerate(conf.target_vars):
            var_idx = conf.target_vars.index(var_out)
            col_start = var_idx * target_array_hist.shape[1] // len(conf.target_vars)
            col_end = (var_idx + 1) * target_array_hist.shape[1] // len(conf.target_vars)

            output_array_hist[i][:, mask_2d] = debiased_hist_ts[:, col_start:col_end]
            output_array_proj[i][:, mask_2d] = debiased_proj_ts[:, col_start:col_end]

    else:
        raise ValueError(f"Invalid bc_config['dims']: {conf.bc_config['dims']}. "
                         f"Expected '2d' or 'full'.")

    # Create a dictionary for the data variables
    data_vars_hist = {var: (('time', 'y', 'x'), output_array_hist[i]) for i, var in
                      enumerate(conf.target_vars)}
    data_vars_proj = {var: (('time', 'y', 'x'), output_array_proj[i]) for i, var in
                      enumerate(conf.target_vars)}

    # Create the xarray dataset
    output_data_hist = xr.Dataset(
        data_vars=data_vars_hist,
        coords={
            'time': input_data_hist.data['time'],
            'y': input_data_hist.data['y'],
            'x': input_data_hist.data['x']
        }
    )
    output_data_proj = xr.Dataset(
        data_vars=data_vars_proj,
        coords={
            'time': input_data_proj.data['time'],
            'y': input_data_proj.data['y'],
            'x': input_data_proj.data['x']
        }
    )

    # Save the debiased dataset to a NetCDF file
    output_path = Path(conf.path_output)
    output_path.mkdir(parents=True, exist_ok=True)
    target_data_hist.data.to_netcdf(output_path / "target_data_hist_original.nc")
    input_data_hist.data.to_netcdf(output_path / "input_data_hist_original.nc")
    output_data_hist.to_netcdf(output_path / "input_data_hist_debiased.nc")
    input_data_proj.data.to_netcdf(output_path / "input_data_proj_original.nc")
    output_data_proj.to_netcdf(output_path / "input_data_proj_debiased.nc")
    logger.info(f"Debiased dataset saved to {output_path}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--config_file", default='../config.yaml',
                           help="Path to the .yml config file")
    args = argParser.parse_args()

    logger.info("Loading configuration...")
    conf = Config(args)
    conf.print()

    logger.info("Starting bias correction")
    run_bias_correction(conf.get())
