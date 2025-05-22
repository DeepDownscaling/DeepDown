from ibicus.utils import iecdf
import scipy.interpolate
import scipy.special
import scipy.stats
import numpy as np
import SBCK, SBCK.tools


def get_ibicus_var_name(variable_name):
    """Convert variable names to ibicus names."""
    if variable_name == 'tp':
        return 'pr'
    elif variable_name == 't':
        return 'tas'
    elif variable_name == 't_min':
        return 'tasmin'
    elif variable_name == 't_max':
        return 'tasmax'

    raise ValueError(f"Variable {variable_name} not listed for ibicus conversion.")


def prepare_for_ibicus(data_loader, variable_name):
    """Prepare data for ibicus."""
    # Get variable of interest
    data = data_loader.data[variable_name]

    # Convert to numpy array
    data_array = data.values

    # Get units
    data_units = None
    if 'units' in data.attrs:
        data_units = data.attrs['units']

    # Convert units
    if variable_name == 'tp':
        if data_units in ['kg/m^2/s', 'kg m-2 s-1']:
            pass
        elif data_units in ['mm', 'mm/day', 'millimeter', 'millimeters']:
            # mm/day to kg/m^2/s
            data_array /= 86400
        elif data_units in ['m', 'm/day', 'meter', 'meters']:
            # m/day to kg/m^2/s
            data_array /= 86.400
        else:
            raise ValueError(f"Unit {data_units} not listed for {variable_name}.")
    elif variable_name in ['t', 't_min', 't_max']:
        if data_units in ['K', 'kelvin']:
            pass
        elif data_units in ['°C', 'C', 'celsius', 'degree Celsius', 'degC']:
            # Degree Celsius to Kelvin
            data_array += 273.15
        elif data_units in ['degree', 'degrees'] or data_units is None:
            v_mean = np.nanmean(data_array)
            if v_mean < 100:
                # Degree Celsius to Kelvin
                data_array += 273.15
        else:
            raise ValueError(f"Unit {data_units} not listed for {variable_name}.")

    # Replace missing values
    data_array = _replace_missing_values(data_array)

    return data_array


def prepare_for_sbck(data_loader, variable_name):
    """Prepare data for SBCK."""
    # Get variable of interest
    data = data_loader.data[variable_name]

    # Convert to numpy array
    data_array = data.values

    # Get units
    data_units = None
    if 'units' in data.attrs:
        data_units = data.attrs['units']

    data_array = _convert_units(data_array, data_units, variable_name)

    # Replace missing values
    data_array = _replace_missing_values(data_array)

    data_loader.data[variable_name] = (data.dims, data_array)

    return data_loader


def extract_for_sbck(data_loader, variable_name, x, y):
    """Extract data for SBCK."""
    # Get variable of interest
    data = data_loader.data[variable_name]

    # Convert to numpy array
    data_array = data.sel(x=x, y=y).values

    # If contains nans, return None
    if np.isnan(data_array).any() or np.isinf(data_array).any():
        return None

    # Get units
    data_units = None
    if 'units' in data.attrs:
        data_units = data.attrs['units']

    data_array = _convert_units(data_array, data_units, variable_name)

    return data_array


def _convert_units(data_array, data_units, variable_name):
    # Convert units
    if variable_name == 'tp':
        if data_units in ['mm', 'mm/day', 'millimeter', 'millimeters']:
            pass
        elif data_units in ['kg/m^2/s', 'kg m-2 s-1']:
            # kg/m^2/s to mm/day
            data_array *= 86400
        elif data_units in ['m', 'm/day', 'meter', 'meters']:
            # m/day to mm/day
            data_array *= 1000
        else:
            raise ValueError(f"Unit {data_units} not listed for {variable_name}.")
    elif variable_name in ['t', 't_min', 't_max']:
        if data_units in ['°C', 'C', 'celsius', 'degree Celsius', 'degC']:
            pass
        elif data_units in ['K', 'kelvin']:
            # Kelvin to Degree Celsius
            data_array -= 273.15
        elif data_units in ['degree', 'degrees'] or data_units is None:
            v_mean = np.nanmean(data_array)
            if v_mean > 100:
                # Kelvin to Degree Celsius
                data_array -= 273.15
        else:
            raise ValueError(f"Unit {data_units} not listed for {variable_name}.")
    return data_array


def debias_with_sbck(bc_method, input_array_clim, input_array_hist, target_array_hist,
                     **kwargs):
    if bc_method == "QM":
        # Empirical quantile mapping
        bc = SBCK.QM(distY0=SBCK.tools.rv_histogram,
                     distX0=SBCK.tools.rv_histogram,
                     **kwargs)
        bc.fit(target_array_hist, input_array_hist)
    elif bc_method == "RBC":  # Only for comparison
        bc = SBCK.RBC()
        bc.fit(target_array_hist, input_array_hist)
    elif bc_method == "IdBC":  # Only for comparison
        bc = SBCK.IdBC()
        bc.fit(target_array_hist, input_array_hist)
    elif bc_method == "CDFt":
        bc = SBCK.CDFt(**kwargs)
        bc.fit(target_array_hist, input_array_hist, input_array_clim)
    elif bc_method == "OTC":
        bc = SBCK.OTC(**kwargs)
        bc.fit(target_array_hist, input_array_hist)
    elif bc_method == "dOTC":
        bc = SBCK.dOTC(**kwargs)
        bc.fit(target_array_hist, input_array_hist, input_array_clim)
    elif bc_method == "ECBC":
        bc = SBCK.ECBC(**kwargs)
        bc.fit(target_array_hist, input_array_hist)
    elif bc_method == "QMrs":
        bc = SBCK.QMrs(**kwargs)
        bc.fit(target_array_hist, input_array_hist)
    elif bc_method == "R2D2":
        bc = SBCK.R2D2(**kwargs)
        bc.fit(target_array_hist, input_array_hist, input_array_clim)
    elif bc_method == "QDM":
        bc = SBCK.QDM(**kwargs)
        bc.fit(target_array_hist, input_array_hist, input_array_clim)
    elif bc_method == "MBCn":
        bc = SBCK.MBCn(**kwargs)
        bc.fit(target_array_hist, input_array_hist, input_array_clim)
    elif bc_method == "MRec":
        bc = SBCK.MRec(**kwargs)
        bc.fit(target_array_hist, input_array_hist, input_array_clim)
    elif bc_method == "TSMBC":
        bc = SBCK.TSMBC(lag=30, **kwargs)
        bc.fit(target_array_hist, input_array_hist)
    elif bc_method == "dTSMBC":
        bc = SBCK.dTSMBC(lag=30, **kwargs)
        bc.fit(target_array_hist, input_array_hist, input_array_clim)
    elif bc_method == "AR2D2":
        bc = SBCK.AR2D2(**kwargs)
        bc.fit(target_array_hist, input_array_hist, input_array_clim)
    else:
        raise ValueError(f"Unknown bias correction method: {bc_method}")
    debiased_hist_ts = bc.predict(input_array_hist)
    debiased_clim_ts = bc.predict(input_array_clim)

    return debiased_clim_ts, debiased_hist_ts


def _replace_missing_values(x):
    """Replace missing values."""
    mask_missing = np.logical_or(np.isnan(x), np.isinf(x))
    mask_missing = mask_missing.any(axis=0)
    if mask_missing.sum() == 0:
        return x

    # Calculate the average of the other cells
    x_mean = np.nanmean(x, axis=(1, 2))

    # Replace missing values by the average of the other cells
    for i in range(mask_missing.shape[0]):
        for j in range(mask_missing.shape[1]):
            if mask_missing[i, j]:
                x[:, i, j] = x_mean

    return x
