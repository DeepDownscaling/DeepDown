from ibicus.utils import iecdf
import scipy.interpolate
import scipy.special
import scipy.stats
import numpy as np


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

    # Replace missing values
    data_array = _replace_missing_values(data_array)

    return data_array


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
