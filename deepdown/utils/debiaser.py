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
        elif data_units in ['°C', 'C', 'celsius', 'degree Celsius']:
            # Degree Celsius to Kelvin
            data_array += 273.15
        elif data_units in ['degree', 'degrees'] or data_units is None:
            v_mean = np.nanmean(data_array)
            if v_mean < 100:
                # Degree Celsius to Kelvin
                data_array += 273.15
        else:
            raise ValueError(f"Unit {data_units} not listed for {variable_name}.")

    # Impute values
    if variable_name == 'tp':
        iecdf_method = 'averaged_inverted_cdf'  # 'closest_observation'
    else:
        iecdf_method = 'linear'
    data_array = _impute_values(data_array, iecdf_method=iecdf_method)

    return data_array


def _get_mask_for_values_to_impute(x):
    return np.logical_or(np.isnan(x), np.isinf(x))


def _impute_values(x, iecdf_method='linear'):
    """
    See https://github.com/ecmwf-projects/ibicus/blob/b4cc9194047164f3fd61a5f6cfe95631797f6282/ibicus/debias/_isimip.py#L519
    """
    mask_values_to_impute = _get_mask_for_values_to_impute(x)
    mask_valid_values = np.logical_not(mask_values_to_impute)
    valid_values = x[mask_valid_values]

    # If all values are invalid raise error
    if valid_values.size == 0:
        raise ValueError(
            "Step2: Imputation not possible because all values are invalid in the given month/window."
        )

    # If only one valid value exist insert this one at locations of invalid values
    if valid_values.size == 1:
        x[mask_values_to_impute] = valid_values[0]
        return x

    # Sample invalid values from the valid ones
    sampled_values = iecdf(
        x=valid_values,
        p=np.random.random(size=mask_values_to_impute.sum()),
        method=iecdf_method,
    )

    # Compute a backsort of how sorted valid values would be reinserted
    indices_valid_values = np.where(mask_valid_values)[0]
    backsort_sorted_valid_values = np.argsort(np.argsort(valid_values))
    interpolated_backsort_valid_values = scipy.interpolate.interp1d(
        indices_valid_values, backsort_sorted_valid_values, fill_value="extrapolate"
    )

    # Use this backsort to compute where sorted sampled values for invalid values would be reinserted
    backsort_invalid_values = np.argsort(
        np.argsort(
            interpolated_backsort_valid_values(np.where(mask_values_to_impute)[0])
        )
    )

    # Reinserted sorted sampled values into the locations
    x[mask_values_to_impute] = np.sort(sampled_values)[backsort_invalid_values]

    return x
