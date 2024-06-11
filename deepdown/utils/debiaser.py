# From https://github.com/ecmwf-projects/ibicus/blob/b4cc9194047164f3fd61a5f6cfe95631797f6282/ibicus/debias/_isimip.py#L519
from ibicus.utils import iecdf
import scipy.interpolate
import scipy.special
import scipy.stats
import numpy as np


def _step_to_get_mask_for_values_to_impute(x):
    return np.logical_or(np.isnan(x), np.isinf(x))


def _step_to_impute_values(x, iecdf_method='linear'):
    """See documentation"""
    mask_values_to_impute = _step_to_get_mask_for_values_to_impute(x)
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
