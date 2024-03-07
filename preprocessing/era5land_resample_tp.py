# This script converts the hourly precipitation data from the ECMWF ERA5-Land
# reanalysis to daily data. The original data is stored in GRIB format and
# is resampled to daily data and stored in NetCDF format.
# Dependencies: xarray, netCDF4, cfgrib (which will require ecCodes to be installed)

import xarray as xr
from pathlib import Path

DATA_DIR = '/storage/homefs/horton/hydro/data/ERA5_Land/Switzerland-hourly-original'
OUTPUT_DIR = '/storage/homefs/horton/hydro/data/ERA5_Land/Switzerland-daily'
Y_START = 1960
Y_END = 2023

for year in range(Y_START, Y_END + 1):
    print(f'Processing year {year}')

    # Read the original data
    data_path = Path(DATA_DIR) / f'prec-{year}.grib'
    ds = xr.open_dataset(data_path, engine='cfgrib')

    # Resample to daily data
    precipitation = ds['tp']
    step = ds['step']
    precipitation = precipitation.sel(step=step.values[-1])
    
    # Remove the step dimension
    precipitation = precipitation.drop_vars('step')

    # Drop the last day of the year as it is incomplete
    t = ds['time']
    precipitation = precipitation.sel(time=t.values[0:-1])

    # Save to NetCDF
    output_path = Path(OUTPUT_DIR) / f'tp-{year}.nc'
    precipitation.to_netcdf(output_path, format='NETCDF4_CLASSIC')

    # Close the files
    ds.close()
    precipitation.close()
