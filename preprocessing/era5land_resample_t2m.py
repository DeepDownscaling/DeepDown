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
    data_path = Path(DATA_DIR) / f't2m-{year}.grib'
    ds = xr.open_dataset(data_path, engine='cfgrib')

    # Resample to daily data
    t2m = ds['t2m']
    step = ds['step']

    # Aggregate over the step dimension
    t2m_min = t2m.min(dim='step')
    t2m_max = t2m.max(dim='step')
    t2m_mean = t2m.mean(dim='step')

    # Rename the variables
    t2m_min = t2m_min.rename('t2m_min')
    t2m_max = t2m_max.rename('t2m_max')

    # Drop the first day of the year as it is incomplete
    t = ds['time']
    t2m_min = t2m_min.sel(time=t.values[1:])
    t2m_max = t2m_max.sel(time=t.values[1:])
    t2m_mean = t2m_mean.sel(time=t.values[1:])

    # Save to NetCDF
    output_path_min = Path(OUTPUT_DIR) / f't2m_min-{year}.nc'
    output_path_max = Path(OUTPUT_DIR) / f't2m_max-{year}.nc'
    output_path_mean = Path(OUTPUT_DIR) / f't2m-{year}.nc'
    t2m_min.to_netcdf(output_path_min, format='NETCDF4_CLASSIC')
    t2m_max.to_netcdf(output_path_max, format='NETCDF4_CLASSIC')
    t2m_mean.to_netcdf(output_path_mean, format='NETCDF4_CLASSIC')

    # Close the files
    ds.close()
    t2m_min.close()
    t2m_max.close()
    t2m_mean.close()
