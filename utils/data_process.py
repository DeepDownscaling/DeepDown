import numpy as np
import xarray as xr
import pandas as pd
import dask
import datetime


# Some constants
G = 9.80665


def rename_dimensions_variables(ds):
    """
    Rename dimensions of the given dataset to homogenize data.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to rename the dimensions of.

    Returns
    -------
    xarray.Dataset
        The dataset with renamed dimensions.
    """
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})

    return ds


def temporal_slice(ds, start, end):
    """
    Slice along the temporal dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to slice.
    start : str
        The start date of the slice.
    end : str
        The end date of the slice.
    
    Returns
    -------
    xarray.Dataset
        The dataset with the temporal slice applied.
    """
    ds = ds.sel(time=slice(start, end))

    if 'time_bnds' in ds.variables:
        ds = ds.drop('time_bnds')

    return ds


def spatial_slice(ds, lon_bnds, lat_bnds):
    """
    Slice along the spatial dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to slice.
    lon_bnds : list
        The longitude bounds of the slice.
    lat_bnds : list
        The latitude bounds of the slice.

    Returns
    -------
    xarray.Dataset
        The dataset with the spatial slice applied.
    """
    if lon_bnds != None:
        ds = ds.sel(lon=slice(min(lon_bnds), max(lon_bnds)))

    if lat_bnds != None:
        if ds.lat[0].values < ds.lat[1].values:
            ds = ds.sel(lat=slice(min(lat_bnds), max(lat_bnds)))
        else:
            ds = ds.sel(lat=slice(max(lat_bnds), min(lat_bnds)))

    return ds


def get_nc_data(files, start, end, lon_bnds=None, lat_bnds=None):
    """
    Extract netCDF data for the given file(s) pattern/path.

    Parameters
    ----------
    files : str or list
        The file(s) pattern/path to extract data from.
    start : str
        The desired start date of the data.
    end : str
        The desired end date of the data.
    lon_bnds : list
        The desired longitude bounds of the data.
    lat_bnds : list
        The desired latitude bounds of the data.

    Returns
    -------
    xarray.Dataset
        The dataset with the extracted data.
    """
    print('Extracting data for the period {} - {}'.format(start, end))
    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = rename_dimensions_variables(ds)
    ds = temporal_slice(ds, start, end)
    ds = spatial_slice(ds, lon_bnds, lat_bnds)

    return ds


def get_era5_data(files, start, end, lon_bnds=None, lat_bnds=None):
    """
    Extract ERA5 data for the given file(s) pattern/path.

    Parameters
    ----------
    files : str or list
        The file(s) pattern/path to extract data from.
    start : str
        The desired start date of the data.
    end : str
        The desired end date of the data.
    lon_bnds : list
        The desired longitude bounds of the data.
    lat_bnds : list
        The desired latitude bounds of the data.

    Returns
    -------
    xarray.Dataset
        The dataset with the extracted data.
    """
    
    return get_nc_data(files, start, end, lon_bnds, lat_bnds)


def precip_exceedance(precip, qt=0.95):
    """
    Computes exceedances of precipitation.

    Parameters
    ----------
    precip : xarray.DataArray
        The precipitation data.
    qt : float
        The quantile to compute the exceedances for.

    Returns
    -------
    xarray.DataArray
        The exceedances of the precipitation data.
    """
    qq = xr.DataArray(precip).quantile(qt, dim='time') 
    out = xr.DataArray(precip > qq)
    out = out*1

    return out


def load_data(vars, paths, date_start, date_end, lons, lats, levels):
    """Load the data.

    Parameters
    ----------
    vars : list
        The variables to load.
    paths : list
        The paths to the data.
    date_start : str
        The starting date.
    date_end : str
        The end date.
    lons : list
        The longitudes to extract.
    lats : list
        The latitudes to extract.
    levels : list
        The levels to extract.

    Returns
    -------
    xarray.Dataset
        The data.
    """
    data = []
    for i_var in range(0, len(vars)):
        
        dat = get_era5_data(paths[i_var] +'*nc', date_start, date_end, lons, lats)

        if 'level' in list(dat.coords): 
            print("Selecting level")
            lev = np.array(dat.level)
            l = [x for x in lev if x in levels]
            dat = dat.sel(level=l)
            
        if vars[i_var] == 'z':
            dat.z.values = dat.z.values/G
            
        dat['time'] = pd.DatetimeIndex(dat.time.dt.date)
    
        data.append(dat)

    ds = xr.merge(data)

    return ds


def convert_to_xarray(a, lat, lon, time):
    """
    Convert a numpy array into a Dataarray.
    
    Parameters
    ----------
    a : numpy.ndarray
        The array to convert.
    lat : numpy.ndarray
        The latitude values.
    lon : numpy.ndarray
        The longitude values.
    time : numpy.ndarray
        The time values.

    Returns
    -------
    xarray.DataArray
        The converted array.
    """
    mx= xr.DataArray(a, dims=["time","lat", "lon"],
                      coords=dict(time = time, lat = lat,lon = lon))
    return mx