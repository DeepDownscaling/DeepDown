import os
import pickle
import hashlib
import numpy as np
import xarray as xr
import pandas as pd
from pyproj import Transformer


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
        The desired longitude bounds of the data ([min, max]) or full longitude array.
    lat_bnds : list
        The desired latitude bounds of the data ([min, max]) or full latitude array.

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
        The desired longitude bounds of the data ([min, max]) or full longitude array.
    lat_bnds : list
        The desired latitude bounds of the data ([min, max]) or full latitude array.

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
    out = out * 1

    return out


def load_data(vars, paths, date_start, date_end, lon_bnds, lat_bnds, levels):
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
    lon_bnds : list
        The desired longitude bounds of the data ([min, max]) or full longitude array.
    lat_bnds : list
        The desired latitude bounds of the data ([min, max]) or full latitude array.
    levels : list
        The levels to extract.

    Returns
    -------
    xarray.Dataset
        The data.
    """
    data = []
    for i_var in range(0, len(vars)):

        dat = get_nc_data(paths[i_var] + '/*nc', date_start, date_end, lon_bnds,
                          lat_bnds)

        if 'level' in list(dat.coords):
            print("Selecting level")
            lev = np.array(dat.level)
            l = [x for x in lev if x in levels]
            dat = dat.sel(level=l)

        if vars[i_var] == 'z':
            dat.z.values = dat.z.values / G

        dat['time'] = pd.DatetimeIndex(dat.time.dt.date)

        data.append(dat)

    return xr.merge(data)


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
    mx = xr.DataArray(a, dims=["time", "lat", "lon"],
                      coords=dict(time=time, lat=lat, lon=lon))
    return mx


def load_target_data(date_start, date_end, path, dump_data_to_pickle=True,
                     path_tmp='../tmp/'):
    """
    Load the target data.

    Parameters
    ----------
    date_start : str
        The starting date ('YYYY-MM-DD').
    date_end : str
        The end date ('YYYY-MM-DD').
    path : str
        The path to the data main directory.
    dump_data_to_pickle : bool
        Whether to dump the data to pickle or not.
    path_tmp : str
        The path to the temporary directory to save pickle files.

    Returns
    -------
    xarray.Dataset
        The target data.
    """

    # Load from pickle
    target_pkl_file = f'{path_tmp}/target_{date_start}_{date_end}.pkl'
    target_loaded_from_pickle = False
    if dump_data_to_pickle and os.path.isfile(target_pkl_file):
        with open(target_pkl_file, 'rb') as f:
            target = pickle.load(f)
            target_loaded_from_pickle = True
            print('Target data loaded from pickle.')

    # Read data from original files
    if not target_loaded_from_pickle:
        print('Extracting target data...')

        pr = get_nc_data(path + '/RhiresD_v2.0_swiss.lv95/*nc', date_start, date_end)
        t_abs = get_nc_data(path + '/TabsD_v2.0_swiss.lv95/*nc', date_start, date_end)
        t_max = get_nc_data(path + '/TmaxD_v2.0_swiss.lv95/*nc', date_start, date_end)
        t_min = get_nc_data(path + '/TminD_v2.0_swiss.lv95/*nc', date_start, date_end)

        # Merge the target data
        target = xr.merge([pr, t_abs, t_max, t_min])

        # Invert lat axis if needed
        if target.N[0].values < target.N[1].values:
            target = target.reindex(N=list(reversed(target.N)))

        # Crop the target data to the final domain
        target = target.sel(E=slice(min(t_abs.E), max(t_abs.E)),
                            N=slice(max(t_abs.N), min(t_abs.N)))

        # Rename coordinates
        target = target.rename({'E': 'x', 'N': 'y'})
        target = target.drop_vars(['lat', 'lon', 'swiss_lv95_coordinates'])

        # Save to pickle
        if dump_data_to_pickle:
            os.makedirs(os.path.dirname(target_pkl_file), exist_ok=True)
            with open(target_pkl_file, 'wb') as f:
                pickle.dump(target, f, protocol=-1)

    return target


def load_input_data(date_start, date_end, path_dem, input_vars, input_paths,
                    levels, resol_low, x_axis, y_axis, dump_data_to_pickle=True,
                    path_tmp='../tmp/'):
    """
    Load the input data.

    Parameters
    ----------
    date_start : str
        The starting date ('YYYY-MM-DD').
    date_end : str
        The end date ('YYYY-MM-DD').
    path_dem : str
        The path to the DEM data.
    input_vars : list
        The variables to load.
    input_paths : list
        The paths to the data.
    levels : list
        The levels to extract.
    resol_low : float
        The resolution of the low resolution data.
    x_axis : numpy.ndarray
        The x coordinates of the final domain.
    y_axis : numpy.ndarray
        The y coordinates of the final domain.
    dump_data_to_pickle : bool
        Whether to dump the data to pickle or not.
    path_tmp : str
        The path to the temporary directory to save pickle files.

    Returns
    -------
    xarray.Dataset
        The input data.
    """

    # Load from pickle
    input_loaded_from_pickle = False
    if dump_data_to_pickle:
        tag = (
                pickle.dumps(input_vars)
                + pickle.dumps(input_paths)
                + pickle.dumps(date_start)
                + pickle.dumps(date_end)
                + pickle.dumps(levels)
                + pickle.dumps(resol_low)
        )

        input_pkl_file = f"{path_tmp}/input_{hashlib.md5(tag).hexdigest()}.pkl"
        if os.path.isfile(input_pkl_file):
            with open(input_pkl_file, "rb") as f:
                input_data = pickle.load(f)
                input_loaded_from_pickle = True
                print("Input data loaded from pickle.")

    # Read data from original files
    if not input_loaded_from_pickle:
        print("Extracting input data...")

        # Load the topography
        topo = xr.open_dataset(path_dem)
        topo = topo.squeeze('band')
        topo = topo.rename({'__xarray_dataarray_variable__': 'topo'})
        topo = topo.drop_vars(['band', 'spatial_ref'])

        # Get extent of the final domain in lat/lon (EPSG:4326) from the original domain in CH1903+ (EPSG:2056)
        x_grid, y_grid = np.meshgrid(x_axis, np.flip(y_axis))
        transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326")
        lat_grid, lon_grid = transformer.transform(x_grid, y_grid)

        # Get the correponding min/max coordinates in the ERA5 grid
        lat_min = np.floor(np.min(lat_grid) * 1 / resol_low) / (1 / resol_low)
        lat_max = np.ceil(np.max(lat_grid) * 1 / resol_low) / (1 / resol_low)
        lon_min = np.floor(np.min(lon_grid) * 1 / resol_low) / (1 / resol_low)
        lon_max = np.ceil(np.max(lon_grid) * 1 / resol_low) / (1 / resol_low)

        # Load the predictors data
        era5_lon = [lon_min, lon_max]
        era5_lat = [lat_min, lat_max]
        inputs = load_data(input_vars, input_paths, date_start, date_end, era5_lon,
                           era5_lat, levels)

        # Interpolate low res data
        # Create a new xarray dataset with the new grid coordinates
        new_data_format = xr.Dataset(coords={'latitude': (('lat', 'lon'), lat_grid),
                                             'longitude': (('lat', 'lon'), lon_grid)})

        # Interpolate the original data onto the new grid
        inputs = inputs.interp(lat=new_data_format.latitude,
                               lon=new_data_format.longitude, method='nearest')

        # Removing duplicate coordinates
        inputs = inputs.drop_vars(['lat', 'lon'])

        # Add the Swiss coordinates
        inputs = inputs.assign_coords(x=(('lat', 'lon'), x_grid),
                                      y=(('lat', 'lon'), y_grid))
        # Rename variables before merging
        inputs = inputs.rename({'lon': 'x', 'lat': 'y'})
        inputs = inputs.drop_vars(['latitude', 'longitude'])

        # Squeeze the 2D coordinates
        x_1d = inputs['x'][0, :]
        y_1d = inputs['y'][:, 0]
        inputs = inputs.assign(x=xr.DataArray(x_1d, dims='x'),
                               y=xr.DataArray(y_1d, dims='y'))

        # Invert y axis if needed
        if inputs.y[0].values < inputs.y[1].values:
            inputs = inputs.reindex(y=list(reversed(inputs.y)))

        # Merge with topo
        input_data = xr.merge([inputs, topo])

        # Save to pickle file
        if dump_data_to_pickle:
            os.makedirs(os.path.dirname(input_pkl_file), exist_ok=True)
            with open(input_pkl_file, 'wb') as f:
                pickle.dump(input_data, f, protocol=-1)

    return input_data
