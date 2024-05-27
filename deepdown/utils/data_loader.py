import os
import pickle
import hashlib
import numpy as np
import xarray as xr
import pandas as pd
from pyproj import Transformer


class DataLoader:
    def __init__(self, lon_bnds=None, lat_bnds=None,
                 path_tmp='../tmp/'):
        """
        Initialize the DataLoader.

        Parameters
        ----------
        lon_bnds : list
            The desired longitude bounds ([min, max]) or full longitude array.
        lat_bnds : list
            The desired latitude bounds ([min, max]) or full latitude array.
        path_tmp : str
            The path to the temporary directory to save pickle files.
        """
        self.data = None
        self.lon_bnds = lon_bnds
        self.lat_bnds = lat_bnds
        self.path_tmp = path_tmp

    def load(self, date_start, date_end, paths, dump_data_to_pickle=True):
        """
        Load the target data.

        Parameters
        ----------
        date_start : str
            The desired start date ('YYYY-MM-DD').
        date_end : str
            The desired end date ('YYYY-MM-DD').
        paths : list
            The paths to the data.
        dump_data_to_pickle : bool
            Whether to dump the data to pickle or not.

        Returns
        -------
        xarray.Dataset
            The target data.
        """
        # Load from pickle
        pkl_filename = self._get_pickle_filename(paths, date_start, date_end)
        if dump_data_to_pickle and os.path.isfile(pkl_filename):
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f)
                print('Data loaded from pickle.')
                return data

        # Read data from original files
        print('Extracting data from files...')
        data = []
        for i_var in range(0, len(paths)):
            dat = self._get_nc_data(paths[i_var] + '/*nc', date_start, date_end)
            data.append(dat)

        # Extract the min/max coordinates of the common domain
        min_x = max([ds.x.min() for ds in data])
        max_x = min([ds.x.max() for ds in data])
        min_y = max([ds.y.min() for ds in data])
        max_y = min([ds.y.max() for ds in data])

        # Convert to xarray
        self.data = xr.merge(data)

        # Invert lat axis if needed
        if self.data.y[0].values < self.data.y[1].values:
            self.data = self.data.reindex(y=list(reversed(self.data.y)))

        # Crop the target data to the final domain
        self.data = self.data.sel(x=slice(min_x, max_x),
                                  y=slice(max_y, min_y))

        # Drop unnecessary variables
        self.data = self.data.drop_vars(['lat', 'lon', 'swiss_lv95_coordinates'],
                                        errors='ignore')

        # Save to pickle
        if dump_data_to_pickle:
            os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, 'wb') as f:
                pickle.dump(self.data, f, protocol=-1)

        return self.data

    def _get_nc_data(self, files, date_start, date_end):
        """
        Extract netCDF data for the given file(s) pattern/path.

        Parameters
        ----------
        files : str or list
            The file(s) pattern/path to extract data from.
        date_start : str
            The desired start date ('YYYY-MM-DD').
        date_end : str
            The desired end date ('YYYY-MM-DD').

        Returns
        -------
        xarray.Dataset
            The extracted data.
        """
        print('Extracting data for {} - {}'.format(date_start, date_end))
        ds = xr.open_mfdataset(files, combine='by_coords')
        ds = self._rename_dimensions_variables(ds)
        ds = self._temporal_slice(ds, date_start, date_end)
        ds = self._spatial_slice(ds)

        return ds

    def _get_pickle_filename(self, paths, date_start, date_end):
        """
        Get the pickle filename for the given paths.

        Parameters
        ----------
        paths : list
            The paths to the data.
        date_start : str
            The desired start date ('YYYY-MM-DD').
        date_end : str
            The desired end date ('YYYY-MM-DD').

        Returns
        -------
        str
            The pickle filename.
        """
        tag = hashlib.md5(
            pickle.dumps(paths)
            + pickle.dumps(date_start)
            + pickle.dumps(date_end)
            + pickle.dumps(self.lon_bnds)
            + pickle.dumps(self.lat_bnds)
        ).hexdigest()

        return f'{self.path_tmp}/data_{tag}.pkl'

    @staticmethod
    def _rename_dimensions_variables(ds):
        """
        Rename dimensions of the given dataset to homogenize data.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to rename dimensions and variables.

        Returns
        -------
        xarray.Dataset
            The dataset with renamed dimensions and variables.
        """
        # Rename dimensions
        if 'latitude' in ds.dims:
            ds = ds.rename({'latitude': 'lat'})
        if 'longitude' in ds.dims:
            ds = ds.rename({'longitude': 'lon'})
        if 'E' in ds.dims:
            ds = ds.rename({'E': 'x'})
        if 'N' in ds.dims:
            ds = ds.rename({'N': 'y'})

        # Rename variables
        if 'RhiresD' in ds.variables:
            ds = ds.rename({'RhiresD': 'tp'})
        if 'TabsD' in ds.variables:
            ds = ds.rename({'TabsD': 't'})
        if 'TmaxD' in ds.variables:
            ds = ds.rename({'TmaxD': 't_max'})
        if 'TminD' in ds.variables:
            ds = ds.rename({'TminD': 't_min'})

        return ds

    @staticmethod
    def _temporal_slice(ds, date_start, date_end):
        """
        Slice along the temporal dimension.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to slice.
        date_start : str
            The desired start date ('YYYY-MM-DD').
        date_end : str
            The desired end date ('YYYY-MM-DD').

        Returns
        -------
        xarray.Dataset
            The dataset with the temporal slice.
        """
        ds = ds.sel(time=slice(date_start, date_end))

        if 'time_bnds' in ds.variables:
            ds = ds.drop('time_bnds')

        return ds

    def _spatial_slice(self, ds):
        """
        Slice along the spatial dimension.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to slice.

        Returns
        -------
        xarray.Dataset
            The dataset with the spatial slice.
        """
        if self.lon_bnds is not None:
            ds = ds.sel(lon=slice(min(self.lon_bnds), max(self.lon_bnds)))

        if self.lat_bnds is not None:
            if ds.lat[0].values < ds.lat[1].values:
                ds = ds.sel(lat=slice(min(self.lat_bnds), max(self.lat_bnds)))
            else:
                ds = ds.sel(lat=slice(max(self.lat_bnds), min(self.lat_bnds)))

        return ds







def load_data(paths, date_start, date_end, lon_bnds, lat_bnds, levels):
    """Load the data.

    Parameters
    ----------
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
    for i_var in range(0, len(paths)):

        dat = get_nc_data(paths[i_var] + '/*nc', date_start, date_end, lon_bnds,
                          lat_bnds)

        if 'level' in list(dat.coords):
            print("Selecting level")
            lev = np.array(dat.level)
            l = [x for x in lev if x in levels]
            dat = dat.sel(level=l)

        if 'z' in dat.variables:
            dat.z.values = dat.z.values / 9.80665

        dat['time'] = pd.DatetimeIndex(dat.time.dt.date)

        data.append(dat)

    return xr.merge(data)

def load_input_data(date_start, date_end, paths, levels, resol_low,
                    x_axis=None, y_axis=None, path_dem=None, dump_data_to_pickle=True,
                    path_tmp='../tmp/'):
    """
    Load the input data.

    Parameters
    ----------
    date_start : str
        The starting date ('YYYY-MM-DD').
    date_end : str
        The end date ('YYYY-MM-DD').
    paths : list
        The paths to the data.
    levels : list
        The levels to extract.
    resol_low : float
        The resolution of the low resolution data.
    x_axis : numpy.ndarray|xr.DataArray
        The x coordinates of the final domain.
        If None, the data will not be interpolated.
    y_axis : numpy.ndarray|xr.DataArray
        The y coordinates of the final domain.
        If None, the data will not be interpolated.
    path_dem : str
        The path to the DEM data. If None, the topography will not be added.
    dump_data_to_pickle : bool
        Whether to dump the data to pickle or not.
    path_tmp : str
        The path to the temporary directory to save pickle files.

    Returns
    -------
    xarray.Dataset
        The input data.
    """
    if x_axis is not None and y_axis is not None:
        if isinstance(x_axis, xr.DataArray):
            x_axis = x_axis.values
        if isinstance(y_axis, xr.DataArray):
            y_axis = y_axis.values

    # Load from pickle
    if dump_data_to_pickle:
        tag = hashlib.md5(
            pickle.dumps(paths)
            + pickle.dumps(date_start)
            + pickle.dumps(date_end)
            + pickle.dumps(levels)
            + pickle.dumps(resol_low)
            + pickle.dumps(x_axis)
            + pickle.dumps(y_axis)
        ).hexdigest()

        input_pkl_file = f"{path_tmp}/input_{tag}.pkl"
        if os.path.isfile(input_pkl_file):
            with open(input_pkl_file, "rb") as f:
                input_data = pickle.load(f)
                print("Input data loaded from pickle.")
                return input_data

    # Read data from original files
    print("Extracting input data...")

    # Load the topography
    topo = None
    if path_dem is not None:
        topo = xr.open_dataset(path_dem)
        topo = topo.squeeze('band')
        if '__xarray_dataarray_variable__' in topo.variables:
            topo = topo.rename({'__xarray_dataarray_variable__': 'topo'})
        topo = topo.drop_vars(['band', 'spatial_ref'])

    # Get extent of the final domain in lat/lon (EPSG:4326) from the original
    # domain in CH1903+ (EPSG:2056)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326")
    lat_grid, lon_grid = transformer.transform(x_grid, y_grid)

    # Get the corresponding min/max coordinates in the ERA5 grid
    lat_min = np.floor(np.min(lat_grid) * 1 / resol_low) / (1 / resol_low)
    lat_max = np.ceil(np.max(lat_grid) * 1 / resol_low) / (1 / resol_low)
    lon_min = np.floor(np.min(lon_grid) * 1 / resol_low) / (1 / resol_low)
    lon_max = np.ceil(np.max(lon_grid) * 1 / resol_low) / (1 / resol_low)

    # Load the predictors data
    lons = [lon_min, lon_max]
    lats = [lat_min, lat_max]
    inputs = load_data(paths, date_start, date_end, lons, lats, levels)

    # Interpolate low res data
    # Create a new xarray dataset with the new grid coordinates
    new_data_format = xr.Dataset(coords={'latitude': (('lat', 'lon'), lat_grid),
                                         'longitude': (('lat', 'lon'), lon_grid)})

    # Interpolate the original input data onto the new grid
    # (increase in resolution => nearest neighbor interpolation is OK)
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
    if topo is not None:
        inputs = xr.merge([inputs, topo])

    # Save to pickle file
    if dump_data_to_pickle:
        os.makedirs(os.path.dirname(input_pkl_file), exist_ok=True)
        with open(input_pkl_file, 'wb') as f:
            pickle.dump(inputs, f, protocol=-1)

    return inputs
