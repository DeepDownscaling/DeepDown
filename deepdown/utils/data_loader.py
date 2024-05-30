import os
import pickle
import hashlib
import numpy as np
import xarray as xr
import pandas as pd
from pyproj import Transformer


class DataLoader:
    def __init__(self, x_bnds=None, y_bnds=None, path_tmp='../tmp/',
                 dump_data_to_pickle=True):
        """
        Initialize the DataLoader.

        Parameters
        ----------
        x_bnds : list
            The desired bounds for the x axis ([min, max]) or full x array.
        y_bnds : list
            The desired bounds for the y axis ([min, max]) or full y array.
        path_tmp : str
            The path to the temporary directory to save pickle files.
        dump_data_to_pickle : bool
            Whether to save the data to a pickle file.
        """
        self.data = None
        self.paths = None
        self.x_bnds = x_bnds
        self.y_bnds = y_bnds
        self.path_tmp = path_tmp
        self.dump_data_to_pickle = dump_data_to_pickle

    def load(self, date_start, date_end, paths):
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

        Returns
        -------
        xarray.Dataset
            The data array.
        """
        self.paths = paths

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

        # Invert y axis if needed
        if self.data.y[0].values < self.data.y[1].values:
            self.data = self.data.reindex(y=list(reversed(self.data.y)))

        # Crop the target data to the final domain
        self.data = self.data.sel(x=slice(min_x, max_x),
                                  y=slice(max_y, min_y))

        # Load from pickle
        pkl_filename = self._get_pickle_filename()
        if self.dump_data_to_pickle and os.path.isfile(pkl_filename):
            with open(pkl_filename, 'rb') as f:
                self.data = pickle.load(f)
                print('Data loaded from pickle.')
                return self.data

        # Save to pickle
        if self.dump_data_to_pickle:
            os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, 'wb') as f:
                pickle.dump(self.data, f, protocol=-1)

        return self.data

    def load_topography(self, path_dem=None):
        """
        Load the topography data.

        Parameters
        ----------
        path_dem : str
            The path to the DEM data.
        """
        if path_dem is None:
            return

        # Load the topography
        topo = xr.open_dataset(path_dem)
        topo = topo.squeeze('band')
        if '__xarray_dataarray_variable__' in topo.variables:
            topo = topo.rename({'__xarray_dataarray_variable__': 'topo'})
        topo = topo.drop_vars(['band', 'spatial_ref'])

        # Extract the min/max coordinates of the common domain
        min_x = self.data.x.min()
        max_x = self.data.x.max()
        min_y = self.data.y.min()
        max_y = self.data.y.max()

        # Crop the topography data to the final domain
        topo = topo.sel(x=slice(min_x, max_x),
                        y=slice(max_y, min_y))

        self.data = xr.merge([self.data, topo])

    def regrid(self, x_axis, y_axis, crs_from='EPSG:2056', crs_to='EPSG:4326',
               method='nearest'):
        """
        Regrid the data to the given axes.

        Parameters
        ----------
        x_axis : numpy.ndarray|xr.DataArray
            The x coordinates of the final domain.
        y_axis : numpy.ndarray|xr.DataArray
            The y coordinates of the final domain.
        crs_from : str
            The original CRS of the data.
        crs_to : str
            The desired CRS of the data.
        method : str
            The interpolation method. Nearest neighbour is OK for an increase in
            resolution. Other options: 'linear', 'cubic'.
        """
        if isinstance(x_axis, xr.DataArray):
            x_axis = x_axis.values
        if isinstance(y_axis, xr.DataArray):
            y_axis = y_axis.values

        # Load from pickle
        pkl_filename = self._get_regridded_pickle_filename(
            x_axis, y_axis, crs_from, crs_to, method)
        if self.dump_data_to_pickle and os.path.isfile(pkl_filename):
            with open(pkl_filename, 'rb') as f:
                self.data = pickle.load(f)
                print('Regridded data loaded from pickle.')
                return

        # Get extent of the final domain in desired projection from the original one
        x_dest_grid, y_dest_grid = np.meshgrid(x_axis, y_axis)
        transformer = Transformer.from_crs(crs_from, crs_to)
        y_orig_grid, x_orig_grid = transformer.transform(x_dest_grid, y_dest_grid)

        # Get the corresponding min/max coordinates in the original grid
        y_min = np.floor(np.min(y_orig_grid))
        y_max = np.ceil(np.max(y_orig_grid))
        x_min = np.floor(np.min(x_orig_grid))
        x_max = np.ceil(np.max(x_orig_grid))
        self.x_bnds = [x_min, x_max]
        self.y_bnds = [y_min, y_max]

        # Create a new xarray dataset with the new grid coordinates
        new_data_format = xr.Dataset(coords={'y_tmp': (('y2', 'x2'), y_orig_grid),
                                             'x_tmp': (('y2', 'x2'), x_orig_grid)})

        # Interpolate the original input data onto the new grid
        self.data = self.data.interp(y2=new_data_format.y_tmp,
                                     x2=new_data_format.x_tmp, method=method)

        # Removing duplicate coordinates
        self.data = self.data.drop_vars(['y2', 'x2'])

        # Add the new coordinates
        self.data = self.data.assign_coords(x=(('y2', 'x2'), x_dest_grid),
                                            y=(('y2', 'x2'), y_dest_grid))
        # Rename variables before merging
        self.data = self.data.rename({'x2': 'x', 'y2': 'y'})
        self.data = self.data.drop_vars(['y_tmp', 'x_tmp'])

        # Squeeze the 2D coordinates
        x_1d = self.data['x'][0, :]
        y_1d = self.data['y'][:, 0]
        self.data = self.data.assign(x=xr.DataArray(x_1d, dims='x'),
                                     y=xr.DataArray(y_1d, dims='y'))

        # Save to pickle
        if self.dump_data_to_pickle:
            os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, 'wb') as f:
                pickle.dump(self.data, f, protocol=-1)

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
        ds = self._remove_unused_variables(ds)
        ds = self._temporal_slice(ds, date_start, date_end)
        ds = self._spatial_slice(ds)

        return ds

    def _get_pickle_filename(self):
        """
        Get the pickle filename for the given paths.

        Returns
        -------
        str
            The pickle filename.
        """
        tag = hashlib.md5(
            pickle.dumps(self.paths)
            + pickle.dumps(self.data.shape)
            + pickle.dumps(self.data.x)
            + pickle.dumps(self.data.y)
            + pickle.dumps(self.data.time[0].values)
            + pickle.dumps(self.data.time[-1].values)
            + pickle.dumps(self.x_bnds)
            + pickle.dumps(self.y_bnds)
        ).hexdigest()

        return f'{self.path_tmp}/data_{tag}.pkl'

    def _get_regridded_pickle_filename(self, x_axis, y_axis, crs_from, crs_to, method):
        """
        Get the pickle filename for the regridded data.

        Parameters
        ----------
        x_axis : numpy.ndarray
            The x coordinates of the final domain.
        y_axis : numpy.ndarray
            The y coordinates of the final domain.
        crs_from : str
            The original CRS of the data.
        crs_to : str
            The desired CRS of the data.
        method : str
            The interpolation method.

        Returns
        -------
        str
            The pickle filename.
        """
        tag = hashlib.md5(
            pickle.dumps(self.paths)
            + pickle.dumps(self.data.shape)  # property of the original data
            + pickle.dumps(self.data.x)  # property of the original data
            + pickle.dumps(self.data.y)  # property of the original data
            + pickle.dumps(self.data.time[0].values)  # property of the original data
            + pickle.dumps(self.data.time[-1].values)  # property of the original data
            + pickle.dumps(x_axis)  # axis of the final data
            + pickle.dumps(y_axis)  # axis of the final data
            + pickle.dumps(crs_from)
            + pickle.dumps(crs_to)
            + pickle.dumps(method)
        ).hexdigest()

        return f'{self.path_tmp}/data_regridded_{tag}.pkl'

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
        if 'longitude' in ds.dims:
            ds = ds.rename({'longitude': 'x'})
        if 'latitude' in ds.dims:
            ds = ds.rename({'latitude': 'y'})
        if 'lon' in ds.dims:
            ds = ds.rename({'lon': 'x'})
        if 'lat' in ds.dims:
            ds = ds.rename({'lat': 'y'})
        if 'E' in ds.dims:
            ds = ds.rename({'E': 'x'})
            ds = ds.drop_vars(['lon'], errors='ignore')
        if 'N' in ds.dims:
            ds = ds.rename({'N': 'y'})
            ds = ds.drop_vars(['lat'], errors='ignore')

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
    def _remove_unused_variables(ds):
        """
        Remove unused variables from the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to remove unused variables.

        Returns
        -------
        xarray.Dataset
            The dataset without unused variables.
        """
        vars_to_remove = ['swiss_lv95_coordinates', 'time_bnds']
        ds = ds.drop_vars(vars_to_remove, errors='ignore')

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
        ds.time = pd.to_datetime(ds.time.values)
        ds = ds.sel(time=slice(date_start, date_end))

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
        if self.x_bnds is not None:
            x_min = min(self.x_bnds)
            x_min = ds.x.where(ds.x <= x_min, drop=True).max().item()
            x_max = max(self.x_bnds)
            x_max = ds.x.where(ds.x >= x_max, drop=True).min().item()

            ds = ds.sel(x=slice(x_min, x_max))

        if self.y_bnds is not None:
            y_min = min(self.y_bnds)
            y_min = ds.y.where(ds.y <= y_min, drop=True).max().item()
            y_max = max(self.y_bnds)
            y_max = ds.y.where(ds.y >= y_max, drop=True).min().item()

            if ds.y[0].values < ds.y[1].values:
                ds = ds.sel(y=slice(y_min, y_max))
            else:
                ds = ds.sel(y=slice(y_max, y_min))

        return ds
