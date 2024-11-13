import os
import pickle
import hashlib
import dask
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

    def load(self, date_start, date_end, paths, load_in_memory=False, x_min=None,
             x_max=None, y_min=None, y_max=None):
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
        load_in_memory : bool
            Whether to load the data in memory.
        x_min : float
            The minimum x coordinate.
        x_max : float
            The maximum x coordinate.
        y_min : float
            The minimum y coordinate.
        y_max : float
            The maximum y coordinate.

        Returns
        -------
        xarray.Dataset
            The data array.
        """
        self.paths = paths

        if x_min is not None and x_max is not None:
            self.x_bnds = [x_min, x_max]
        if y_min is not None and y_max is not None:
            self.y_bnds = [y_min, y_max]

        # Load from pickle
        pkl_filename = self._get_pickle_filename(date_start, date_end)
        if self.dump_data_to_pickle and os.path.isfile(pkl_filename):
            with open(pkl_filename, 'rb') as f:
                self.data = pickle.load(f)
                print('Data loaded from pickle.')
                return self.data

        # Read data from original files
        print('Extracting data from files...')
        data = []
        for i_var in range(0, len(paths)):
            dat = self._get_nc_data(paths[i_var] + '/*nc', date_start, date_end)
            data.append(dat)

        # Extract the min/max coordinates of the common domain
        if x_min is None and x_max is None:
            x_min = float(max([ds.x.min() for ds in data]).values)
            x_max = float(min([ds.x.max() for ds in data]).values)
        if y_min is None and y_max is None:
            y_min = float(max([ds.y.min() for ds in data]).values)
            y_max = float(min([ds.y.max() for ds in data]).values)

        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            # Convert to xarray
            self.data = xr.merge(data)

        # Invert y axis if needed
        if self.data.y[0].values < self.data.y[1].values:
            self.data = self.data.reindex(y=list(reversed(self.data.y)))

        # Crop the target data to the final domain
        self.data = self.data.sel(x=slice(x_min, x_max),
                                  y=slice(y_max, y_min))

        if load_in_memory:
            self.data.load()

        # Save to pickle
        if self.dump_data_to_pickle:
            os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, 'wb') as f:
                pickle.dump(self.data, f, protocol=-1)

        return self.data

    def select_domain(self, x_min, x_max, y_min, y_max):
        """
        Crop the data to the given domain.

        Parameters
        ----------
        x_min : float
            The minimum x coordinate.
        x_max : float
            The maximum x coordinate.
        y_min : float
            The minimum y coordinate.
        y_max : float
            The maximum y coordinate.
        """
        self.data = self.data.sel(x=slice(x_min, x_max),
                                  y=slice(y_max, y_min))

    def select_period(self, date_start, date_end):
        """
        Slice along the temporal dimension.

        Parameters
        ----------
        date_start : str
            The desired start date ('YYYY-MM-DD').
        date_end : str
            The desired end date ('YYYY-MM-DD').
        """
        self.data = self.data.sel(time=slice(date_start, date_end))

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

    def coarsen(self, x_axis, y_axis, from_proj='WGS84', to_proj='CH1903+',
                load_in_memory=False):
        """
        Coarsen the data to the given axes.

        Parameters
        ----------
        x_axis : numpy.ndarray|xr.DataArray
            The x coordinates of the final domain.
        y_axis : numpy.ndarray|xr.DataArray
            The y coordinates of the final domain.
        from_proj : str
            The original projection of the data (as EPSG or projection name).
        to_proj : str
            The desired projection of the data (as EPSG or projection name).
        load_in_memory : bool
            Whether to load the data in memory.
        """
        if isinstance(x_axis, xr.DataArray):
            x_axis = x_axis.values
        if isinstance(y_axis, xr.DataArray):
            y_axis = y_axis.values

        # Convert the projection to EPSG format
        from_proj = self._proj_to_epsg(from_proj)
        to_proj = self._proj_to_epsg(to_proj)

        # Load from pickle
        pkl_filename = self._get_regridded_pickle_filename(
            x_axis, y_axis, from_proj, to_proj, 'coarsen')
        if self.dump_data_to_pickle and os.path.isfile(pkl_filename):
            with open(pkl_filename, 'rb') as f:
                self.data = pickle.load(f)
                print('Regridded data loaded from pickle.')
                return

        # Get x/y axes in the desired projection for a high-res grid +- compatible with
        # the coarsening (multiple of the final grid), but not larger
        x_pts = int(np.round(len(self.data.x) / x_axis.size) * x_axis.size)
        x_axis_hi = np.linspace(x_axis[0], x_axis[-1], x_pts)
        y_pts = int(np.round(len(self.data.y) / y_axis.size) * y_axis.size)
        y_axis_hi = np.linspace(y_axis[0], y_axis[-1], y_pts)

        # Get extent in desired projection from the original one
        x_dest_grid_hi, y_dest_grid_hi = np.meshgrid(x_axis_hi, y_axis_hi)
        transformer = Transformer.from_crs(to_proj, from_proj, always_xy=True)
        x_orig_grid_hi, y_orig_grid_hi = transformer.transform(
            x_dest_grid_hi, y_dest_grid_hi)

        # Create a new xarray dataset with the new grid coordinates
        new_data_format_hi = xr.Dataset(
            coords={'y_tmp': (('y2', 'x2'), y_orig_grid_hi),
                    'x_tmp': (('y2', 'x2'), x_orig_grid_hi)})

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            self.data = self.data.interp(
                y=new_data_format_hi.y_tmp,
                x=new_data_format_hi.x_tmp, method='linear')

            # Coarsen the data
            coarsen_x = int(x_orig_grid_hi.shape[1] / x_axis.size)
            coarsen_y = int(y_orig_grid_hi.shape[0] / y_axis.size)
            self.data = self.data.coarsen(y2=coarsen_y, x2=coarsen_x,
                                          boundary='trim').mean()

        # Removing duplicate coordinates
        self.data = self.data.drop_vars(['y', 'x'])

        # Add the new coordinates
        x_dest_grid, y_dest_grid = np.meshgrid(x_axis, y_axis)
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

        if load_in_memory:
            self.data.load()

        # Save to pickle
        if self.dump_data_to_pickle:
            os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, 'wb') as f:
                pickle.dump(self.data, f, protocol=-1)

    def interpolate(self, x_axis, y_axis, from_proj='WGS84', to_proj='CH1903+',
                    method='nearest', load_in_memory=False):
        """
        Interpolate the data to the given axes.

        Parameters
        ----------
        x_axis : numpy.ndarray|xr.DataArray
            The x coordinates of the final domain.
        y_axis : numpy.ndarray|xr.DataArray
            The y coordinates of the final domain.
        from_proj : str
            The original projection of the data (as EPSG or projection name).
        to_proj : str
            The desired projection of the data (as EPSG or projection name).
        method : str
            The interpolation method. Options are: "linear", "nearest", "zero",
            "slinear", "quadratic", "cubic", "polynomial".
        load_in_memory : bool
            Whether to load the data in memory.
        """
        if isinstance(x_axis, xr.DataArray):
            x_axis = x_axis.values
        if isinstance(y_axis, xr.DataArray):
            y_axis = y_axis.values

        # Convert the projection to EPSG format
        from_proj = self._proj_to_epsg(from_proj)
        to_proj = self._proj_to_epsg(to_proj)

        # Load from pickle
        pkl_filename = self._get_regridded_pickle_filename(
            x_axis, y_axis, from_proj, to_proj, method)
        if self.dump_data_to_pickle and os.path.isfile(pkl_filename):
            with open(pkl_filename, 'rb') as f:
                self.data = pickle.load(f)
                print('Regridded data loaded from pickle.')
                return

        # Get extent of the final domain in desired projection from the original one
        x_dest_grid, y_dest_grid = np.meshgrid(x_axis, y_axis)
        transformer = Transformer.from_crs(to_proj, from_proj, always_xy=True)  # reverted
        x_orig_grid, y_orig_grid = transformer.transform(x_dest_grid, y_dest_grid)

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
        self.data = self.data.interp(y=new_data_format.y_tmp,
                                     x=new_data_format.x_tmp, method=method)

        # Removing duplicate coordinates
        self.data = self.data.drop_vars(['y', 'x'])

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

        if load_in_memory:
            self.data.load()

        # Save to pickle
        if self.dump_data_to_pickle:
            os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, 'wb') as f:
                pickle.dump(self.data, f, protocol=-1)

    @staticmethod
    def _proj_to_epsg(proj):
        """
        Convert the projection to the EPSG format.
        """
        if 'EPSG' in proj:
            return proj

        if proj in ['CH1903', 'CH1903+', 'CH1903_LV95']:
            return 'EPSG:2056'
        elif proj in ['WGS84', 'WGS_84']:
            return 'EPSG:4326'

        raise ValueError('Unknown projection for from_proj.')

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

    def _get_pickle_filename(self, date_start, date_end):
        """
        Get the pickle filename for the given paths.

        Returns
        -------
        str
            The pickle filename.
        date_start : str
            The desired start date ('YYYY-MM-DD').
        date_end : str
            The desired end date ('YYYY-MM-DD').
        """
        tag = hashlib.md5(
            pickle.dumps(self.paths)
            + pickle.dumps(self.x_bnds)
            + pickle.dumps(self.y_bnds)
            + pickle.dumps(date_start)
            + pickle.dumps(date_end)
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
            + pickle.dumps(self.data.sizes)  # property of the original data
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
        if 't2m' in ds.variables:
            ds = ds.rename({'t2m': 't'})
        if 't2m_min' in ds.variables:
            ds = ds.rename({'t2m_min': 't_min'})
        if 't2m_max' in ds.variables:
            ds = ds.rename({'t2m_max': 't_max'})
        if 'pr' in ds.variables:
            ds = ds.rename({'pr': 'tp'})
        if 'tas' in ds.variables:
            ds = ds.rename({'tas': 't'})

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
        ds['time'] = pd.to_datetime(ds.time.values)
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
