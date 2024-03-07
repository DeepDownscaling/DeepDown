# This script converts a raster file to NetCDF format and plots the data.

import xarray as xr
import matplotlib.pyplot as plt
import rioxarray

raster = rioxarray.open_rasterio("/path/to/srtm_1k.tif")

# Check the raster file
plt.imshow(raster.squeeze())

# Convert to NetCDF
nc_file = '/path/to/srtm_1k.nc'
raster.to_netcdf(nc_file)

# Open the NetCDF file using xarray
dataset = xr.open_dataset(nc_file)

# Access the data variables
data_var = dataset['__xarray_dataarray_variable__']

# Plot the data using imshow()
plt.imshow(data_var.squeeze())

# Add a colorbar
plt.colorbar()

# Show the plot
plt.show()
