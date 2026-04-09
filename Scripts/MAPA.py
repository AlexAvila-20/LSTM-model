from datetime import datetime as dt
import os

from gtMapTools import gtRaster, nc_info

# input file
ncfile = 'predicciones_2026.nc'

# view file variable, coordinate names and other info
print(nc_info(ncfile))

# declare object
example_raster = gtRaster()


date_sel = dt(2026, 1, 1, 0, 0, 0)

example_raster.getNc_data(
	ncfile,
	latnm='lat',
	lonnm='lon',
	timenm='time',
	datanm='predicted',
	datefilter=date_sel,
	operation='acum',
)
# the input in this example does not have a time axis
# if the input has a time dimension:
#   give it's name with timenm=...
#   select a date range with datefilter=(datetime(start ...), datetime(end ...))
#   apply an operation to the selected range with operation={}'acum', 'mean', ...}

# titles and text
example_raster.setTitle('Data from NetCDF example')
example_raster.setDataFrom('NetCDF example: Data from text')
example_raster.setInfo('NetCDF example: info text\nSpatial resolution 1 arcmin')
# plot map
os.makedirs('output', exist_ok=True)
example_raster.interpolate(resolution=1, sigma=2)
example_raster.plotData('fromNC.png', 'output/', settings='precip:month')

