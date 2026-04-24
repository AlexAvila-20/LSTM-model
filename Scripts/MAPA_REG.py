from datetime import datetime as dt
import os
import shapefile
import numpy as np
from matplotlib.path import Path

from gtMapTools import gtRaster, nc_info

# input file
ncfile = 'predicciones.nc'

# view file variable, coordinate names and other info
print(nc_info(ncfile))

# declare object
example_raster = gtRaster()

date_sel = dt(2025, 12, 1, 0, 0, 0)

example_raster.getNc_data(
	ncfile,
	latnm='lat',
	lonnm='lon',
	timenm='time',
	datanm='predicted',
	datefilter=date_sel,
	operation='acum',
)

# cargar shapefile de regiones
regions = shapefile.Reader(os.path.join(utils_path, 'departamentos', 'departamentos.shp'))

# seleccionar región
# Regiones disponibles:
# 0: Petén
# 1: Franja Transversal del Norte
# 2: Pacífico
# 3: Boca Costa
# 4: Valles de Oriente
# 5: Occidente
# 6: Altiplano Central
# 7: Caribe
region_num = 3  # Cambiar este número para seleccionar otra región

# crear máscara para la región
polys = [regions.shapeRecords()[region_num].shape.points]
lon, lat = np.meshgrid(example_raster.longitudearray, example_raster.latitudearray)
points = np.vstack((lon.ravel(), lat.ravel())).T
mask = np.zeros(points.shape[0], dtype=bool)
for poly in polys:
    path = Path(poly)
    mask |= path.contains_points(points)
mask = mask.reshape(lon.shape)

# aplicar máscara al raster
if hasattr(example_raster, 'dataarray') and example_raster.dataarray is not None:
    example_raster.dataarray = np.where(mask, example_raster.dataarray, np.nan)

# titles and text
region_names = [
    'Petén',
    'Franja Transversal del Norte',
    'Pacífico',
    'Boca Costa',
    'Valles de Oriente',
    'Occidente',
    'Altiplano Central',
    'Caribe'
]
region_name = region_names[region_num]

example_raster.setTitle(f'Data from NetCDF - Región: {region_name}')
example_raster.setDataFrom('NetCDF example: Data from text')
example_raster.setInfo(f'NetCDF example: Recorte de región {region_num} ({region_name})\nSpatial resolution 1 arcmin')

# plot map
os.makedirs('output', exist_ok=True)
example_raster.interpolate(resolution=1, sigma=2)
example_raster.plotData(f'fromNC_region_{region_num:02d}.png', 'output/', settings='precip-ell:month')
