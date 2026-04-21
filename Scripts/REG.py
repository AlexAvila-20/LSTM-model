import shapefile, numpy as np, os
from gtMapTools import gtRaster
import gtMapTools as _gtMapTools
from matplotlib.path import Path
from datetime import datetime as dt

r = gtRaster()
infile = 'C1E1.nc'
r.getNc_data(infile, latnm='lat', lonnm='lon', timenm='time')

# cargar shapefile de regiones
utils_path = os.path.join(os.path.dirname(_gtMapTools.__file__), 'utilities', 'maps')
# Nota: ajustar la ruta según donde estén disponibles los shapes de regiones
# Opción 1: Si están en gtMapTools
# regions = shapefile.Reader(os.path.join(utils_path, 'regiones', 'regiones.shp'))
# Opción 2: Ruta externa
regions = shapefile.Reader('/home/alex/Downloads/ELL/ELLv3-main/utilities/regiones_climaticas/regiones_gcs_wgs_1984.shp')

# seleccionar polígonos de la región (ej. región 1)
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
polys = [shape.shape.points for shape in regions.shapeRecords() if int(shape.record[1]) == region_num]

# crear máscara (True dentro)
lon, lat = np.meshgrid(r.longitudearray, r.latitudearray)
points = np.vstack((lon.ravel(), lat.ravel())).T
mask = np.zeros(points.shape[0], dtype=bool)
for poly in polys:
    path = Path(poly)
    mask |= path.contains_points(points)
mask = mask.reshape(lon.shape)

# aplicar máscara (fuera -> nan)
if hasattr(r, 'dataarray') and r.dataarray is not None and getattr(r, 'dataarray').ndim == 3:
    # ya tiene dimensión temporal
    r.dataarray = np.where(mask, r.dataarray, np.nan)
else:
    # si r.dataarray es 2D (solo un frame), leer el archivo de entrada completo
    try:
        from netCDF4 import Dataset
        ds_in = Dataset(infile)
        # leer ambas variables si existen
        full_pred = None
        full_obs = None
        if 'predicted' in ds_in.variables:
            full_pred = ds_in.variables['predicted'][:]
        if 'observed' in ds_in.variables:
            full_obs = ds_in.variables['observed'][:]
        # si no existen, buscar la primera variable 3D para cada uno
        if full_pred is None or full_obs is None:
            for vn, v in ds_in.variables.items():
                if len(v.shape) == 3:
                    if full_pred is None:
                        full_pred = v[:]
                    elif full_obs is None:
                        full_obs = v[:]
                if full_pred is not None and full_obs is not None:
                    break

        # cerrar
        ds_in.close()

        # aplicar máscara espacial a todas las timesteps para las variables disponibles
        if full_pred is not None:
            full_pred = np.where(mask, full_pred, np.nan)
        if full_obs is not None:
            full_obs = np.where(mask, full_obs, np.nan)

        # preferir asignar `r.dataarray` al predicted si existe
        if full_pred is not None:
            r.dataarray = full_pred
        elif full_obs is not None:
            r.dataarray = full_obs
        else:
            raise RuntimeError('No se encontraron variables 3D en ' + infile)
    except Exception:
        # fallback: aplicar máscara al arreglo actual (2D) y expandir
        r.dataarray = np.where(mask, getattr(r, 'dataarray', np.nan), np.nan)
        if r.dataarray is not None and r.dataarray.ndim == 2:
            r.dataarray = r.dataarray[np.newaxis, ...]
