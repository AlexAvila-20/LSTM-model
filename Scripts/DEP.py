import shapefile, numpy as np, os
from gtMapTools import gtRaster
import gtMapTools as _gtMapTools
from matplotlib.path import Path
from datetime import datetime as dt

r = gtRaster()
infile = 'modelo_pixel_predictions.nc'
r.getNc_data(infile, latnm='lat', lonnm='lon', timenm='time')
'''
1: Alta Verapaz
2: Baja Verapaz
3: Chimaltenango
4: Chiquimula
5: El Progreso
6: Escuintla
7: Guatemala
8: Huehuetenango
9: Izabal
10: Jalapa
11: Jutiapa
12: Petén
13: Quetzaltenango
14: Quiché
15: Retalhuleu
16: Sacatepéquez
17: San Marcos
18: Santa Rosa
19: Sololá
20: Suchitepéquez
21: Totonicapán
22: Zacapa
'''

# cargar shapefile de municipios (ruta interna del paquete)
utils_path = os.path.join(os.path.dirname(_gtMapTools.__file__), 'utilities', 'maps')
dep = shapefile.Reader(os.path.join(utils_path, 'departamentos', 'departamentos.shp'))

# seleccionar polígonos del departamento (ej. dept 1)
dept_num = 6
polys = [shape.shape.points for shape in dep.shapeRecords() if int(shape.record[0]) == dept_num]

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

# guardar recortado (manteniendo todas las timesteps si existen)
# gtRaster.saveData espera normalmente un arreglo 2D; si tenemos 3D
# evitamos llamar a saveData para no provocar errores y usamos el
# NetCDF creado más abajo (`recortado_depto08_full.nc`).
if hasattr(r, 'dataarray') and getattr(r, 'dataarray') is not None and getattr(r, 'dataarray').ndim == 2:
    r.saveData('recortado_depto08.nc', savepath='output/', type='.nc', latnm='Y', lonnm='X', datanm='predicted')
else:
    print('dataarray es 3D; saltando r.saveData() y escribiendo archivo NetCDF completo en output/recortado_depto08_full.nc')

# Escribir un archivo NetCDF que conserve todas las timesteps (sin colapsar)
try:
    import netCDF4 as nc
except Exception:
    nc = None

if r.dataarray is not None:
    arr = r.dataarray
    # asegurar formato (time, lat, lon)
    if arr.ndim == 3:
        # detectar si el orden es (time, lat, lon) o (lat, lon, time)
        if arr.shape[1:] == mask.shape:
            t_arr = arr
        elif arr.shape[0:2] == mask.shape:
            t_arr = np.transpose(arr, (2, 0, 1))
        else:
            t_arr = arr
    elif arr.ndim == 2:
        t_arr = arr[np.newaxis, ...]
    else:
        t_arr = arr

    # preparar salida
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    out_full = os.path.join(out_dir, 'recortado_depto08_full.nc')

    if nc is not None:
        try:
            # si disponemos de arrays leídos antes (full_pred/full_obs), úsalos
            try:
                t_arr_pred = full_pred
            except NameError:
                t_arr_pred = None
            try:
                t_arr_obs = full_obs
            except NameError:
                t_arr_obs = None

            # si no fueron leídos previamente, intentar cargar desde infile
            if t_arr_pred is None or t_arr_obs is None:
                try:
                    ds_chk = nc.Dataset(infile)
                    if t_arr_pred is None and 'predicted' in ds_chk.variables:
                        t_arr_pred = ds_chk.variables['predicted'][:]
                    if t_arr_obs is None and 'observed' in ds_chk.variables:
                        t_arr_obs = ds_chk.variables['observed'][:]
                    ds_chk.close()
                except Exception:
                    pass

            # fallback: usar el array cargado en r (t_arr)
            if t_arr_pred is None:
                t_arr_pred = t_arr
            # si only one available, shape from that
            nt = t_arr_pred.shape[0]
            ny = t_arr_pred.shape[1]
            nx = t_arr_pred.shape[2]

            ds = nc.Dataset(out_full, 'w')
            ds.createDimension('time', nt)
            ds.createDimension('lat', ny)
            ds.createDimension('lon', nx)

            lat_v = ds.createVariable('lat', 'f4', ('lat',))
            lon_v = ds.createVariable('lon', 'f4', ('lon',))
            time_v = ds.createVariable('time', 'f8', ('time',))

            pred = ds.createVariable('predicted', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4, fill_value=np.nan)
            obs = None
            if t_arr_obs is not None:
                obs = ds.createVariable('observed', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4, fill_value=np.nan)

            # cargar coords si están disponibles en `r`
            if hasattr(r, 'latitudearray'):
                lat_v[:] = r.latitudearray
            else:
                lat_v[:] = np.arange(ny)
            if hasattr(r, 'longitudearray'):
                lon_v[:] = r.longitudearray
            else:
                lon_v[:] = np.arange(nx)

            # intentar obtener tiempos originales
            time_vals = None
            for attr in ('timearray', 'time', 'dates', 'time_vals', 'time_values'):
                if hasattr(r, attr):
                    time_vals = getattr(r, attr)
                    break

            if time_vals is None:
                time_v[:] = np.arange(nt)
            else:
                try:
                    # si son datetimes, convertir a números
                    if hasattr(nc, 'date2num') and hasattr(time_vals[0], 'year'):
                        units = 'days since 1970-01-01'
                        time_v[:] = nc.date2num(time_vals, units=units)
                        time_v.units = units
                    else:
                        time_v[:] = np.array(time_vals)
                except Exception:
                    time_v[:] = np.arange(nt)

            # escribir variables
            pred[:, :, :] = t_arr_pred
            if obs is not None:
                # asegurar que observed tiene la misma forma; si no, intentar transponer
                if t_arr_obs.shape != (nt, ny, nx):
                    try:
                        t_arr_obs = np.transpose(t_arr_obs, (2, 0, 1))
                    except Exception:
                        pass
                obs[:, :, :] = t_arr_obs

            ds.description = 'Recorte del departamento %02d conservando todas las timesteps' % dept_num
            ds.close()
        except Exception as e:
            print('Error escribiendo', out_full, e)
    else:
        # fallback: guardar como .npy si netCDF4 no está disponible
        np.save(out_full.replace('.nc', '.npy'), t_arr)