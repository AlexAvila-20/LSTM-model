import os
import argparse
import shapefile
import numpy as np
from matplotlib.path import Path
from gtMapTools import gtRaster
import gtMapTools as _gtMapTools


def ensure_time_lat_lon(arr, lat_size, lon_size):
    """Asegura que el arreglo quede en orden (time, lat, lon)."""
    if arr is None:
        return None
    if arr.ndim != 3:
        raise ValueError(f"La variable debe ser 3D, pero tiene forma {arr.shape}")

    # (time, lat, lon)
    if arr.shape[1] == lat_size and arr.shape[2] == lon_size:
        return arr

    # (lat, lon, time)
    if arr.shape[0] == lat_size and arr.shape[1] == lon_size:
        return np.transpose(arr, (2, 0, 1))

    # (lat, time, lon)
    if arr.shape[0] == lat_size and arr.shape[2] == lon_size:
        return np.transpose(arr, (1, 0, 2))

    # (lon, lat, time)
    if arr.shape[0] == lon_size and arr.shape[1] == lat_size:
        return np.transpose(arr, (2, 1, 0))

    raise ValueError(f"No se pudo identificar el orden de dimensiones del arreglo con forma {arr.shape}")


def copy_var_attributes(src_var, dst_var):
    """Copia atributos de una variable netCDF, excepto _FillValue."""
    for attr in src_var.ncattrs():
        if attr != '_FillValue':
            dst_var.setncattr(attr, src_var.getncattr(attr))


def get_var_dimensions(ds, var_name):
    """Obtiene los nombres y tamaños de las dimensiones de una variable."""
    var = ds.variables[var_name]
    dims = var.dimensions
    sizes = {dim: len(ds.dimensions[dim]) for dim in dims}
    return dims, sizes


def get_coordinates_for_var(ds, var_name, lat_cand, lon_cand, time_cand):
    """Obtiene los nombres de coordenadas asociadas a una variable."""
    dims, sizes = get_var_dimensions(ds, var_name)
    
    # Buscar en las dimensiones de la variable
    var_lat = None
    var_lon = None
    var_time = None
    
    for dim in dims:
        # Buscar coincidencias directas
        if dim in lat_cand:
            var_lat = dim
        elif dim in lon_cand:
            var_lon = dim
        elif dim in time_cand:
            var_time = dim
        else:
            # Buscar variantes sin sufijos (e.g., Y_2 -> Y)
            base_dim = dim.split('_')[0]
            if base_dim in lat_cand:
                var_lat = dim
            elif base_dim in lon_cand:
                var_lon = dim
            elif base_dim in time_cand:
                var_time = dim
    
    return var_lat, var_lon, var_time, sizes


def main():
    parser = argparse.ArgumentParser(
        description="Recorta un archivo NetCDF por región climática conservando el tiempo original."
    )
    parser.add_argument(
        "infile",
        help="Ruta al archivo NetCDF de entrada, por ejemplo: predicciones.nc"
    )
    parser.add_argument(
        "--region",
        type=int,
        default=3,
        help="Número de región climática (0 a 7). Por defecto: 3"
    )
    parser.add_argument(
        "--shape",
        default="/home/alex/Downloads/ELL/ELLv3-main/utilities/regiones_climaticas/regiones_gcs_wgs_1984.shp",
        help="Ruta al shapefile de regiones climáticas."
    )
    parser.add_argument(
        "--outdir",
        default="output",
        help="Directorio de salida. Por defecto: output"
    )
    args = parser.parse_args()

    infile = args.infile
    region_num = args.region
    shapefile_path = args.shape
    out_dir = args.outdir

    if region_num < 0 or region_num > 7:
        raise ValueError("La región debe estar entre 0 y 7.")

    try:
        import netCDF4 as nc
    except Exception as e:
        raise ImportError(f"No se pudo importar netCDF4: {e}")

    # Abrir archivo para detectar nombres de coordenadas
    ds_check = nc.Dataset(infile)
    
    # Candidatos para nombres de coordenadas
    lat_candidates = ['lat', 'latitude', 'lats', 'y', 'Y']
    lon_candidates = ['lon', 'longitude', 'lons', 'x', 'X']
    time_candidates = ['time', 'times', 'T', 't']
    
    # Variables de datos disponibles
    var_3d = [vn for vn, v in ds_check.variables.items() 
              if len(v.dimensions) == 3]
    
    if not var_3d:
        raise RuntimeError(f"No se encontraron variables 3D en {infile}")
    
    # Usar la primera variable 3D para detectar dimensiones
    first_var = var_3d[0]
    var_lat, var_lon, var_time, sizes = get_coordinates_for_var(
        ds_check, first_var, lat_candidates, lon_candidates, time_candidates
    )
    
    if var_lat is None or var_lon is None:
        raise RuntimeError(f"No se pudieron detectar las dimensiones lat/lon en {infile}")
    
    time_name = var_time if var_time else None
    
    # Leer coordenadas directamente
    lat_array = ds_check.variables[var_lat][:] if var_lat in ds_check.variables else None
    lon_array = ds_check.variables[var_lon][:] if var_lon in ds_check.variables else None
    
    ds_check.close()
    
    if lat_array is None or lon_array is None:
        raise RuntimeError(f"No se pudieron leer las coordenadas en {infile}")

    # Cargar shapefile
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"No se encontró el shapefile: {shapefile_path}")

    regions = shapefile.Reader(shapefile_path)

    # Regiones disponibles:
    # 0: Petén
    # 1: Franja Transversal del Norte
    # 2: Pacífico
    # 3: Boca Costa
    # 4: Valles de Oriente
    # 5: Occidente
    # 6: Altiplano Central
    # 7: Caribe
    polys = [regions.shapeRecords()[region_num].shape.points]

    # Crear máscara espacial usando coordenadas leídas directamente
    lon, lat = np.meshgrid(lon_array, lat_array)
    points = np.vstack((lon.ravel(), lat.ravel())).T
    mask = np.zeros(points.shape[0], dtype=bool)

    for poly in polys:
        path = Path(poly)
        mask |= path.contains_points(points)

    mask = mask.reshape(lon.shape)

    # Abrir archivo fuente
    ds_in = nc.Dataset(infile)

    try:
        # Leer variables de datos
        full_pred = ds_in.variables['predicted'][:] if 'predicted' in ds_in.variables else None
        full_obs = ds_in.variables['observed'][:] if 'observed' in ds_in.variables else None

        # Fallback: buscar variables 3D si no existen predicted/observed
        pred_name = 'predicted' if 'predicted' in ds_in.variables else None
        obs_name = 'observed' if 'observed' in ds_in.variables else None

        if full_pred is None and full_obs is None:
            vars_3d = []
            for vn, v in ds_in.variables.items():
                if len(v.shape) == 3 and vn not in [var_lat, var_lon, time_name]:
                    vars_3d.append(vn)

            if len(vars_3d) >= 1:
                pred_name = vars_3d[0]
                full_pred = ds_in.variables[pred_name][:]
            if len(vars_3d) >= 2:
                obs_name = vars_3d[1]
                full_obs = ds_in.variables[obs_name][:]

        if full_pred is None and full_obs is None:
            raise RuntimeError(f"No se encontraron variables 3D en {infile}")

        # Procesar cada variable según sus dimensiones
        if full_pred is not None:
            pred_dims, pred_sizes = get_var_dimensions(ds_in, pred_name)
            pred_lat, pred_lon, pred_time, _ = get_coordinates_for_var(
                ds_in, pred_name, lat_candidates, lon_candidates, time_candidates
            )
            
            if pred_lat and pred_lon and pred_lat in ds_in.variables and pred_lon in ds_in.variables:
                pred_lat_array = ds_in.variables[pred_lat][:]
                pred_lon_array = ds_in.variables[pred_lon][:]
                pred_lat_size = len(pred_lat_array)
                pred_lon_size = len(pred_lon_array)
                
                # Crear máscara para las dimensiones de predicted
                lon_p, lat_p = np.meshgrid(pred_lon_array, pred_lat_array)
                points_p = np.vstack((lon_p.ravel(), lat_p.ravel())).T
                mask_p = np.zeros(points_p.shape[0], dtype=bool)
                for poly in polys:
                    path = Path(poly)
                    mask_p |= path.contains_points(points_p)
                mask_p = mask_p.reshape(lon_p.shape)
                
                # Asegurar orden (time, lat, lon)
                full_pred = ensure_time_lat_lon(full_pred, pred_lat_size, pred_lon_size)
                # Aplicar máscara
                full_pred = np.where(mask_p[np.newaxis, :, :], full_pred, np.nan)
        
        if full_obs is not None:
            obs_dims, obs_sizes = get_var_dimensions(ds_in, obs_name)
            obs_lat, obs_lon, obs_time, _ = get_coordinates_for_var(
                ds_in, obs_name, lat_candidates, lon_candidates, time_candidates
            )
            
            if obs_lat and obs_lon and obs_lat in ds_in.variables and obs_lon in ds_in.variables:
                obs_lat_array = ds_in.variables[obs_lat][:]
                obs_lon_array = ds_in.variables[obs_lon][:]
                obs_lat_size = len(obs_lat_array)
                obs_lon_size = len(obs_lon_array)
                
                # Crear máscara para las dimensiones de observed
                lon_o, lat_o = np.meshgrid(obs_lon_array, obs_lat_array)
                points_o = np.vstack((lon_o.ravel(), lat_o.ravel())).T
                mask_o = np.zeros(points_o.shape[0], dtype=bool)
                for poly in polys:
                    path = Path(poly)
                    mask_o |= path.contains_points(points_o)
                mask_o = mask_o.reshape(lon_o.shape)
                
                # Asegurar orden (time, lat, lon)
                full_obs = ensure_time_lat_lon(full_obs, obs_lat_size, obs_lon_size)
                # Aplicar máscara
                full_obs = np.where(mask_o[np.newaxis, :, :], full_obs, np.nan)

        # Preparar salida
        os.makedirs(out_dir, exist_ok=True)
        out_full = os.path.join(out_dir, f"recortado_region_{region_num:02d}_full.nc")

        # Crear archivo de salida
        ds_out = nc.Dataset(out_full, 'w')

        try:
            # Copiar atributos globales
            for attr in ds_in.ncattrs():
                ds_out.setncattr(attr, ds_in.getncattr(attr))

            # Crear dimensiones necesarias
            dims_to_create = set()
            
            if full_pred is not None and pred_lat and pred_lon:
                dims_to_create.update([time_name if time_name else 'time', pred_lat, pred_lon])
            
            if full_obs is not None and obs_lat and obs_lon:
                dims_to_create.update([time_name if time_name else 'time', obs_lat, obs_lon])
            
            for dim_name in dims_to_create:
                if dim_name in ds_in.dimensions:
                    dim = ds_in.dimensions[dim_name]
                    if dim_name in [time_name]:
                        ds_out.createDimension(dim_name, None if dim.isunlimited() else len(dim))
                    else:
                        ds_out.createDimension(dim_name, len(dim))

            # Copiar variables de coordenadas
            if full_pred is not None and pred_lat and pred_lat in ds_in.variables:
                src_lat = ds_in.variables[pred_lat]
                lat_out = ds_out.createVariable(pred_lat, src_lat.dtype, src_lat.dimensions)
                lat_out[:] = src_lat[:]
                copy_var_attributes(src_lat, lat_out)
            
            if full_pred is not None and pred_lon and pred_lon in ds_in.variables:
                src_lon = ds_in.variables[pred_lon]
                lon_out = ds_out.createVariable(pred_lon, src_lon.dtype, src_lon.dimensions)
                lon_out[:] = src_lon[:]
                copy_var_attributes(src_lon, lon_out)
            
            if full_obs is not None and obs_lat and obs_lat in ds_in.variables and obs_lat != pred_lat:
                src_lat = ds_in.variables[obs_lat]
                lat_out = ds_out.createVariable(obs_lat, src_lat.dtype, src_lat.dimensions)
                lat_out[:] = src_lat[:]
                copy_var_attributes(src_lat, lat_out)
            
            if full_obs is not None and obs_lon and obs_lon in ds_in.variables and obs_lon != pred_lon:
                src_lon = ds_in.variables[obs_lon]
                lon_out = ds_out.createVariable(obs_lon, src_lon.dtype, src_lon.dimensions)
                lon_out[:] = src_lon[:]
                copy_var_attributes(src_lon, lon_out)
            
            # Copiar time
            if time_name and time_name in ds_in.variables:
                src_time = ds_in.variables[time_name]
                time_out = ds_out.createVariable(time_name, src_time.dtype, src_time.dimensions)
                time_out[:] = src_time[:]
                copy_var_attributes(src_time, time_out)

            # Escribir predicted
            if full_pred is not None and pred_lat and pred_lon and pred_time:
                src_var = ds_in.variables[pred_name]
                fill_value = getattr(src_var, '_FillValue', np.nan)
                nt_pred, ny_pred, nx_pred = full_pred.shape
                pred_out = ds_out.createVariable(
                    pred_name,
                    'f4',
                    (pred_time, pred_lat, pred_lon),
                    zlib=True,
                    complevel=4,
                    fill_value=fill_value
                )
                pred_out[:, :, :] = full_pred
                copy_var_attributes(src_var, pred_out)

            # Escribir observed
            if full_obs is not None and obs_lat and obs_lon and obs_time:
                src_var = ds_in.variables[obs_name]
                fill_value = getattr(src_var, '_FillValue', np.nan)
                nt_obs, ny_obs, nx_obs = full_obs.shape
                obs_out = ds_out.createVariable(
                    obs_name,
                    'f4',
                    (obs_time, obs_lat, obs_lon),
                    zlib=True,
                    complevel=4,
                    fill_value=fill_value
                )
                obs_out[:, :, :] = full_obs
                copy_var_attributes(src_var, obs_out)

            ds_out.description = f"Recorte de la región {region_num:02d} conservando todas las timesteps"
            ds_out.source_file = infile

        finally:
            ds_out.close()

        print(f"Archivo guardado en: {out_full}")

    finally:
        ds_in.close()


if __name__ == "__main__":
    main()