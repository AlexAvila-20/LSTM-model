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

    # Leer con gtRaster para obtener coordenadas y generar la máscara
    r = gtRaster()
    r.getNc_data(infile, latnm='lat', lonnm='lon', timenm='time')

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

    # Crear máscara espacial
    lon, lat = np.meshgrid(r.longitudearray, r.latitudearray)
    points = np.vstack((lon.ravel(), lat.ravel())).T
    mask = np.zeros(points.shape[0], dtype=bool)

    for poly in polys:
        path = Path(poly)
        mask |= path.contains_points(points)

    mask = mask.reshape(lon.shape)

    # Abrir archivo fuente
    ds_in = nc.Dataset(infile)

    try:
        # Detectar nombres de variables coordenadas
        lat_name = 'lat' if 'lat' in ds_in.variables else None
        lon_name = 'lon' if 'lon' in ds_in.variables else None
        time_name = 'time' if 'time' in ds_in.variables else None

        if lat_name is None or lon_name is None:
            raise RuntimeError("No se encontraron variables 'lat' y/o 'lon' en el archivo de entrada.")

        lat_src = ds_in.variables[lat_name]
        lon_src = ds_in.variables[lon_name]
        time_src = ds_in.variables[time_name] if time_name is not None else None

        lat_size = len(lat_src[:])
        lon_size = len(lon_src[:])

        # Leer variables de datos
        full_pred = ds_in.variables['predicted'][:] if 'predicted' in ds_in.variables else None
        full_obs = ds_in.variables['observed'][:] if 'observed' in ds_in.variables else None

        # Fallback: buscar variables 3D si no existen predicted/observed
        pred_name = 'predicted' if 'predicted' in ds_in.variables else None
        obs_name = 'observed' if 'observed' in ds_in.variables else None

        if full_pred is None and full_obs is None:
            vars_3d = []
            for vn, v in ds_in.variables.items():
                if len(v.shape) == 3 and vn not in [lat_name, lon_name, time_name]:
                    vars_3d.append(vn)

            if len(vars_3d) >= 1:
                pred_name = vars_3d[0]
                full_pred = ds_in.variables[pred_name][:]
            if len(vars_3d) >= 2:
                obs_name = vars_3d[1]
                full_obs = ds_in.variables[obs_name][:]

        if full_pred is None and full_obs is None:
            raise RuntimeError(f"No se encontraron variables 3D en {infile}")

        # Asegurar orden (time, lat, lon)
        if full_pred is not None:
            full_pred = ensure_time_lat_lon(full_pred, lat_size, lon_size)
        if full_obs is not None:
            full_obs = ensure_time_lat_lon(full_obs, lat_size, lon_size)

        # Aplicar máscara espacial
        if full_pred is not None:
            full_pred = np.where(mask[np.newaxis, :, :], full_pred, np.nan)
        if full_obs is not None:
            full_obs = np.where(mask[np.newaxis, :, :], full_obs, np.nan)

        # Compatibilidad con r.dataarray
        if full_pred is not None:
            r.dataarray = full_pred
        elif full_obs is not None:
            r.dataarray = full_obs

        # Preparar salida
        os.makedirs(out_dir, exist_ok=True)
        out_full = os.path.join(out_dir, f"recortado_region_{region_num:02d}_full.nc")

        if full_pred is not None:
            nt, ny, nx = full_pred.shape
        else:
            nt, ny, nx = full_obs.shape

        # Crear archivo de salida
        ds_out = nc.Dataset(out_full, 'w')

        try:
            # Copiar atributos globales
            for attr in ds_in.ncattrs():
                ds_out.setncattr(attr, ds_in.getncattr(attr))

            # Crear dimensiones
            for dim_name, dim in ds_in.dimensions.items():
                if dim_name == time_name:
                    ds_out.createDimension(dim_name, None if dim.isunlimited() else nt)
                elif dim_name == lat_name:
                    ds_out.createDimension(dim_name, ny)
                elif dim_name == lon_name:
                    ds_out.createDimension(dim_name, nx)
                else:
                    ds_out.createDimension(dim_name, len(dim))

            # Copiar lat exactamente
            lat_out = ds_out.createVariable(lat_name, lat_src.dtype, lat_src.dimensions)
            lat_out[:] = lat_src[:]
            copy_var_attributes(lat_src, lat_out)

            # Copiar lon exactamente
            lon_out = ds_out.createVariable(lon_name, lon_src.dtype, lon_src.dimensions)
            lon_out[:] = lon_src[:]
            copy_var_attributes(lon_src, lon_out)

            # Copiar time exactamente, sin reinterpretar
            if time_src is not None:
                time_out = ds_out.createVariable(time_name, time_src.dtype, time_src.dimensions)
                time_out[:] = time_src[:]
                copy_var_attributes(time_src, time_out)

            # Escribir predicted
            if full_pred is not None:
                src_var = ds_in.variables[pred_name]
                fill_value = getattr(src_var, '_FillValue', np.nan)
                pred_out = ds_out.createVariable(
                    pred_name,
                    'f4',
                    (time_name, lat_name, lon_name),
                    zlib=True,
                    complevel=4,
                    fill_value=fill_value
                )
                pred_out[:, :, :] = full_pred
                copy_var_attributes(src_var, pred_out)

            # Escribir observed
            if full_obs is not None:
                src_var = ds_in.variables[obs_name]
                fill_value = getattr(src_var, '_FillValue', np.nan)
                obs_out = ds_out.createVariable(
                    obs_name,
                    'f4',
                    (time_name, lat_name, lon_name),
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