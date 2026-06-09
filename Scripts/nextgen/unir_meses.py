#!/usr/bin/env python3
"""
Une archivos mensuales de precipitación (1.nc..12.nc) en una serie temporal,
remallando a la grilla común 91x101 aquellos archivos que tengan otra resolución.
Versión corregida: detecta automáticamente el nombre de la coordenada de tiempo.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

# ------------------------------------------------------------
# 1. Definir la grilla objetivo (91 x 101)
# ------------------------------------------------------------
target_lon = np.arange(-92.5, -87.95, 0.05)   # 91 puntos
target_lat = np.arange(18.5, 13.45, -0.05)    # 101 puntos

# ------------------------------------------------------------
# 2. Leer cada archivo, interpolar si es necesario y acumular
# ------------------------------------------------------------
data_arrays = []

for month in range(1, 13):
    fname = f"{month}.nc"
    print(f"Procesando {fname} ...", end=" ")

    ds = xr.open_dataset(fname)

    # Identificar la variable de datos (ignorar coordenadas)
    # Buscamos la variable principal: suele ser la única variable que no es coordenada
    data_var = None
    for var in ds.data_vars:
        if var not in ds.coords:
            data_var = var
            break
    if data_var is None:
        # Si no se encontró, intentar con 'deterministic' (nombre mostrado por CDO)
        if 'deterministic' in ds:
            data_var = 'deterministic'
        else:
            raise ValueError(f"No se encontró variable de datos en {fname}")

    da = ds[data_var]

    # Identificar nombres de dimensiones espaciales y temporal
    dims = list(da.dims)
    # Las dimensiones espaciales se asumen llamadas 'X','Y' o 'lon','lat' o ser las dos últimas.
    # Buscamos 'X','Y' primero
    if 'X' in dims and 'Y' in dims:
        lon_name, lat_name = 'X', 'Y'
    elif 'lon' in dims and 'lat' in dims:
        lon_name, lat_name = 'lon', 'lat'
    else:
        # Asumir que las dos últimas dimensiones son espaciales
        lon_name, lat_name = dims[-1], dims[-2]

    # La dimensión temporal será la que queda (debería ser solo una)
    time_dims = [d for d in dims if d not in (lon_name, lat_name)]
    if len(time_dims) != 1:
        raise ValueError(f"Se esperaba una sola dimensión no espacial, pero se encontraron: {time_dims}")
    time_dim = time_dims[0]

    # Obtener la coordenada temporal (puede estar en el DataArray o en el dataset)
    if time_dim in da.coords:
        time_coord = da[time_dim]
    elif time_dim in ds.coords:
        time_coord = ds[time_dim]
    else:
        # Último recurso: construir una coordenada a partir del atributo 'units' si existe
        # (no debería ser necesario)
        raise ValueError(f"No se encontró la coordenada temporal '{time_dim}' en {fname}")

    # Coordenadas espaciales del archivo
    lons = da[lon_name].values
    lats = da[lat_name].values

    # Verificar si la grilla coincide con la objetivo
    grid_match = (
        len(lons) == len(target_lon) and
        len(lats) == len(target_lat) and
        np.allclose(lons, target_lon) and
        np.allclose(lats, target_lat)
    )

    if grid_match:
        print("grilla correcta, no se interpola.")
        da_fixed = da
    else:
        print(f"grilla {len(lats)}x{len(lons)} -> interpolando a 101x91...", end=" ")
        # Crear mallas de puntos originales y de la nueva grilla
        lon2d, lat2d = np.meshgrid(lons, lats)
        target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)

        # Obtener los valores del campo (asumiendo un solo paso de tiempo)
        # da puede tener dimensiones [T, Y, X] o [Y, X] si T es singleton y se eliminó.
        # Para generalizar, seleccionamos el primer paso temporal si existe
        if time_dim in da.dims:
            values = da.isel({time_dim: 0}).values
        else:
            values = da.values.squeeze()

        # Interpolar con griddata (bilineal)
        new_values = griddata(
            (lon2d.ravel(), lat2d.ravel()),
            values.ravel(),
            (target_lon2d, target_lat2d),
            method='linear'
        )

        # Crear DataArray con las nuevas coordenadas
        da_fixed = xr.DataArray(
            new_values[np.newaxis, ...],  # añadir dimensión tiempo
            dims=(time_dim, lat_name, lon_name),
            coords={
                time_dim: time_coord,
                lat_name: target_lat,
                lon_name: target_lon
            },
            name=data_var
        )
        print("listo.")

    data_arrays.append(da_fixed)
    ds.close()

# ------------------------------------------------------------
# 3. Concatenar a lo largo del tiempo y guardar
# ------------------------------------------------------------
# Todos los DataArrays deberían tener la misma dimensión temporal.
# Usamos xr.concat, que alinea automáticamente por coordenadas
combined = xr.concat(data_arrays, dim=time_dim)
combined = combined.sortby(time_dim)

outfile = 'precip_acumulada_anual.nc'
combined.to_netcdf(outfile)
print(f"\nArchivo unido guardado como: {outfile}")
