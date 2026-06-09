from datetime import datetime as dt, timedelta
import os
import argparse
import shapefile
import numpy as np
from matplotlib.path import Path
import netCDF4

from gtMapTools import gtRaster, nc_info

# Nombres de variables equivalentes
EQUIVALENT_VARS = {
    'longitude': ['lon', 'X'],
    'latitude': ['lat', 'Y'],
    'time': ['time', 'T'],
    'data': ['predicted', 'rfe', 'deterministic']
}

def find_variable_name(nc_file, possible_names):
    """Busca cuál de los nombres posibles existe en el archivo NetCDF."""
    with netCDF4.Dataset(nc_file, 'r') as ds:
        for name in possible_names:
            if name in ds.variables:
                return name
    raise ValueError(f"Ninguna de las variables {possible_names} fue encontrada en {nc_file}")

def get_month_name(month):
    """Retorna el nombre del mes en español."""
    months = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    return months.get(month, 'Mes Desconocido')

# Parsear argumentos de línea de comandos
parser = argparse.ArgumentParser(
    description='Genera mapas de precipitación por región climática'
)
parser.add_argument(
    'ncfile',
    help='Archivo NetCDF de entrada'
)
parser.add_argument(
    '--date',
    type=str,
    default='2025,1,1',
    help='Fecha en formato YYYY,M[,D]. Con --frequency=daily: se interpreta como mes completo (ej: 2025,1 acumula TODO enero 2025). Con --frequency=monthly: fecha específica (ej: 2025,1,1 para enero 1)'
)
parser.add_argument(
    '--region',
    type=int,
    default=3,
    help='Número de región climática (0-7, default: 3)'
)
parser.add_argument(
    '--frequency',
    type=str,
    choices=['daily', 'monthly'],
    default='daily',
    help='Frecuencia temporal del archivo: daily o monthly (default: daily)'
)
parser.add_argument(
    '--pronostico',
    action='store_true',
    default=False,
    help='Si se activa, muestra el gráfico como pronóstico (default: False)'
)
parser.add_argument(
    '--tipo-pronostico',
    type=str,
    choices=['LSTM', 'NextGen'],
    default=None,
    help='Tipo de pronóstico: LSTM o NextGen (required si se usa --pronostico)'
)

args = parser.parse_args()

# Validar que si se usa --pronostico, se especifique tipo-pronostico
if args.pronostico and args.tipo_pronostico is None:
    parser.error('El argumento --tipo-pronostico es requerido cuando se usa --pronostico')

# Procesar fecha
date_parts = args.date.split(',')
year = int(date_parts[0])
month = int(date_parts[1])
day = int(date_parts[2]) if len(date_parts) > 2 else 1

# IMPORTANTE: La interpretación del parámetro --date depende de la frecuencia
# - frequency='daily': --date se interpreta como MES COMPLETO (año,mes)
#   Ej: 2025,1,1 → acumula TODO enero 2025
# - frequency='monthly': --date se interpreta como FECHA específica (año,mes,día)
#   Ej: 2025,1,1 → datos del mes de enero 2025

if args.frequency == 'daily':
    # Crear rango para TODO el mes
    start_date = dt(year, month, 1, 0, 0, 0)
    if month == 12:
        end_date = dt(year + 1, 1, 1, 0, 0, 0)
    else:
        end_date = dt(year, month + 1, 1, 0, 0, 0)
    date_sel = (start_date, end_date)
    print(f"[FREQUENCY=DAILY] Acumulando precipitación de TODO el mes: {start_date.strftime('%B %Y')}")
else:
    # frequency='monthly': usar la fecha específica
    date_sel = dt(year, month, day, 0, 0, 0)

# view file variable, coordinate names and other info
print(nc_info(args.ncfile))

# Encontrar nombres de variables en el archivo
lon_name = find_variable_name(args.ncfile, EQUIVALENT_VARS['longitude'])
lat_name = find_variable_name(args.ncfile, EQUIVALENT_VARS['latitude'])
time_name = find_variable_name(args.ncfile, EQUIVALENT_VARS['time'])
data_name = find_variable_name(args.ncfile, EQUIVALENT_VARS['data'])

print(f"Variables detectadas: lon='{lon_name}', lat='{lat_name}', time='{time_name}', data='{data_name}'")

# declare object
example_raster = gtRaster()

# Determinar operación según frecuencia
operation = 'acum' if args.frequency == 'daily' else 'mean'

example_raster.getNc_data(
	args.ncfile,
	latnm=lat_name,
	lonnm=lon_name,
	timenm=time_name,
	datanm=data_name,
	datefilter=date_sel,
	operation=operation,
)

# Interpolar antes del recorte
example_raster.interpolate(resolution=1, sigma=2)

# cargar shapefile de regiones climáticas
shp_path = '/home/alex/Downloads/ELL/ELLv3-main/utilities/regiones_climaticas/regiones_gcs_wgs_1984.shp'
regions = shapefile.Reader(shp_path)

# seleccionar región climática
region_num = args.region

# Validar rango de región
if region_num < 0 or region_num > 7:
    raise ValueError(f"Región {region_num} fuera de rango. Debe estar entre 0 y 7")

# Nombres de las regiones climáticas
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

# crear máscara para la región climática
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

# calcular los límites de la región para usar como locate (para mostrar con detalle)
shape = regions.shapeRecords()[region_num].shape
shape_points = np.array(shape.points)
lon_min, lat_min = shape_points.min(axis=0)
lon_max, lat_max = shape_points.max(axis=0)

# crear locate con coordenadas de la región [lonmin, lonmax, latmin, latmax]
region_locate = [lon_min, lon_max, lat_min, lat_max]

# titles and text
month_name = get_month_name(month)

if args.pronostico:
    title = f'Pronóstico {args.tipo_pronostico} de precipitación acumulada\n{month_name} {year}'
    filename_prefix = f'Pronóstico_{args.tipo_pronostico}_{month}_{year}_{region_name}_{region_num:02d}'
else:
    title = f'Precipitación acumulada\n{month_name} {year}'
    filename_prefix = f'Precipitación_{month}_{year}_{region_name}_{region_num:02d}'

example_raster.setTitle(title)
example_raster.setDataFrom('Precipitación acumulada mensual')
example_raster.setInfo('\nResolución espacial de 1 arcmin')

# plot map con locate para mostrar con detalle los municipios
os.makedirs('output', exist_ok=True)
# Determinar settings según el mes
if 1 <= month <= 4:
    settings = 'precip:month'
elif 5 <= month <= 10:
    settings = 'precip-ell:month'
else:  # noviembre-diciembre (11-12)
    settings = 'precip-transition:month'

example_raster.plotData(
	f'{filename_prefix}.png',
	'output/',
	settings=settings,
	locate=region_locate,  # Mostrar esta región con detalle (municipios, etc)
	logo=False,  # Desabilitar logo exterior
	edge=False
)
