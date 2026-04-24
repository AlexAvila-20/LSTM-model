from datetime import datetime as dt
import argparse
import os
import shapefile
import numpy as np
from matplotlib.path import Path

from gtMapTools import gtRaster, nc_info


REGIONES = [
    'Petén',
    'Franja Transversal del Norte',
    'Pacífico',
    'Boca Costa',
    'Valles de Oriente',
    'Occidente',
    'Altiplano Central',
    'Caribe'
]


# input file
ncfile = 'predicciones.nc'

# shapefile de regiones climáticas
SHAPEFILE_REGIONES = '/home/alex/Downloads/ELL/ELLv3-main/utilities/regiones_climaticas/regiones_gcs_wgs_1984.shp'


def normalizar_texto(texto):
    """Normaliza texto para comparar nombres de regiones."""
    reemplazos = str.maketrans({
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u',
        'ñ': 'n', 'Ñ': 'n'
    })
    return str(texto).strip().lower().translate(reemplazos)


def resolver_region(locate):
    """
    Convierte `locate` en índice y nombre de región.

    Admite:
    - entero o string entero: índice 0..7
    - nombre de región, con o sin acentos
    """
    if locate is None:
        raise ValueError(
            'Debes indicar una región con --locate. '
            f'Regiones disponibles: {", ".join(f"{i}:{r}" for i, r in enumerate(REGIONES))}'
        )

    # caso índice
    if isinstance(locate, int) or (isinstance(locate, str) and locate.strip().isdigit()):
        idx = int(locate)
        if idx < 0 or idx >= len(REGIONES):
            raise ValueError(f'Índice de región inválido: {idx}. Debe estar entre 0 y {len(REGIONES)-1}.')
        return idx, REGIONES[idx]

    # caso nombre
    locate_norm = normalizar_texto(locate)
    for idx, nombre in enumerate(REGIONES):
        if normalizar_texto(nombre) == locate_norm:
            return idx, nombre

    raise ValueError(
        f'Región no reconocida: {locate}. '
        f'Regiones disponibles: {", ".join(f"{i}:{r}" for i, r in enumerate(REGIONES))}'
    )



def construir_mascara_region(example_raster, shapefile_path, region_num):
    """Construye máscara booleana para la región seleccionada."""
    regions = shapefile.Reader(shapefile_path)
    shape_record = regions.shapeRecords()[region_num]

    lon, lat = np.meshgrid(example_raster.longitudearray, example_raster.latitudearray)
    points = np.vstack((lon.ravel(), lat.ravel())).T
    mask = np.zeros(points.shape[0], dtype=bool)

    # Una región puede venir con varias partes; se recorren todas.
    shape = shape_record.shape
    puntos = shape.points
    partes = list(shape.parts) + [len(puntos)]

    for i in range(len(partes) - 1):
        inicio = partes[i]
        fin = partes[i + 1]
        poly = puntos[inicio:fin]
        path = Path(poly)
        mask |= path.contains_points(points)

    return mask.reshape(lon.shape)



def generar_mapa_region(
    locate,
    ncfile='predicciones.nc',
    shapefile_path=SHAPEFILE_REGIONES,
    output_dir='output',
    output_prefix='fromNC_region',
    latnm='lat',
    lonnm='lon',
    timenm='time',
    datanm='predicted',
    date_sel=dt(2025, 12, 1, 0, 0, 0),
    operation='acum',
    interpolation_resolution=1,
    interpolation_sigma=2,
    settings='precip:month',
):
    """
    Genera un mapa de una región climática específica a partir del NetCDF.

    Parámetro clave:
        locate: índice o nombre de región climática.
    """
    region_num, region_name = resolver_region(locate)

    # view file variable, coordinate names and other info
    print(nc_info(ncfile))

    # declare object
    example_raster = gtRaster()

    example_raster.getNc_data(
        ncfile,
        latnm=latnm,
        lonnm=lonnm,
        timenm=timenm,
        datanm=datanm,
        datefilter=date_sel,
        operation=operation,
    )

    # crear y aplicar máscara de la región
    mask = construir_mascara_region(example_raster, shapefile_path, region_num)
    if hasattr(example_raster, 'dataarray') and example_raster.dataarray is not None:
        example_raster.dataarray = np.where(mask, example_raster.dataarray, np.nan)

    # títulos y texto
    example_raster.setTitle(f'Data from NetCDF - Región: {region_name}')
    example_raster.setDataFrom('NetCDF example: Data from text')
    example_raster.setInfo(
        f'NetCDF example: Recorte de región {region_num} ({region_name})\n'
        'Spatial resolution 1 arcmin'
    )

    # plot map
    os.makedirs(output_dir, exist_ok=True)
    example_raster.interpolate(resolution=interpolation_resolution, sigma=interpolation_sigma)

    safe_name = normalizar_texto(region_name).replace(' ', '_')
    output_file = f'{output_prefix}_{region_num:02d}_{safe_name}.png'
    example_raster.plotData(output_file, output_dir, settings=settings)

    print(f'Región seleccionada: {region_num} - {region_name}')
    print(f'Mapa generado: {os.path.join(output_dir, output_file)}')



def main():
    parser = argparse.ArgumentParser(
        description='Genera un mapa recortado a una región climática específica'
    )
    parser.add_argument(
        '--locate',
        required=True,
        help=(
            'Región a graficar. Puede ser índice (0..7) o nombre. '
            f'Disponibles: {", ".join(f"{i}:{r}" for i, r in enumerate(REGIONES))}'
        ),
    )
    parser.add_argument('--ncfile', default=ncfile, help='Archivo NetCDF de entrada')
    parser.add_argument('--shapefile', default=SHAPEFILE_REGIONES, help='Shapefile de regiones climáticas')
    parser.add_argument('--output-dir', default='output', help='Directorio de salida')

    args = parser.parse_args()

    generar_mapa_region(
        locate=args.locate,
        ncfile=args.ncfile,
        shapefile_path=args.shapefile,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
