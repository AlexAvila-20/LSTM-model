#!/usr/bin/env python
"""
MODELO.py — Versión enfocada en evitar exceder la RAM disponible
====================================================================

Misma funcionalidad que num1_opt.py, pero optimizada para NUNCA
exceder la RAM disponible sin importar el tamaño de los predictores.

CAMBIOS PRINCIPALES respecto a num1_opt.py:
  1. Monitor de RAM en tiempo real (psutil) ajusta dinámicamente
     los tamaños de chunk en TODAS las operaciones.
  2. NUNCA carga predictores completos en RAM — siempre lee por chunks
     desde disco vía xarray/memmap.
  3. pca_components=0 → auto-PCA: usa IncrementalPCA con umbral de
     varianza explicada (99.9%) para encontrar el mínimo de componentes
     que captura toda la variabilidad, sin exceder la RAM.
  4. Predicciones por chunks en lugar de model.predict() completo.
  5. gc.collect() agresivo en cada etapa.
  6. Todos los temporales usan memmap en disco.
  7. tf.data generator lee desde memmap sin copias redundantes.

USO: Mismos argumentos que num1_opt.py. Con --pca_components 0:
  python num1_opt_lowram.py \
    --precip ENACTS2_prcp.nc --varname rfe \
    --predictor "reg.sst.day.mean.1981-2026.nc:sst" \
    --predictor "era5.1.nc:z,r,q" \
    --predictor "precip.daily.1981-2026.fixed.nc:tp" \
    --predictor "era5.2.nc:u,v" \
    --n_jobs 4 --max_ram_gb 4.0 --batch_size 64 --epochs 500 \
    --pca_components 0 --spatial_embed 512 \
    --temporal_dim 192 --hidden_dim 256 --dtype float32 \
    --huber_delta 1.5 --target_transform log1p \
    --tweedie_weight 0.4 --quantile_weight 0.2  --tweedie_power 1.3 \
    --quantile_tau 0.85 --low_threshold 30 --high_threshold 150 \
    --corr_weight 0.05 --extreme_boost 3.5 \
    --use_swa --noise_std 0.05 --mixup_alpha 0.4 \
    --n_attn_heads 8 --force_preprocess

====================================================================
"""

# ─── PYTHONHASHSEED debe fijarse ANTES de cualquier import ──────────
# Python usa hashing interno para sets/dicts; fijar PYTHONHASHSEED
# garantiza que el orden de iteración sea determinístico entre ejecuciones.
import os as _os
import sys as _sys

_SEED_ENV_KEY = 'PYTHONHASHSEED'
_default_seed = '42'
# Buscar si el usuario pasa --seed por CLI para usarla también como PYTHONHASHSEED
for _i, _arg in enumerate(_sys.argv):
    if _arg == '--seed' and _i + 1 < len(_sys.argv):
        _default_seed = _sys.argv[_i + 1]
        break

# Si PYTHONHASHSEED no coincide con la semilla deseada, re-lanzar el proceso
# con la variable de entorno correcta (debe fijarse ANTES de que Python arranque)
if _os.environ.get(_SEED_ENV_KEY) != _default_seed:
    _os.environ[_SEED_ENV_KEY] = _default_seed
    if __name__ == '__main__':
        import subprocess
        _result = subprocess.run(
            [_sys.executable] + _sys.argv,
            env=_os.environ.copy()
        )
        _sys.exit(_result.returncode)

# ─── Imports principales ────────────────────────────────────────────
import argparse          # Parseo de argumentos de línea de comandos
import os                # Operaciones de sistema de archivos
import gc                # Recolector de basura manual (crítico para control de RAM)
import json              # Lectura/escritura de metadatos y estadísticas
import pickle            # Serialización de objetos PCA como fallback
import time              # Cronómetro para medir tiempos de cada fase
import concurrent.futures  # Paralelización de la Fase 2 (cacheo por chunks)
import multiprocessing   # Contexto forkserver para workers seguros
import random            # Semilla de Python nativo
from datetime import timedelta  # Aritmética de fechas para ventanas temporales

import numpy as np       # Operaciones numéricas y memmap en disco
import xarray as xr      # Lectura de archivos NetCDF (predictores y target)
import pandas as pd      # Manejo de índices temporales y fechas
import tensorflow as tf  # Framework de deep learning para el modelo
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, TimeDistributed,   # Capas básicas
    BatchNormalization, LSTM, Concatenate,     # Normalización, recurrencia, fusión
    Multiply, Permute, RepeatVector, Flatten, Softmax,  # Mecanismo de atención
    Bidirectional,                             # LSTM bidireccional
)
from tensorflow.keras.layers import Add, Lambda  # Conexiones residuales y ops custom
from scipy.stats import pearsonr               # Correlación de Pearson para métricas

# Imports opcionales — el script funciona sin ellos pero con capacidad reducida
try:
    from sklearn.decomposition import IncrementalPCA, PCA as SklearnPCA, TruncatedSVD
    HAS_SKLEARN = True  # Necesario para reducción de dimensionalidad PCA
except ImportError:
    HAS_SKLEARN = False

try:
    import joblib
    HAS_JOBLIB = True  # Serialización eficiente de modelos PCA
except ImportError:
    HAS_JOBLIB = False

try:
    import psutil
    HAS_PSUTIL = True  # Monitoreo preciso de RAM del proceso
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib.pyplot as plt  # Gráficos (no usado directamente aquí)
except Exception:
    plt = None

# ─── GPU memory growth ──────────────────────────────────────────────
# Evita que TensorFlow reserve toda la VRAM de golpe; crece según necesidad
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


#  Monitor de RAM — núcleo del control de memoria

class RAMMonitor:
    """Monitor de RAM en tiempo real. Provee presupuesto disponible
    y calcula chunk sizes seguros para cualquier operación."""

    def __init__(self, max_ram_gb=4.0, safety_margin=0.20):
        """
        max_ram_gb: RAM máxima que el proceso puede usar en total (GB).
        safety_margin: fracción de RAM que siempre se deja libre (20%).
        """
        self.max_ram_bytes = int(max_ram_gb * 1024**3)
        self.safety_margin = safety_margin

    def _get_process_rss(self):
        """RSS actual del proceso en bytes."""
        if HAS_PSUTIL:
            try:
                return psutil.Process(os.getpid()).memory_info().rss
            except Exception:
                pass
        # Fallback: leer /proc/self/status en Linux
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) * 1024  # kB → bytes
        except Exception:
            pass
        return 0

    def _get_system_available(self):
        """RAM disponible del sistema en bytes."""
        if HAS_PSUTIL:
            try:
                return psutil.virtual_memory().available
            except Exception:
                pass
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return self.max_ram_bytes

    def available_bytes(self):
        """Bytes disponibles para nuevas asignaciones, considerando
        tanto el límite del usuario como la RAM real del sistema."""
        rss = self._get_process_rss()
        budget_from_limit = max(0, self.max_ram_bytes - rss)
        sys_avail = self._get_system_available()
        usable = min(budget_from_limit, sys_avail)
        safe = int(usable * (1.0 - self.safety_margin))
        return max(safe, 1024 * 1024)  # mínimo 1 MB

    def available_gb(self):
        return self.available_bytes() / (1024**3)

    def safe_chunk_rows(self, cols, dtype_bytes=4, fraction=0.5):
        """Calcula cuántas filas de una matriz (rows, cols) caben en
        `fraction` de la RAM disponible.
        Usado para determinar el tamaño de chunk al procesar predicciones,
        diagnósticos, etc. sin exceder la memoria."""
        avail = self.available_bytes() * fraction
        row_bytes = cols * dtype_bytes  # Bytes por fila de la matriz
        rows = max(1, int(avail / row_bytes))  # Mínimo 1 fila siempre
        return rows

    def safe_chunk_time(self, n_valid, dtype_bytes=4, fraction=0.3):
        """Cuántos timesteps de un predictor (time, n_valid) caben.
        Cada timestep ocupa n_valid * dtype_bytes. Se usa fraction=0.3
        para dejar margen a otras operaciones simultáneas."""
        avail = self.available_bytes() * fraction
        step_bytes = n_valid * dtype_bytes
        steps = max(1, int(avail / step_bytes))
        return steps

    def report(self, label=""):
        rss_gb = self._get_process_rss() / (1024**3)
        avail_gb = self.available_gb()
        limit_gb = self.max_ram_bytes / (1024**3)
        print(f"   [RAM{' '+label if label else ''}] "
              f"RSS={rss_gb:.2f} GB | Disponible={avail_gb:.2f} GB | "
              f"Límite={limit_gb:.2f} GB")


# Instancia global — se inicializa en main()
_RAM: RAMMonitor = None


def get_ram():
    global _RAM
    if _RAM is None:
        _RAM = RAMMonitor(max_ram_gb=4.0)
    return _RAM



#  Control de semillas para reproducibilidad

GLOBAL_SEED: int = 42


def set_global_seeds(seed: int) -> None:
    global GLOBAL_SEED
    GLOBAL_SEED = int(seed)
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    tf.random.set_seed(GLOBAL_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    print(f"   [seed] Semillas globales fijadas: {GLOBAL_SEED} "
          f"(Python, NumPy, TensorFlow, CUDA, PYTHONHASHSEED)")


def _worker_seed(pred_idx: int) -> int:
    return GLOBAL_SEED + int(pred_idx) + 1


def _init_worker_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as _tf
        _tf.random.set_seed(seed)
    except Exception:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)


#  Constantes de clip físico


_PRECIP_MAX_MM = 2000.0

CLIP_RANGES = {
    'log1p':    (-0.5,  float(np.log1p(_PRECIP_MAX_MM))),
    'cbrt':     (0.0,   float(_PRECIP_MAX_MM ** (1/3))),
    'pow025':   (0.0,   float(_PRECIP_MAX_MM ** 0.25)),
    'pow05':    (0.0,   float(np.sqrt(_PRECIP_MAX_MM))),
    'standard': (-10.0, 10.0),
    'none':     (0.0,   _PRECIP_MAX_MM),
}


#  Utilidades
def detect_variable_domain(da, varname):
    """Detecta automáticamente si una variable es oceánica, terrestre o global.
    Primero busca palabras clave en el nombre de la variable;
    si no encuentra coincidencia, analiza el patrón de NaN:
      - >90% NaN → probablemente oceánico (tierra enmascarada)
      - >30% NaN → probablemente terrestre (océano enmascarado)
      - <30% NaN → global (cobertura completa)
    """
    ocean_kw = ['sst', 'sea_surface', 'ocean', 'sss', 'salinity']
    land_kw  = ['soil', 'vegetation', 'ndvi', 'sm', 'lst']
    vl = varname.lower()
    for ov in ocean_kw:
        if ov in vl:
            return 'ocean'
    for lv in land_kw:
        if lv in vl:
            return 'land'
    # Fallback: inferir por proporción de NaN en una muestra temporal
    try:
        sample = (da.isel(time=slice(0, min(10, len(da['time']))))
                  if 'time' in da.dims else da)
        nan_ratio = np.isnan(sample.values).sum() / sample.values.size
        if nan_ratio > 0.9:
            return 'ocean'
        elif nan_ratio > 0.3:
            return 'land'
        return 'global'
    except Exception:
        return 'global'


def create_validity_mask(da, n_sample=50):
    """Genera máscara booleana (lat, lon) indicando qué píxeles tienen
    al menos un valor finito (no-NaN) en las primeras n_sample timesteps.
    Los píxeles siempre-NaN se descartan para reducir dimensionalidad."""
    nt = min(n_sample, len(da['time'])) if 'time' in da.dims else 1
    if 'time' in da.dims:
        sample = da.isel(time=slice(0, nt)).values
    else:
        sample = da.values[np.newaxis, ...]
    return np.any(np.isfinite(sample), axis=0)


def _normalize_time_dim(ds):
    """Renombra variantes comunes de la dimensión temporal a 'time'.
    Distintos datasets usan 'T', 'valid_time', etc. — esto lo estandariza."""
    for tname in ['T', 'valid_time', 'time_counter']:
        if tname in ds.dims or tname in ds.coords:
            try:
                ds = ds.rename({tname: 'time'})
            except Exception:
                pass
            break
    return ds


def _normalize_spatial_dims(da):
    """Estandariza las dimensiones espaciales a 'lat'/'lon' y ordena
    a (time, lat, lon). Si faltan dimensiones espaciales, las añade
    como dimensiones unitarias para que el código downstream sea uniforme."""
    rename = {}
    # Buscar variantes de latitud
    for name in ['latitude', 'y', 'Y']:
        if name in da.dims or name in da.coords:
            rename[name] = 'lat'
            break
    # Buscar variantes de longitud
    for name in ['longitude', 'x', 'X']:
        if name in da.dims or name in da.coords:
            rename[name] = 'lon'
            break
    if rename:
        try:
            da = da.rename(rename)
        except Exception:
            pass
    # Expandir dimensiones faltantes (e.g. datos de 1 punto)
    if 'lat' not in da.dims and 'lon' not in da.dims:
        da = da.expand_dims({'lat': [0.0], 'lon': [0.0]})
    elif 'lat' not in da.dims:
        da = da.expand_dims({'lat': [0.0]})
    elif 'lon' not in da.dims:
        da = da.expand_dims({'lon': [0.0]})
    # Asegurar orden canónico de dimensiones
    try:
        if all(d in da.dims for d in ('time', 'lat', 'lon')):
            da = da.transpose('time', 'lat', 'lon')
    except Exception:
        pass
    return da


def get_month_ranges(times):
    """Genera lista de tuplas (inicio_mes, fin_mes) para cada mes único
    presente en el array de tiempos. Cada tupla delimita un mes calendario
    completo, usado para calcular precipitación mensual acumulada."""
    idx = pd.DatetimeIndex(times)
    months = idx.to_period('M').unique()
    return [
        (m.to_timestamp(), (m + 1).to_timestamp() - pd.Timedelta('1D'))
        for m in months
    ]


def estimate_predictor_ram_gb(pinfo):
    """Estima cuántos GB ocuparía cargar un predictor completo en RAM.
    Fórmula: timesteps × píxeles_válidos × 4 bytes (float32).
    Se usa para reportar y decidir estrategias de procesamiento."""
    n_t = len(pinfo['times'])
    n_v = pinfo['n_valid']
    return (n_t * n_v * 4) / (1024 ** 3)


def _get_clip_range(target_transform):
    return CLIP_RANGES.get(target_transform, (-10.0, 1e6))


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Red multi-torre v4 LOW-RAM: nunca excede la RAM disponible"
    )
    p.add_argument('--precip', required=True)
    p.add_argument('--varname', default='rfe')
    p.add_argument('--predictor', action='append', required=True,
                   help='"archivo.nc:var1[,var2,...][:dominio]"')
    p.add_argument('--input_days', type=int, default=31)
    p.add_argument('--pca_components', type=int, default=1500,
                   help='Componentes PCA por predictor. '
                        '0 = auto (retiene varianza según --pca_variance). '
                        '-1 = SIN PCA (usa toda la información espacial cruda). '
                        'Con -1, batch_size se ajusta automáticamente.')
    p.add_argument('--no_pca', action='store_true',
                   help='Equivalente a --pca_components -1. '
                        'Usa TODA la información espacial sin reducción.')
    p.add_argument('--pca_variance', type=float, default=0.999,
                   help='Varianza explicada objetivo cuando pca_components=0. Default 0.999')
    p.add_argument('--pca_samples', type=int, default=2000)
    p.add_argument('--stats_chunk_time', type=int, default=64)
    p.add_argument('--pca_chunk_time', type=int, default=32)
    p.add_argument('--spatial_embed', type=int, default=256)
    p.add_argument('--temporal_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--huber_delta', type=float, default=1.0)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--split_frac', type=float, default=0.8)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--target_transform', default='cbrt',
                   choices=['none', 'standard', 'log1p', 'cbrt', 'pow025', 'pow05'])
    p.add_argument('--cache_dir', default='./preprocessing_cache')
    p.add_argument('--force_preprocess', action='store_true')
    p.add_argument('--save', default='modelo_pixel.keras')
    p.add_argument('--dtype', default='float32',
                   choices=['float16', 'float32'])
    p.add_argument('--n_jobs', type=int, default=-1)
    p.add_argument('--max_ram_gb', type=float, default=4.0,
                   help='RAM máxima total que el proceso puede usar (GB). '
                        'El monitor de RAM ajusta chunks dinámicamente.')
    p.add_argument('--quantile_weight', type=float, default=0.30)
    p.add_argument('--quantile_tau', type=float, default=0.72)
    p.add_argument('--low_threshold', type=float, default=50.0)
    p.add_argument('--high_threshold', type=float, default=250.0)
    p.add_argument('--tweedie_weight', type=float, default=0.40)
    p.add_argument('--tweedie_power', type=float, default=1.30)
    p.add_argument('--tweedie_scale_norm', type=lambda x: x.lower() != 'false',
                   default=True)
    p.add_argument('--pixel_embed_dim', type=int, default=64)
    p.add_argument('--l2_output', type=float, default=1e-5)
    p.add_argument('--grad_accum_steps', type=int, default=1)
    p.add_argument('--coord_embed_dim', type=int, default=32)
    p.add_argument('--spatial_refine', action='store_true')
    p.add_argument('--smooth_weight', type=float, default=0.001)
    p.add_argument('--precip_max_mm', type=float, default=2000.0)
    p.add_argument('--corr_weight', type=float, default=0.10)
    p.add_argument('--extreme_boost', type=float, default=2.0)
    p.add_argument('--n_attn_heads', type=int, default=4)
    p.add_argument('--use_swa', action='store_true')
    p.add_argument('--noise_std', type=float, default=0.05)
    p.add_argument('--mixup_alpha', type=float, default=0.4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seed_split', type=int, default=None,
                   help='Seed para division aleatoria de años (train/val). '
                        'Si no se especifica, usa --seed')
    args = p.parse_args()
    # --no_pca equivale a --pca_components -1
    if args.no_pca:
        args.pca_components = -1
    return args


def parse_predictor_str(pred_str):
    """Parsea una cadena de predictor con formato 'archivo.nc:var1[,var2,...][:dominio]'.
    Retorna lista de tuplas (filepath, varname, domain) — una por cada variable.
    El dominio puede ser 'ocean', 'land', 'global' o 'auto' (detección automática)."""
    parts = pred_str.split(':')
    if len(parts) < 2:
        raise ValueError(
            f"Formato inválido: '{pred_str}'  →  archivo.nc:var1[,var2,...][:dominio]"
        )
    filepath = parts[0]                        # Ruta al archivo NetCDF
    varnames = parts[1].split(',')             # Lista de variables a extraer
    domain   = parts[2] if len(parts) > 2 else 'auto'  # Dominio explícito o auto
    # Intentar añadir extensión .nc si el archivo no existe tal cual
    if not os.path.isfile(filepath) and os.path.isfile(filepath + '.nc'):
        filepath = filepath + '.nc'
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    return [(filepath, vn.strip(), domain) for vn in varnames]


# ════════════════════════════════════════════════════════════════════
#  Apertura normalizada de un predictor
# ════════════════════════════════════════════════════════════════════

def _open_predictor_da(filepath, varname):
    """Abre un archivo NetCDF de predictor y devuelve el DataArray normalizado.
    Estandariza dimensiones temporales y espaciales para compatibilidad."""
    ds = xr.open_dataset(filepath)
    ds = _normalize_time_dim(ds)
    da = ds[varname]
    da = _normalize_spatial_dims(da)
    # Convertir tiempos no-datetime64 (e.g. cftime) a datetime64
    if not np.issubdtype(da['time'].dtype, np.datetime64):
        da['time'] = pd.to_datetime(da['time'].values)
    return da, ds


# ════════════════════════════════════════════════════════════════════
#  Vecinos espaciales para suavizado
# ════════════════════════════════════════════════════════════════════

def compute_neighbor_pairs(flat_indices, target_h, target_w):
    """Calcula pares de píxeles vecinos (horizontal y vertical) en la grilla
    del target, considerando solo los píxeles válidos (tierra).
    Se usa para la componente de suavizado espacial en la función de pérdida:
    penaliza diferencias grandes entre predicciones de píxeles adyacentes.
    
    Retorna (pairs_i, pairs_j): arrays donde pairs_i[k] y pairs_j[k]
    son índices (en el espacio land-only) de dos píxeles vecinos.
    """
    flat_set = set(int(fi) for fi in flat_indices)     # Búsqueda O(1)
    flat_to_land = {int(fi): li for li, fi in enumerate(flat_indices)}  # flat → land idx
    pairs_i, pairs_j = [], []
    for li, fi in enumerate(flat_indices):
        fi = int(fi)
        row, col = fi // target_w, fi % target_w
        # Vecino derecho
        if col + 1 < target_w:
            right_fi = row * target_w + (col + 1)
            if right_fi in flat_set:
                pairs_i.append(li)
                pairs_j.append(flat_to_land[right_fi])
        # Vecino inferior
        if row + 1 < target_h:
            bottom_fi = (row + 1) * target_w + col
            if bottom_fi in flat_set:
                pairs_i.append(li)
                pairs_j.append(flat_to_land[bottom_fi])
    return np.array(pairs_i, dtype=np.int32), np.array(pairs_j, dtype=np.int32)


# ════════════════════════════════════════════════════════════════════
#  PÉRDIDA TRIPLE: Huber + Quantil adaptativo + Tweedie
# ════════════════════════════════════════════════════════════════════

def make_adaptive_loss(huber_delta=1.0,
                       quantile_weight=0.30,
                       quantile_tau=0.72,
                       low_threshold=50.0,
                       high_threshold=250.0,
                       tweedie_weight=0.40,
                       tweedie_power=1.30,
                       tweedie_scale_norm=True,
                       target_transform='log1p',
                       neighbor_pairs=None,
                       smooth_weight=0.0,
                       corr_weight=0.0,
                       extreme_boost=1.0):
    total_named = quantile_weight + tweedie_weight + corr_weight
    if not (0.0 <= quantile_weight < 1.0):
        raise ValueError(f"quantile_weight debe estar en [0, 1), got {quantile_weight}")
    if not (0.0 <= tweedie_weight < 1.0):
        raise ValueError(f"tweedie_weight debe estar en [0, 1), got {tweedie_weight}")
    if not (0.0 <= corr_weight < 1.0):
        raise ValueError(f"corr_weight debe estar en [0, 1), got {corr_weight}")
    if total_named >= 1.0:
        raise ValueError(
            f"quantile_weight ({quantile_weight}) + tweedie_weight "
            f"({tweedie_weight}) + corr_weight ({corr_weight}) "
            f"debe ser < 1.0. Suma actual: {total_named:.3f}."
        )
    if not (0.5 <= quantile_tau <= 1.0):
        raise ValueError(f"quantile_tau debe estar en [0.5, 1.0], got {quantile_tau}")
    if not (1.0 < tweedie_power < 2.0):
        raise ValueError(f"tweedie_power debe estar en (1, 2). Got {tweedie_power}.")
    if low_threshold >= high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) debe ser < high_threshold ({high_threshold})"
        )

    # Peso implícito de Huber = 1 - (quantile + tweedie + corr)
    huber_weight = 1.0 - quantile_weight - tweedie_weight - corr_weight
    # Anchos de transición suave (sigmoide) para las regiones quantil
    low_width  = max(low_threshold  * 0.10, 1.0)
    high_width = max(high_threshold * 0.10, 1.0)
    # Factor de normalización de Tweedie para estabilidad numérica
    _tw_scale = float(1.0 / max(2.0 - tweedie_power, 1e-3)) \
                if tweedie_scale_norm else 1.0

    has_smooth = (smooth_weight > 0
                  and neighbor_pairs is not None
                  and len(neighbor_pairs[0]) > 0)
    if has_smooth:
        pairs_i_tf = tf.constant(neighbor_pairs[0], dtype=tf.int32)
        pairs_j_tf = tf.constant(neighbor_pairs[1], dtype=tf.int32)
        n_pairs = len(neighbor_pairs[0])
        print(f"   Suavizado espacial: {n_pairs} pares de vecinos, peso={smooth_weight}")

    _transform = target_transform

    @tf.function
    def _invert_transform(y_t):
        """Invierte la transformación del target para obtener mm reales.
        Se usa dentro del loss para calcular pesos adaptativos y Tweedie."""
        if _transform == 'log1p':
            return tf.math.expm1(tf.maximum(y_t, 0.0))
        elif _transform == 'cbrt':
            return tf.pow(tf.maximum(y_t, 0.0), 3.0)
        elif _transform == 'pow025':
            return tf.pow(tf.maximum(y_t, 0.0), 4.0)
        elif _transform == 'pow05':
            return tf.pow(tf.maximum(y_t, 0.0), 2.0)
        else:
            return tf.maximum(y_t, 0.0)

    _extreme_boost = float(extreme_boost)
    _corr_w = float(corr_weight)

    # ── Variables EMA para normalizar las escalas de cada componente ──
    # Durante el warmup, se actualizan las medias móviles exponenciales
    # de cada pérdida cruda. Después del warmup, las escalas se congelan.
    _ema_mom    = 0.95     # Momentum de la media móvil
    _ema_eps    = 1e-8     # Epsilon para evitar división por cero
    _ema_warmup = 50       # Pasos de calentamiento para estabilizar escalas
    _ema_h    = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='ema_huber')
    _ema_q    = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='ema_quant')
    _ema_t    = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='ema_tweed')
    _ema_c    = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='ema_corr')
    _ema_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='ema_step')
    _in_train_ph = tf.Variable(True, trainable=False, dtype=tf.bool, name='ema_in_train')
    print(f"   Loss balancing: EMA escalas fijas tras warmup={_ema_warmup} pasos")

    @tf.function
    def triple_loss(y_true, y_pred):
        # Invertir a mm reales para calcular pesos adaptativos
        y_mm_true = _invert_transform(y_true)
        mean_y_mm = tf.reduce_mean(y_mm_true) + 1e-6

        # Pesos focales: aumentar importancia de eventos extremos
        if _extreme_boost > 1.0:
            focal_w = 1.0 + (_extreme_boost - 1.0) * tf.nn.sigmoid(
                (y_mm_true - high_threshold) / high_width
            )
        else:
            focal_w = tf.ones_like(y_mm_true)

        # ========== Componente 1: Huber focal ==========
        # Huber es robusta a outliers; focal_w amplifica el error en extremos
        huber_fn     = tf.keras.losses.Huber(delta=huber_delta, reduction='none')
        huber_per_px = huber_fn(tf.expand_dims(y_true, -1),
                                tf.expand_dims(y_pred, -1))
        loss_huber   = tf.reduce_mean(huber_per_px * focal_w)

        # ========== Componente 2: Quantil adaptativo ==========
        # Usa diferentes cuantiles según la intensidad de lluvia:
        #   - Zona baja  (< low_threshold mm):  tau=0.50 (mediana)
        #   - Zona media (entre umbrales):       tau=0.60
        #   - Zona alta  (> high_threshold mm):  tau=quantile_tau (asimétrica)
        # Los pesos w_high, w_low, w_mid son suaves (sigmoide)
        if quantile_weight > 0.0:
            w_high_raw = tf.sigmoid((y_mm_true - high_threshold) / high_width)
            w_low_raw  = tf.sigmoid((low_threshold - y_mm_true) / low_width)
            w_mid_raw  = tf.maximum(0.0, 1.0 - w_high_raw - w_low_raw)
            w_sum  = w_high_raw + w_low_raw + w_mid_raw + 1e-8
            w_high = w_high_raw / w_sum
            w_low  = w_low_raw  / w_sum
            w_mid  = w_mid_raw  / w_sum
            e = y_true - y_pred

            def q_term(tau):
                return tf.maximum(tau * e, (tau - 1.0) * e)

            loss_quantile = tf.reduce_mean(
                focal_w * (
                    w_low  * q_term(0.50) +
                    w_mid  * q_term(0.60) +
                    w_high * q_term(quantile_tau)
                )
            )
        else:
            loss_quantile = tf.constant(0.0)

        # ========== Componente 3: Tweedie ==========
        # Modela la distribución Tweedie compound Poisson-Gamma,
        # natural para precipitación (masa en cero + cola continua).
        # p ∈ (1,2): p=1 → Poisson, p=2 → Gamma
        if tweedie_weight > 0.0:
            y_pred_mm = tf.maximum(_invert_transform(y_pred), 1e-6)
            p   = float(tweedie_power)
            p1  = 1.0 - p
            p2  = 2.0 - p
            term1 = -y_mm_true * tf.pow(y_pred_mm, p1) / p1
            term2 = tf.pow(y_pred_mm, p2) / p2
            loss_tweedie_raw = tf.reduce_mean((term1 + term2) * focal_w)
            loss_tweedie = loss_tweedie_raw / _tw_scale
        else:
            loss_tweedie = tf.constant(0.0)

        # ========== Componente 4: Correlación de Pearson ==========
        # Maximiza la correlación espacial entre predicciones y observaciones.
        # loss_corr = 1 - r, donde r es Pearson sobre el batch aplanado.
        if _corr_w > 0.0:
            y_t_flat = tf.reshape(y_true, [-1])
            y_p_flat = tf.reshape(y_pred, [-1])
            mu_t = tf.reduce_mean(y_t_flat)
            mu_p = tf.reduce_mean(y_p_flat)
            dt   = y_t_flat - mu_t
            dp   = y_p_flat - mu_p
            cov  = tf.reduce_mean(dt * dp)
            std_t = tf.sqrt(tf.reduce_mean(dt * dt) + 1e-8)
            std_p = tf.sqrt(tf.reduce_mean(dp * dp) + 1e-8)
            pearson_r = cov / (std_t * std_p)
            loss_corr = 1.0 - pearson_r
        else:
            loss_corr = tf.constant(0.0)

        # ========== Actualización de escalas EMA ==========
        # Solo durante warmup se actualizan las escalas de normalización.
        # Después se congelan para estabilidad.
        is_first      = tf.equal(_ema_step, 0)
        still_warming = tf.logical_and(_in_train_ph,
                                       tf.less(_ema_step, _ema_warmup))

        def _ema_up(ema_var, raw):
            """Actualiza una variable EMA con el valor crudo actual."""
            raw_sg  = tf.stop_gradient(raw)  # No propagar gradientes por la escala
            new_val = tf.where(is_first, raw_sg,
                               _ema_mom * ema_var
                               + (1.0 - _ema_mom) * raw_sg)
            updated = tf.maximum(new_val, _ema_eps)
            ema_var.assign(tf.cond(still_warming,
                                   lambda: updated,
                                   lambda: ema_var.value()))

        _ema_up(_ema_h, loss_huber)
        if quantile_weight > 0.0:
            _ema_up(_ema_q, loss_quantile)
        if tweedie_weight > 0.0:
            _ema_up(_ema_t, loss_tweedie)
        if _corr_w > 0.0:
            _ema_up(_ema_c, loss_corr)
        _ema_step.assign(tf.cond(
            _in_train_ph,
            lambda: tf.minimum(_ema_step + 1, _ema_warmup + 1),
            lambda: _ema_step
        ))

        # ========== Pérdida total ponderada ==========
        # Cada componente se normaliza dividiéndola por su escala EMA,
        # luego se pondera según los pesos del usuario.
        norm_h = loss_huber / tf.stop_gradient(tf.maximum(_ema_h, _ema_eps))
        norm_q = (loss_quantile / tf.stop_gradient(tf.maximum(_ema_q, _ema_eps))
                  if quantile_weight > 0.0 else loss_quantile)
        norm_t = (loss_tweedie / tf.stop_gradient(tf.maximum(_ema_t, _ema_eps))
                  if tweedie_weight > 0.0 else loss_tweedie)
        norm_c = (loss_corr / tf.stop_gradient(tf.maximum(_ema_c, _ema_eps))
                  if _corr_w > 0.0 else loss_corr)

        total = (huber_weight    * norm_h
               + quantile_weight * norm_q
               + tweedie_weight  * norm_t
               + _corr_w         * norm_c)

        if has_smooth:
            pred_i      = tf.gather(y_pred, pairs_i_tf, axis=-1)
            pred_j      = tf.gather(y_pred, pairs_j_tf, axis=-1)
            smooth_loss = tf.reduce_mean(tf.square(pred_i - pred_j))
            total       = total + smooth_weight * smooth_loss

        return total

    triple_loss.__name__ = (
        f'triple_loss'
        f'_hw{huber_weight:.2f}_hd{huber_delta}'
        f'_qw{quantile_weight}_qt{quantile_tau}'
        f'_tw{tweedie_weight}_tp{tweedie_power}'
        f'_lo{low_threshold}_hi{high_threshold}'
        f'_sm{smooth_weight}'
    )
    return triple_loss, _in_train_ph


def make_combined_loss(**kwargs):
    return make_adaptive_loss(**kwargs)


# ════════════════════════════════════════════════════════════════════
#  Procesamiento de predictores — SIEMPRE DESDE DISCO (low-RAM)
# ════════════════════════════════════════════════════════════════════

def _prepare_predictor_lowram(args_tuple):
    """Fase 1: Estadísticas + PCA desde disco, sin cargar predictor completo.
    
    Para cada predictor:
      1. Calcula media y desviación estándar sobre el período de entrenamiento,
         leyendo en chunks temporales desde disco para no exceder la RAM.
      2. Ajusta IncrementalPCA (también por chunks) para reducir dimensionalidad.
         Con pca_components=0, encuentra automáticamente el mínimo de componentes
         que captura la varianza objetivo (por defecto 99.9%).
      3. Guarda el modelo PCA y sus componentes en disco como memmap/joblib.
    
    Se ejecuta en un worker separado con semilla propia para reproducibilidad.
    """
    (
        pred_idx, filepath, varname, domain, pinfo,
        cache_dir, train_years,
        pca_components, pca_samples, max_ram_gb,
        stats_chunk_time, pca_chunk_time,
        worker_seed,
        pca_variance_target,
    ) = args_tuple

    import numpy as np
    import gc
    import os
    import time
    import pickle
    import xarray as xr
    import pandas as pd

    _init_worker_seeds(worker_seed)

    try:
        from sklearn.decomposition import IncrementalPCA
        has_sklearn = True
    except ImportError:
        has_sklearn = False
    try:
        import joblib
        has_joblib = True
    except ImportError:
        has_joblib = False

    t0 = time.time()
    n_valid   = pinfo['n_valid']
    valid_idx = pinfo['valid_indices']
    times_p   = pinfo['times']

    # train_years es una lista de años de entrenamiento
    train_years_set = set(train_years) if train_years else None
    if train_years_set is not None:
        times_years = pd.DatetimeIndex(times_p).year
        train_time_idx = np.where(
            np.isin(times_years, list(train_years_set))
        )[0]
    else:
        train_time_idx = np.arange(len(times_p))

    # ── Calcular chunk_time seguro basado en RAM ────────────────────
    # Cada timestep ocupa n_valid * 4 bytes (float32)
    # Usamos máximo 15% de max_ram_gb para este cálculo,
    # dejando el resto para PCA y otros procesos
    avail_bytes = int(max_ram_gb * 1024**3 * 0.15)
    step_bytes = n_valid * 4
    safe_chunk = max(1, int(avail_bytes / max(step_bytes, 1)))
    chunk = min(safe_chunk, max(1, int(stats_chunk_time)))

    # ── Estadísticas desde disco (chunked) ──────────────────────────
    ds = xr.open_dataset(filepath)
    ds = _normalize_time_dim(ds)
    da = ds[varname]
    da = _normalize_spatial_dims(da)
    if not np.issubdtype(da['time'].dtype, np.datetime64):
        da['time'] = pd.to_datetime(da['time'].values)

    da_train = da.isel(time=train_time_idx) if train_years_set else da
    n_t_train = len(da_train['time'])

    # Muestreo para stats y PCA
    rng = np.random.default_rng(worker_seed)
    n_sample_stats = min(n_t_train, pca_samples)
    if n_sample_stats < n_t_train:
        sample_idx = np.sort(rng.choice(n_t_train, size=n_sample_stats, replace=False))
    else:
        sample_idx = np.arange(n_t_train)

    acc_n = acc_sum = acc_sq = 0.0  # Acumuladores para media/std incremental
    for start in range(0, len(sample_idx), chunk):
        idx_c = sample_idx[start:start + chunk]
        vals  = da_train.isel(time=idx_c).values  # Lee chunk desde disco
        flat_c = vals.reshape(vals.shape[0], -1)[:, valid_idx]  # Extraer solo píxeles válidos
        vv = flat_c[np.isfinite(flat_c)].astype(np.float64)  # Ignorar NaN
        if vv.size > 0:
            acc_n  += vv.size
            acc_sum += float(np.sum(vv))
            acc_sq  += float(np.sum(vv * vv))  # Para var = E[x²] - E[x]²
        del vals, flat_c, vv  # Liberar inmediatamente
        gc.collect()

    # Calcular media y std a partir de las sumas acumuladas
    mean_val = acc_sum / acc_n if acc_n > 0 else 0.0
    std_val  = float(np.sqrt(max((acc_sq / acc_n) - mean_val ** 2, 0.0))) \
               if acc_n > 0 else 1.0
    if std_val < 1e-8:  # Evitar división por cero en normalización
        std_val = 1.0

    stats  = {'mean': mean_val, 'std': std_val, 'varname': varname, 'domain': domain}
    p_mean = mean_val
    p_std  = max(std_val, 1e-8)

    ds.close()
    del da, da_train, ds
    gc.collect()

    # ── PCA ────────────────────
    pca_model = None
    feat_size = n_valid
    auto_pca = (pca_components == 0)
    do_pca = has_sklearn and (pca_components > 0 or auto_pca) and n_valid > 2

    if do_pca:
        if auto_pca:
            # Auto-PCA: usamos IncrementalPCA con n_components grande
            # y luego truncamos por varianza explicada.
            # Máximo componentes que caben en RAM:
            # IncrementalPCA necesita ~(n_components * n_valid * 4) bytes para components_
            # + ~(batch_size * n_valid * 4) para cada batch
            # Presupuesto: 20% de max_ram_gb
            budget_bytes = int(max_ram_gb * 1024**3 * 0.20)
            # Máximo componentes factibles considerando RAM
            max_comp_ram = max(1, int(budget_bytes / (n_valid * 4 * 2)))
            # No más que min(n_samples, n_features) - 1
            max_comp_data = min(len(sample_idx) - 1, n_valid - 1)
            actual_n = min(max_comp_ram, max_comp_data)
            # Piso mínimo razonable
            actual_n = max(actual_n, 1)
            print(f"      [auto-PCA pred_{pred_idx}] n_valid={n_valid}, "
                  f"max_comp_ram={max_comp_ram}, max_comp_data={max_comp_data}, "
                  f"usando {actual_n} componentes iniciales")
        else:
            actual_n = min(pca_components, n_valid - 1, len(sample_idx) - 1)
            actual_n = max(actual_n, 1)

        ipca = IncrementalPCA(n_components=actual_n)

        # Chunk PCA: batch >= actual_n + 1
        pca_batch = max(actual_n + 1, chunk)
        # Pero también limitado por RAM
        pca_batch_bytes = pca_batch * n_valid * 4
        if pca_batch_bytes > avail_bytes:
            pca_batch = max(actual_n + 1, int(avail_bytes / (n_valid * 4)))

        ds2 = xr.open_dataset(filepath)
        ds2 = _normalize_time_dim(ds2)
        da2 = ds2[varname]
        da2 = _normalize_spatial_dims(da2)
        if not np.issubdtype(da2['time'].dtype, np.datetime64):
            da2['time'] = pd.to_datetime(da2['time'].values)
        da2_train = da2.isel(time=train_time_idx) if train_years_set else da2

        n_fitted = 0
        for start in range(0, len(sample_idx), pca_batch):
            idx_c = sample_idx[start:start + pca_batch]
            if len(idx_c) <= actual_n:
                # IncrementalPCA necesita batch > n_components
                continue
            vals  = da2_train.isel(time=idx_c).values
            flat_c = vals.reshape(vals.shape[0], -1)[:, valid_idx].astype(np.float32)
            flat_c = np.nan_to_num(flat_c, nan=p_mean, posinf=p_mean, neginf=p_mean)
            flat_c = (flat_c - p_mean) / p_std
            ipca.partial_fit(flat_c)
            n_fitted += len(idx_c)
            del vals, flat_c
            gc.collect()

        ds2.close()
        del da2, da2_train, ds2
        gc.collect()

        if n_fitted == 0:
            # Fallback: sin PCA si no pudimos ajustar
            pca_model = None
            feat_size = n_valid
            print(f"      [auto-PCA pred_{pred_idx}] FALLBACK: sin PCA "
                  f"(n_fitted=0)")
        else:
            pca_model = ipca

            if auto_pca:
                # Truncar por varianza explicada acumulada
                var_ratio = ipca.explained_variance_ratio_
                cumvar = np.cumsum(var_ratio)
                # Encontrar mínimo componentes para alcanzar objetivo
                idx_ok = np.where(cumvar >= pca_variance_target)[0]
                if len(idx_ok) > 0:
                    n_keep = int(idx_ok[0]) + 1
                else:
                    n_keep = actual_n  # no se alcanzó el objetivo, usar todos
                feat_size = n_keep
                print(f"      [auto-PCA pred_{pred_idx}] Varianza acumulada: "
                      f"{cumvar[-1]:.4f} con {actual_n} comp | "
                      f"Reteniendo {n_keep} comp para {pca_variance_target:.3f} "
                      f"varianza ({cumvar[n_keep-1]:.4f} real)")
            else:
                feat_size = actual_n

    # ── Guardar PCA en disco ────────────────────────────────────────
    pca_comp_path = None
    pca_mean_path = None

    if pca_model is not None:
        pca_base = os.path.join(cache_dir, f'pca_{pred_idx}')
        if has_joblib:
            joblib.dump(pca_model, pca_base + '.joblib')
        else:
            with open(pca_base + '.pkl', 'wb') as f:
                pickle.dump(pca_model, f)

        # Solo guardar los componentes que realmente usamos (truncados)
        comp = pca_model.components_[:feat_size].astype(np.float32)
        pca_comp_path = os.path.join(cache_dir, f'_pca_comp_{pred_idx}.npy')
        mm_c = np.lib.format.open_memmap(
            pca_comp_path, mode='w+', dtype=np.float32, shape=comp.shape
        )
        mm_c[:] = comp
        mm_c.flush()
        del mm_c, comp
        gc.collect()

        if hasattr(pca_model, 'mean_') and pca_model.mean_ is not None:
            pca_mean_path = os.path.join(cache_dir, f'_pca_mean_{pred_idx}.npy')
            np.save(pca_mean_path, pca_model.mean_.astype(np.float32))

    del pca_model
    gc.collect()

    elapsed = time.time() - t0
    msg = (f"\u2713 {varname} [pred_{pred_idx}]: stats+PCA "
           f"({feat_size} feats, from {n_valid} valid) | {elapsed:.1f}s | "
           f"disk-chunked | seed={worker_seed}")

    return {
        'pred_idx':      pred_idx,
        'stats':         stats,
        'feat_size':     feat_size,
        'use_ram':       False,  # Siempre False en low-ram
        'data_ram_path': None,   # Nunca usamos RAM completa
        'pca_comp_path': pca_comp_path,
        'pca_mean_path': pca_mean_path,
        'msg':           msg,
    }


def _cache_months_chunk_lowram(args_tuple):
    """Fase 2 LOW-RAM: Cachea ventanas temporales mensuales en disco.
    
    Para cada mes válido, lee los `input_days` días previos del predictor,
    normaliza (z-score), aplica PCA si corresponde, y escribe el resultado
    en un archivo memmap compartido.
    
    Cada worker procesa un rango de meses [mi_start, mi_end) de forma
    independiente, escribiendo directamente al memmap sin conflictos.
    """
    (
        pred_idx, mi_start, mi_end,
        valid_months_chunk,
        input_days, feat_size, out_path,
        times_p,
        filepath, varname, valid_idx, n_valid,
        p_mean, p_std, pca_comp_path, pca_mean_path,
        worker_seed,
    ) = args_tuple

    import numpy as np
    import xarray as xr
    import pandas as pd
    from datetime import timedelta

    _init_worker_seeds(worker_seed)

    out = np.lib.format.open_memmap(out_path, mode='r+')

    pca_comp = None
    pca_mean_vec = None
    if pca_comp_path is not None:
        pca_comp = np.load(pca_comp_path, mmap_mode='r')
    if pca_mean_path is not None:
        pca_mean_vec = np.load(pca_mean_path)

    ds = xr.open_dataset(filepath)
    ds = _normalize_time_dim(ds)
    da = ds[varname]
    da = _normalize_spatial_dims(da)
    if not np.issubdtype(da['time'].dtype, np.datetime64):
        da['time'] = pd.to_datetime(da['time'].values)

    for ci, mi in enumerate(range(mi_start, mi_end)):
        m_start, m_end_ts = valid_months_chunk[ci]
        # Calcular ventana de entrada: input_days días ANTES del inicio del mes
        input_end   = m_start - timedelta(days=1)       # Último día antes del mes
        input_start = input_end - timedelta(days=input_days - 1)  # Primer día de la ventana
        try:
            window = da.sel(time=slice(input_start, input_end)).values
        except Exception:
            times_a = da['time'].values
            i_s = int(np.argmin(np.abs(times_a - np.datetime64(input_start))))
            i_e = int(np.argmin(np.abs(times_a - np.datetime64(input_end))))
            window = da.isel(time=slice(i_s, i_e + 1)).values

        if window.ndim == 2:
            window = np.repeat(window[np.newaxis], input_days, axis=0)

        T = window.shape[0]
        # Rellenar con la media si hay menos días que input_days (e.g. inicio del dataset)
        if T < input_days:
            pad = np.full((input_days - T,) + window.shape[1:],
                          p_mean, dtype=np.float32)
            window = np.concatenate([window, pad], axis=0)
        elif T > input_days:  # Truncar si hay más días de los necesarios
            window = window[:input_days]

        # Aplanar grilla espacial y extraer solo píxeles válidos
        flat = window.reshape(window.shape[0], -1)[:, valid_idx].astype(np.float32)
        flat = np.nan_to_num(flat, nan=p_mean, posinf=p_mean, neginf=p_mean)  # NaN → media
        flat = (flat - p_mean) / p_std  # Normalizar (z-score)

        # Proyectar a espacio PCA si hay componentes disponibles
        if pca_comp is not None:
            if pca_mean_vec is not None:
                flat = flat - pca_mean_vec  # Centrar con la media de PCA
            flat = (flat @ pca_comp.T).astype(np.float32)  # Transformar: (T, n_valid) → (T, n_pca)

        out[mi] = flat
        del window, flat

    ds.close()
    out.flush()
    del out
    return pred_idx, mi_end - mi_start


#  DataPreprocessor LOW-RAM
class DataPreprocessor:

    def __init__(self, cache_dir, dtype='float32'):
        self.cache_dir = cache_dir
        self.np_dtype  = np.float16 if dtype == 'float16' else np.float32
        os.makedirs(cache_dir, exist_ok=True)

    def is_cached(self):
        return os.path.isfile(os.path.join(self.cache_dir, 'metadata.json'))
    
    def is_base_cached(self):
        """Verifica si existe cache base (P1-P4: stats + PCA + target, sin división de años)"""
        return os.path.isfile(os.path.join(self.cache_dir, 'base_config.json'))
    
    def _get_base_config_hash(self, precip_path, predictor_configs, input_days, 
                              pca_components, pca_samples, target_transform,
                              pca_variance_target, seed):
        """Calcula hash de parámetros que determinan P1-P4"""
        import hashlib
        config_str = (
            f"{precip_path}|"
            f"{sorted(predictor_configs)}|"
            f"{input_days}|{pca_components}|{pca_samples}|"
            f"{target_transform}|{pca_variance_target}|{seed}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def preprocess(self, precip_path, precip_var, predictor_configs,
                   input_days, split_frac, pca_components, pca_samples,
                   target_transform, stats_chunk_time=64, pca_chunk_time=32,
                   n_jobs=1, max_ram_gb=4.0, seed=42, seed_split=None,
                   pca_variance_target=0.999):

        print("\n" + "=" * 70)
        print("  FASE DE PREPROCESAMIENTO (LOW-RAM)")
        print("=" * 70)
        ram = get_ram()
        ram.report("inicio preprocessing")

        print("\n[P1] Analizando precipitación (target)...")
        tinfo = self._analyze_target(precip_path, precip_var)
        gc.collect()

        print("\n[P2] Analizando predictores...")
        pinfos = []
        for filepath, varname, domain in predictor_configs:
            info = self._analyze_predictor(filepath, varname, domain)
            pinfos.append(info)
            gc.collect()

        print("\n[P3] Calculando meses válidos...")
        valid_months = self._compute_valid_months(tinfo, pinfos, input_days)
        n_months = len(valid_months)

        # ── Split por AÑOS aleatorios (80% train / 20% val) ────────
        all_years = sorted(set(m_start.year for m_start, _ in valid_months))
        seed_split_actual = seed_split if seed_split is not None else seed
        rng_split = random.Random(seed_split_actual)
        n_val_years = max(1, int(len(all_years) * (1.0 - split_frac)))
        val_years = set(rng_split.sample(all_years, n_val_years))
        train_years = set(all_years) - val_years

        train_indices = sorted(
            i for i, (ms, _) in enumerate(valid_months)
            if ms.year in train_years
        )
        val_indices = sorted(
            i for i, (ms, _) in enumerate(valid_months)
            if ms.year in val_years
        )
        n_train = len(train_indices)
        n_val   = len(val_indices)

        print(f"   Total: {n_months} meses | Train: {n_train} | Val: {n_val}")
        print(f"   Años train ({len(train_years)}): "
              f"{sorted(train_years)}")
        print(f"   Años val   ({len(val_years)}): "
              f"{sorted(val_years)}")

        n_preds     = len(predictor_configs)
        actual_jobs = (max(1, os.cpu_count() // 2)
                       if n_jobs == -1 else max(n_jobs, 1))

        # LOW-RAM: Procesar predictores UNO A UNO secuencialmente
        # para evitar que múltiples workers compitan por RAM
        print(f"\n[P4] Procesando {n_preds} predictores (LOW-RAM, secuencial)...")
        ram.report("antes P4")

        print(f"\n  [P4a] Stats + PCA (secuencial, 1 predictor a la vez)...")
        phase1_results = [None] * n_preds
        all_stats  = [None] * n_preds
        feat_sizes = [None] * n_preds
        pred_names = [cfg[1] for cfg in predictor_configs]

        t4_start = time.time()

        for i, (filepath, varname, domain) in enumerate(predictor_configs):
            ram_est = estimate_predictor_ram_gb(pinfos[i])
            w_seed  = seed + i + 1
            print(f"\n   Predictor {i} ({varname}): "
                  f"{pinfos[i]['n_valid']} px válidos, "
                  f"~{ram_est:.2f} GB on-disk, seed={w_seed}")
            ram.report(f"antes pred_{i}")

            task = (
                i, filepath, varname, domain, pinfos[i],
                self.cache_dir, sorted(train_years),
                pca_components, pca_samples, max_ram_gb,
                stats_chunk_time, pca_chunk_time,
                w_seed,
                pca_variance_target,
            )
            r = _prepare_predictor_lowram(task)
            phase1_results[r['pred_idx']] = r
            all_stats[r['pred_idx']]  = r['stats']
            feat_sizes[r['pred_idx']] = r['feat_size']
            print(f"   {r['msg']}")
            gc.collect()
            ram.report(f"después pred_{i}")

        t4a_elapsed = time.time() - t4_start
        print(f"\n   Fase 1 completa en {t4a_elapsed:.1f}s")

        # Crear archivos memmap de salida
        for i in range(n_preds):
            out_path = os.path.join(self.cache_dir, f'pred_{i}.npy')
            mm = np.lib.format.open_memmap(
                out_path, mode='w+', dtype=np.float32,
                shape=(n_months, input_days, feat_sizes[i])
            )
            del mm
        gc.collect()

        # Fase 2: cachear ventanas mensuales
        # LOW-RAM: Procesar UN predictor a la vez, con workers limitados
        print(f"\n  Cacheando ventanas (secuencial por predictor, "
              f"{actual_jobs} workers/predictor)...")
        ram.report("antes P4b")

        t4b_start = time.time()
        total_months_all = n_months * n_preds
        completed_total = [0]

        for i, (filepath, varname, domain) in enumerate(predictor_configs):
            r = phase1_results[i]
            p_mean_i = r['stats']['mean']
            p_std_i  = max(r['stats']['std'], 1e-8)
            w_seed   = seed + i + 1

            # Calcular chunks para este predictor
            n_chunks_target = max(actual_jobs * 2, 8)
            chunk_sz = max(1, (n_months + n_chunks_target - 1) // n_chunks_target)
            tasks = []
            for ci_start in range(0, n_months, chunk_sz):
                ci_end = min(ci_start + chunk_sz, n_months)
                months_chunk = valid_months[ci_start:ci_end]
                tasks.append((
                    i, ci_start, ci_end,
                    months_chunk,
                    input_days, feat_sizes[i],
                    os.path.join(self.cache_dir, f'pred_{i}.npy'),
                    pinfos[i]['times'],
                    filepath, varname,
                    pinfos[i]['valid_indices'], pinfos[i]['n_valid'],
                    p_mean_i, p_std_i,
                    r.get('pca_comp_path'), r.get('pca_mean_path'),
                    w_seed,
                ))

            if actual_jobs <= 1:
                for task in tasks:
                    _, n_done = _cache_months_chunk_lowram(task)
                    completed_total[0] += n_done
                    pct = 100 * completed_total[0] / total_months_all
                    print(f"\r   Pred {i}: {completed_total[0]}/{total_months_all} "
                          f"ventanas totales ({pct:.0f}%)", end='', flush=True)
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=actual_jobs,
                    mp_context=multiprocessing.get_context('forkserver'),
                ) as executor:
                    futures = [executor.submit(_cache_months_chunk_lowram, t)
                               for t in tasks]
                    for fut in concurrent.futures.as_completed(futures):
                        try:
                            _, n_done = fut.result()
                            completed_total[0] += n_done
                            pct = 100 * completed_total[0] / total_months_all
                            print(f"\r   Pred {i}: {completed_total[0]}/{total_months_all} "
                                  f"ventanas totales ({pct:.0f}%)", end='', flush=True)
                        except Exception as exc:
                            print(f"\n   Fallo en Fase 2 pred {i}: {exc}")
                            raise

            # Limpiar temporales de este predictor
            for pattern in [f'_pca_comp_{i}.npy', f'_pca_mean_{i}.npy']:
                tmp = os.path.join(self.cache_dir, pattern)
                if os.path.exists(tmp):
                    os.remove(tmp)
            gc.collect()

        print()
        t4b_elapsed = time.time() - t4b_start

        t4_elapsed = time.time() - t4_start
        for i in range(n_preds):
            out_path = os.path.join(self.cache_dir, f'pred_{i}.npy')
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"   ✓ pred_{i} ({predictor_configs[i][1]}): "
                  f"({n_months}×{input_days}×{feat_sizes[i]}) = {size_mb:.1f} MB")
        print(f"\n   P4 completo en {t4_elapsed:.1f}s "
              f"(Fase1={t4a_elapsed:.1f}s + Fase2={t4b_elapsed:.1f}s)")
        ram.report("después P4")

        print(f"\n[P5] Procesando target (precipitación)...")
        target_stats = self._cache_target(
            precip_path, precip_var, tinfo,
            valid_months, target_transform, train_indices
        )
        gc.collect()
        ram.report("después P5")

        # Calcular train_end como el timestamp del último mes de entrenamiento
        train_end_idx = max(train_indices) if train_indices else 0
        train_end_timestamp = [str(ms) for ms, _ in valid_months][train_end_idx]
        
        metadata = {
            'version':              6,
            'n_months':             n_months,
            'n_train':              n_train,
            'n_val':                n_val,
            'train_indices':        train_indices,
            'val_indices':          val_indices,
            'train_years':          sorted(train_years),
            'val_years':            sorted(val_years),
            'train_end':            train_end_timestamp,
            'input_days':           input_days,
            'n_land':               tinfo['n_land'],
            'target_h':             tinfo['h'],
            'target_w':             tinfo['w'],
            'flat_indices':         tinfo['flat_indices'].tolist(),
            'lat_coords':           tinfo['lat_coords'].tolist(),
            'lon_coords':           tinfo['lon_coords'].tolist(),
            'lat_name':             tinfo['lat_name'],
            'lon_name':             tinfo['lon_name'],
            'predictor_stats':      all_stats,
            'predictor_feat_sizes': feat_sizes,
            'predictor_names':      pred_names,
            'predictor_n_valid':    [p['n_valid'] for p in pinfos],
            'target_transform':     target_transform,
            'target_mean':          target_stats.get('mean'),
            'target_std':           target_stats.get('std'),
            'pca_components':       pca_components,
            'pca_variance_target':  pca_variance_target,
            'month_timestamps':     [str(ms) for ms, _ in valid_months],
            'split_frac':           split_frac,
            'n_predictors':         len(predictor_configs),
            'seed':                 seed,
            'seed_split':           seed_split_actual,
        }
        with open(os.path.join(self.cache_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Guardar base_config.json para reutilización con diferentes seed_split
        base_config = {
            'base_config_hash':     self._get_base_config_hash(precip_path, predictor_configs, 
                                                               input_days, pca_components, 
                                                               pca_samples, target_transform,
                                                               pca_variance_target, seed),
            'n_months':             n_months,
            'month_timestamps':     [str(ms) for ms, _ in valid_months],
            'all_years':            sorted(all_years),
            'split_frac':           split_frac,
            'n_land':               tinfo['n_land'],
            'target_h':             tinfo['h'],
            'target_w':             tinfo['w'],
            'flat_indices':         tinfo['flat_indices'].tolist(),
            'lat_coords':           tinfo['lat_coords'].tolist(),
            'lon_coords':           tinfo['lon_coords'].tolist(),
            'lat_name':             tinfo['lat_name'],
            'lon_name':             tinfo['lon_name'],
            'predictor_stats':      all_stats,
            'predictor_feat_sizes': feat_sizes,
            'predictor_names':      pred_names,
            'predictor_n_valid':    [p['n_valid'] for p in pinfos],
            'target_transform':     target_transform,
            'target_mean':          target_stats.get('mean'),
            'target_std':           target_stats.get('std'),
            'pca_components':       pca_components,
            'pca_variance_target':  pca_variance_target,
            'n_predictors':         len(predictor_configs),
            'input_days':           input_days,
        }
        with open(os.path.join(self.cache_dir, 'base_config.json'), 'w') as f:
            json.dump(base_config, f, indent=2, default=str)

        total_mb = sum(
            os.path.getsize(os.path.join(self.cache_dir, fn)) / (1024 * 1024)
            for fn in os.listdir(self.cache_dir)
            if os.path.isfile(os.path.join(self.cache_dir, fn))
        )
        print(f"\n   ✓ Cache completo: {self.cache_dir}/ ({total_mb:.1f} MB)")
        return metadata
    
    def _quick_resplit(self, base_config, split_frac, seed_split):
        """Reutiliza cache base (P1-P4) pero recalcula división de años (P5 simplificado).
        Mucho más rápido que preprocessing completo."""
        print("\n" + "=" * 70)
        print("  RAPIDISPLIT: Reutilizando cache base, solo recalculando división...")
        print("=" * 70)
        
        month_timestamps = base_config['month_timestamps']
        all_years = base_config['all_years']
        n_months = base_config['n_months']
        
        # Recalcular división con nuevo seed_split
        rng_split = random.Random(seed_split)
        n_val_years = max(1, int(len(all_years) * (1.0 - split_frac)))
        val_years = set(rng_split.sample(all_years, n_val_years))
        train_years = set(all_years) - val_years
        
        # Mapear años a índices
        timestamps = [pd.Timestamp(t) for t in month_timestamps]
        train_indices = sorted([i for i, t in enumerate(timestamps) if t.year in train_years])
        val_indices = sorted([i for i, t in enumerate(timestamps) if t.year in val_years])
        
        n_train = len(train_indices)
        n_val = len(val_indices)
        
        train_end_idx = max(train_indices) if train_indices else 0
        train_end_timestamp = month_timestamps[train_end_idx]
        
        print(f"   Total: {n_months} meses | Train: {n_train} | Val: {n_val}")
        print(f"   Años train ({len(train_years)}): {sorted(train_years)}")
        print(f"   Años val   ({len(val_years)}): {sorted(val_years)}")
        
        # Construir metadata con la nueva división
        metadata = {
            'version':              6,
            'n_months':             n_months,
            'n_train':              n_train,
            'n_val':                n_val,
            'train_indices':        train_indices,
            'val_indices':          val_indices,
            'train_years':          sorted(train_years),
            'val_years':            sorted(val_years),
            'train_end':            train_end_timestamp,
            'input_days':           base_config['input_days'],
            'n_land':               base_config['n_land'],
            'target_h':             base_config['target_h'],
            'target_w':             base_config['target_w'],
            'flat_indices':         base_config['flat_indices'],
            'lat_coords':           base_config['lat_coords'],
            'lon_coords':           base_config['lon_coords'],
            'lat_name':             base_config['lat_name'],
            'lon_name':             base_config['lon_name'],
            'predictor_stats':      base_config['predictor_stats'],
            'predictor_feat_sizes': base_config['predictor_feat_sizes'],
            'predictor_names':      base_config['predictor_names'],
            'predictor_n_valid':    base_config['predictor_n_valid'],
            'target_transform':     base_config['target_transform'],
            'target_mean':          base_config['target_mean'],
            'target_std':           base_config['target_std'],
            'pca_components':       base_config['pca_components'],
            'pca_variance_target':  base_config['pca_variance_target'],
            'month_timestamps':     month_timestamps,
            'split_frac':           split_frac,
            'n_predictors':         base_config['n_predictors'],
            'seed_split':           seed_split,
        }
        
        with open(os.path.join(self.cache_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   ✓ División recalculada y guardada en metadata.json")
        return metadata

    def _analyze_target(self, path, varname):
        ds = xr.open_dataset(path)
        ds = _normalize_time_dim(ds)
        da = ds[varname]
        da = _normalize_spatial_dims(da)
        if not np.issubdtype(da['time'].dtype, np.datetime64):
            da['time'] = pd.to_datetime(da['time'].values)
        times = da['time'].values
        if 'lat' in da.dims:
            h, w = len(da['lat']), len(da['lon'])
            lat_c, lon_c = da['lat'].values, da['lon'].values
            lat_name, lon_name = 'lat', 'lon'
        else:
            h, w = len(da['Y']), len(da['X'])
            lat_c, lon_c = da['Y'].values, da['X'].values
            lat_name, lon_name = 'Y', 'X'
        sample    = da.isel(time=slice(0, min(30, len(times)))).values
        land_mask = ~np.all(np.isnan(sample), axis=0)
        flat_idx  = np.where(land_mask.ravel())[0].astype(np.int32)
        n_land    = len(flat_idx)
        print(f"   Grid: {h}×{w}, tierra: {n_land}/{h*w} "
              f"({100*n_land/(h*w):.1f}%)")
        print(f"   Tiempo: {times[0]} → {times[-1]}")
        ds.close()
        del sample
        gc.collect()
        return {
            'times': times, 'h': h, 'w': w,
            'lat_coords': lat_c, 'lon_coords': lon_c,
            'lat_name': lat_name, 'lon_name': lon_name,
            'land_mask': land_mask, 'flat_indices': flat_idx,
            'n_land': n_land,
        }

    def _analyze_predictor(self, filepath, varname, domain):
        da, ds = _open_predictor_da(filepath, varname)
        times = da['time'].values
        h = len(da['lat']) if 'lat' in da.dims else da.shape[1]
        w = len(da['lon']) if 'lon' in da.dims else da.shape[2]
        if domain == 'auto':
            domain = detect_variable_domain(da, varname)
        validity  = create_validity_mask(da)
        valid_idx = np.where(validity.ravel())[0].astype(np.int32)
        n_valid   = len(valid_idx)
        print(f"   {varname}: {h}×{w} = {h*w} total, "
              f"{n_valid} válidos ({100*n_valid/(h*w):.1f}%), "
              f"dominio={domain}")
        ds.close()
        gc.collect()
        return {
            'times': times, 'h': h, 'w': w,
            'domain': domain,
            'n_pixels': h * w,
            'n_valid': n_valid,
            'valid_indices': valid_idx,
        }

    def _compute_valid_months(self, tinfo, pinfos, input_days):
        times = tinfo['times']
        month_ranges = get_month_ranges(times)
        valid = []
        for m_start, m_end in month_ranges:
            input_end   = m_start - timedelta(days=1)
            input_start = input_end - timedelta(days=input_days - 1)
            if not (np.datetime64(input_start) >= times[0]
                    and np.datetime64(m_end) <= times[-1]):
                continue
            ok = True
            for pi in pinfos:
                pt = pi['times']
                if pt is not None:
                    if (np.datetime64(input_start) < pt[0]
                            or np.datetime64(input_end) > pt[-1]):
                        ok = False
                        break
            if ok:
                valid.append((m_start, m_end))
        return valid

    def _cache_target(self, path, varname, tinfo,
                      valid_months, transform, train_indices):
        n_months = len(valid_months)
        n_land   = tinfo['n_land']
        flat_idx = tinfo['flat_indices']
        times    = tinfo['times']
        out_path = os.path.join(self.cache_dir, 'target.npy')
        out = np.lib.format.open_memmap(
            out_path, mode='w+', dtype=np.float32,
            shape=(n_months, n_land)
        )
        ds = xr.open_dataset(path)
        ds = _normalize_time_dim(ds)
        da = ds[varname]
        da = _normalize_spatial_dims(da)
        target_mean = target_std = None
        train_set = set(train_indices)
        if transform == 'standard':
            y_train_vals = []
            for mi in train_indices:
                m_start, m_end = valid_months[mi]
                ts = int(np.argmin(np.abs(times - np.datetime64(m_start))))
                te = int(np.argmin(np.abs(times - np.datetime64(m_end))))
                yg = da.isel(time=slice(ts, te + 1)).values
                y_train_vals.append(
                    np.nansum(yg, axis=0).astype(np.float32).ravel()[flat_idx]
                )
                del yg
                gc.collect()
            if y_train_vals:
                y_all = np.concatenate(y_train_vals)
                target_mean = float(np.nanmean(y_all))
                target_std  = float(np.nanstd(y_all))
                if target_std < 1e-8:
                    target_std = 1.0
                print(f"   Target stats: mean={target_mean:.4f}, std={target_std:.4f}")
                del y_all, y_train_vals
                gc.collect()
        for mi, (m_start, m_end) in enumerate(valid_months):
            ts = int(np.argmin(np.abs(times - np.datetime64(m_start))))
            te = int(np.argmin(np.abs(times - np.datetime64(m_end))))
            yg    = da.isel(time=slice(ts, te + 1)).values
            y_sum = np.nansum(yg, axis=0).astype(np.float32)
            if transform == 'log1p':
                y_out = np.log1p(np.maximum(y_sum, 0.0))
            elif transform == 'cbrt':
                y_out = np.cbrt(np.maximum(y_sum, 0.0))
            elif transform == 'pow025':
                y_out = np.power(np.maximum(y_sum, 0.0), 0.25)
            elif transform == 'pow05':
                y_out = np.sqrt(np.maximum(y_sum, 0.0))
            elif transform == 'standard' and target_mean is not None:
                y_out = (y_sum - target_mean) / max(target_std, 1e-8)
            else:
                y_out = y_sum
            out[mi] = y_out.ravel()[flat_idx].astype(np.float32)
            del yg, y_sum, y_out
            if (mi + 1) % 100 == 0 or (mi + 1) == n_months:
                print(f"     Target: {mi+1}/{n_months} meses", end='\r')
        out.flush()
        del out
        ds.close()
        gc.collect()
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"   ✓ target.npy ({n_months}×{n_land}) = {size_mb:.1f} MB")
        return {'mean': target_mean, 'std': target_std}



#  tf.data Dataset — baja RAM con lectura mínima desde memmap
def create_tf_dataset(cache_dir, metadata, indices, batch_size,
                      shuffle=True, prefetch_n=2, seed=42,
                      noise_std=0.0, mixup_alpha=0.0):
    """Crea un tf.data.Dataset que lee desde memmaps en disco.
    
    Args:
        cache_dir: Directorio con los archivos .npy cacheados
        metadata: Dict con configuración (n_predictors, input_days, etc.)
        indices: Índices de meses a incluir (train o val)
        batch_size: Tamaño de batch para el entrenamiento
        shuffle: Si True, baraja los índices cada época
        noise_std: Desviación del ruido gaussiano para augmentation
        mixup_alpha: Parámetro alpha de distribución Beta para mixup
    
    Returns:
        (dataset, pred_mmaps, target_mmap) — el dataset y referencias a los memmaps
    """
    n_preds    = metadata['n_predictors']
    input_days = metadata['input_days']
    n_land     = metadata['n_land']
    feat_sizes = metadata['predictor_feat_sizes']

    # Abrir memmaps en modo read-only (no consume RAM extra;
    # la OS maneja las páginas vía page cache de forma transparente)
    pred_mmaps = [
        np.load(os.path.join(cache_dir, f'pred_{i}.npy'), mmap_mode='r')
        for i in range(n_preds)
    ]
    target_mmap = np.load(
        os.path.join(cache_dir, 'target.npy'), mmap_mode='r'
    )
    indices_arr   = np.array(indices, dtype=np.int32)
    epoch_counter = [0]

    do_noise = shuffle and noise_std > 0.0   # Solo augmentation en train
    do_mixup = shuffle and mixup_alpha > 0.0  # Mixup: mezcla dos muestras

    x_spec = {
        f'pred_{k}': tf.TensorSpec(
            shape=(input_days, feat_sizes[k]), dtype=tf.float32
        )
        for k in range(n_preds)
    }
    y_spec = tf.TensorSpec(shape=(n_land,), dtype=tf.float32)

    def generator():
        """Generador que produce (x_dict, y) leyendo del memmap.
        Baraja los índices cada época con una semilla única por época
        para reproducibilidad determinista."""
        order = np.array(indices_arr)
        rng = np.random.default_rng(seed + epoch_counter[0])
        if shuffle:
            rng.shuffle(order)
            epoch_counter[0] += 1
        for idx_pos in range(len(order)):
            mi = order[idx_pos]
            # Lee directamente del memmap — la OS se encarga del page cache
            x = {f'pred_{k}': np.array(pred_mmaps[k][mi], dtype=np.float32)
                 for k in range(n_preds)}
            y = np.array(target_mmap[mi], dtype=np.float32)

            # Ruido gaussiano: pequeña perturbación para regularización
            if do_noise:
                for k in range(n_preds):
                    x[f'pred_{k}'] = x[f'pred_{k}'] + rng.normal(
                        0.0, noise_std, size=x[f'pred_{k}'].shape
                    ).astype(np.float32)

            # Mixup: interpolación convexa de dos muestras (30% de probabilidad)
            # Mejora generalización al crear muestras sintéticas intermedias
            if do_mixup and len(order) > 1 and rng.random() < 0.3:
                mi2 = order[rng.integers(len(order))]
                lam = float(rng.beta(mixup_alpha, mixup_alpha))
                x2 = {f'pred_{k}': np.array(pred_mmaps[k][mi2], dtype=np.float32)
                       for k in range(n_preds)}
                y2 = np.array(target_mmap[mi2], dtype=np.float32)
                for k in range(n_preds):
                    x[f'pred_{k}'] = lam * x[f'pred_{k}'] + (1 - lam) * x2[f'pred_{k}']
                y = lam * y + (1 - lam) * y2

            yield x, y

    ds_base = tf.data.Dataset.from_generator(
        generator, output_signature=(x_spec, y_spec)
    ).batch(batch_size)

    if shuffle:
        ds_base = ds_base.repeat()

    return ds_base.prefetch(prefetch_n), pred_mmaps, target_mmap


#  LR Schedule y Gradient Accumulation
class WarmupCosineDecayWarmRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule con 3 fases:
    1. Warmup lineal: sube de 0 a initial_lr en warmup_steps pasos
    2. Cosine decay: baja suavemente hacia min_lr siguiendo coseno
    3. Warm restarts: reinicia el ciclo, cada vez multiplicando la
       duración del ciclo por t_mult para una exploración más fina.
    """
    def __init__(self, initial_lr, warmup_steps, total_steps,
                 t0_steps=None, t_mult=2.0, min_lr=1e-6):
        super().__init__()
        self.initial_lr   = initial_lr
        self.warmup_steps = float(max(warmup_steps, 1))
        self.total_steps  = float(max(total_steps, warmup_steps + 1))
        self.t0_steps     = float(t0_steps if t0_steps else
                                  max((total_steps - warmup_steps) / 5.0, 10))
        self.t_mult       = float(max(t_mult, 1.0))
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Fase 1: warmup lineal (LR crece proporcional al paso)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)

        # Fase 2+3: determinar en qué ciclo y posición dentro del ciclo estamos
        t = step - self.warmup_steps
        if self.t_mult == 1.0:
            cycle_len = self.t0_steps
            t_in_cycle = tf.math.floormod(t, cycle_len)
        else:
            n = tf.math.floor(
                tf.math.log(t / self.t0_steps * (self.t_mult - 1.0) + 1.0)
                / tf.math.log(self.t_mult)
            )
            n = tf.maximum(n, 0.0)
            cumul = self.t0_steps * (tf.pow(self.t_mult, n) - 1.0) / (self.t_mult - 1.0)
            cycle_len = self.t0_steps * tf.pow(self.t_mult, n)
            t_in_cycle = t - cumul
            t_in_cycle = tf.maximum(t_in_cycle, 0.0)

        cosine_lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
            1.0 + tf.cos(np.pi * t_in_cycle / tf.maximum(cycle_len, 1.0))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'initial_lr':   self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps':  self.total_steps,
            't0_steps':     self.t0_steps,
            't_mult':       self.t_mult,
            'min_lr':       self.min_lr,
        }


class SWACallback(tf.keras.callbacks.Callback):
    """Stochastic Weight Averaging: promedia los pesos del modelo
    durante el último 25% de las épocas (configurable con swa_start_frac).
    Produce modelos más robustos con mejor generalización al suavizar
    el landscape de la pérdida."""
    def __init__(self, swa_start_frac=0.75):
        super().__init__()
        self.swa_start_frac = swa_start_frac
        self.swa_weights = None
        self.swa_count = 0

    def on_epoch_end(self, epoch, logs=None):
        total_epochs = self.params.get('epochs', 1)
        if epoch < int(total_epochs * self.swa_start_frac):
            return
        current_weights = self.model.get_weights()
        if self.swa_weights is None:
            self.swa_weights = [w.copy() for w in current_weights]
            self.swa_count = 1
        else:
            self.swa_count += 1
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] += (current_weights[i] - self.swa_weights[i]) / self.swa_count

    def on_train_end(self, logs=None):
        if self.swa_weights is not None and self.swa_count > 1:
            self.model.set_weights(self.swa_weights)
            print(f"   [SWA] Pesos promediados de {self.swa_count} épocas aplicados.")


class TrainingPhaseCallback(tf.keras.callbacks.Callback):
    """Cambia el flag _in_train_ph entre train/validation.
    Esto controla si las escalas EMA del loss se actualizan (solo en train)
    o se congelan (en validation)."""
    def __init__(self, in_train_flag):
        super().__init__()
        self._flag = in_train_flag

    def on_train_batch_begin(self, batch, logs=None):
        self._flag.assign(True)

    def on_test_batch_begin(self, batch, logs=None):
        self._flag.assign(False)

    def on_test_batch_end(self, batch, logs=None):
        self._flag.assign(True)


class RAMWatchdogCallback(tf.keras.callbacks.Callback):
    """Monitorea RAM cada N batches y ejecuta gc.collect() si necesario."""
    def __init__(self, check_every=50):
        super().__init__()
        self.check_every = check_every
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.check_every == 0:
            gc.collect()


class GradientAccumulationModel(Model):
    """Modelo con gradient accumulation: acumula gradientes de N mini-batches
    antes de aplicar una actualización, simulando un batch_size N veces mayor.
    Útil cuando el batch completo no cabe en RAM/VRAM."""
    def __init__(self, accum_steps=1, **kwargs):
        super().__init__(**kwargs)
        self.accum_steps = accum_steps

    def train_step(self, data):
        if self.accum_steps <= 1:
            return super().train_step(data)
        x, y = data
        accum_grads = [tf.zeros_like(v) for v in self.trainable_variables]
        total_loss  = 0.0
        for _ in range(self.accum_steps):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss   = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
            accum_grads = [
                ag + (g if g is not None else tf.zeros_like(ag))
                for ag, g in zip(accum_grads, grads)
            ]
            total_loss += loss
        accum_grads = [g / float(self.accum_steps) for g in accum_grads]
        self.optimizer.apply_gradients(zip(accum_grads, self.trainable_variables))
        y_pred_eval = self(x, training=False)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(total_loss / float(self.accum_steps))
            else:
                metric.update_state(y, y_pred_eval)
        return {m.name: m.result() for m in self.metrics}

#  Capas personalizadas
class PixelBias(tf.keras.layers.Layer):
    """Sesgo aprendible por píxel terrestre.
    Captura la climatología local: cada punto de grilla tiene su propio
    offset que se suma a la predicción, permitiendo ajuste fino espacial."""
    def __init__(self, n_land, **kwargs):
        super().__init__(**kwargs)
        self.n_land = n_land

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(self.n_land,), initializer='zeros',
            trainable=True, name='pixel_bias'
        )

    def call(self, inputs):
        return inputs + self.bias

    def get_config(self):
        cfg = super().get_config()
        cfg['n_land'] = self.n_land
        return cfg


class CoordinateModulation(tf.keras.layers.Layer):
    """Modulación de la predicción según coordenadas geográficas.
    Aprende una función scale(lat,lon)*x + shift(lat,lon) que permite
    al modelo ajustar la magnitud y el sesgo de la predicción según
    la ubicación geográfica del píxel. Inicializado en identidad
    (scale=1, shift=0) para no perturbar al inicio del entrenamiento."""
    def __init__(self, coords_data, embed_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.coords_data = np.array(coords_data, dtype=np.float32)
        self.embed_dim   = embed_dim
        self.n_land      = self.coords_data.shape[0]

    def build(self, input_shape):
        self.coords = self.add_weight(
            shape=(self.n_land, 2),
            initializer=tf.keras.initializers.Constant(self.coords_data),
            trainable=False, name='pixel_coords'
        )
        self.coord_dense1 = Dense(self.embed_dim, activation='relu',
                                  name='coord_embed_1')
        self.coord_dense2 = Dense(self.embed_dim, activation='relu',
                                  name='coord_embed_2')
        self.scale_layer  = Dense(1, activation='linear',
                                  kernel_initializer='zeros',
                                  bias_initializer='ones',
                                  name='coord_scale')
        self.shift_layer  = Dense(1, activation='linear',
                                  kernel_initializer='zeros',
                                  bias_initializer='zeros',
                                  name='coord_shift')
        super().build(input_shape)

    def call(self, x):
        h      = self.coord_dense2(self.coord_dense1(self.coords))
        scale  = tf.squeeze(self.scale_layer(h), -1)
        shift  = tf.squeeze(self.shift_layer(h), -1)
        return x * scale + shift

    def get_config(self):
        cfg = super().get_config()
        cfg['coords_data'] = self.coords_data.tolist()
        cfg['embed_dim']   = self.embed_dim
        return cfg


class SpatialRefinement(tf.keras.layers.Layer):
    """Refinamiento espacial via convoluciones 2D.
    Reconstruye la grilla 2D (target_h × target_w) a partir de los valores
    predichos en los píxeles terrestres, aplica conv2D para capturar
    coherencia espacial, y extrae el ajuste solo en los píxeles válidos.
    Se suma como residual (output = original + adjustment)."""
    def __init__(self, target_h, target_w, flat_indices, n_filters=16, **kwargs):
        super().__init__(**kwargs)
        self.target_h        = target_h
        self.target_w        = target_w
        self.flat_indices_np = np.array(flat_indices, dtype=np.int32)
        self.n_land          = len(flat_indices)
        self.n_filters       = n_filters

    def build(self, input_shape):
        self.flat_indices_tf = tf.constant(self.flat_indices_np, dtype=tf.int32)
        self.conv1 = tf.keras.layers.Conv2D(
            self.n_filters, (3, 3), padding='same', activation='relu',
            name='spatial_conv1')
        self.bn1   = tf.keras.layers.BatchNormalization(name='spatial_bn1')
        self.conv2 = tf.keras.layers.Conv2D(
            1, (3, 3), padding='same', activation='linear',
            kernel_initializer='zeros', bias_initializer='zeros',
            name='spatial_conv2')
        super().build(input_shape)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        hw         = self.target_h * self.target_w
        # Reconstruir grilla 2D dispersando los valores land a sus posiciones
        batch_idx  = tf.repeat(tf.range(batch_size), self.n_land)
        flat_idx   = tf.tile(self.flat_indices_tf, [batch_size])
        indices    = tf.stack([batch_idx, flat_idx], axis=1)
        values     = tf.reshape(x, [-1])
        grid       = tf.scatter_nd(indices, values, tf.stack([batch_size, hw]))
        grid_2d    = tf.reshape(grid, [-1, self.target_h, self.target_w, 1])
        # Aplicar conv2D para suavizar/refinar espacialmente
        refined    = self.conv2(self.bn1(self.conv1(grid_2d), training=training))
        refined_fl = tf.reshape(refined, [-1, hw])
        # Extraer solo los píxeles terrestres del resultado
        adjustment = tf.gather(refined_fl, self.flat_indices_tf, axis=1)
        return x + adjustment  # Conexión residual

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'target_h':    self.target_h,
            'target_w':    self.target_w,
            'flat_indices': self.flat_indices_np.tolist(),
            'n_filters':   self.n_filters,
        })
        return cfg

#  Modelo
def build_multitower_model(input_days, predictor_sizes, n_land,
                           spatial_embed=256, temporal_dim=128,
                           hidden_dim=256,
                           huber_delta=1.0,
                           quantile_weight=0.30,
                           quantile_tau=0.72,
                           low_threshold=50.0,
                           high_threshold=250.0,
                           tweedie_weight=0.40,
                           tweedie_power=1.30,
                           tweedie_scale_norm=True,
                           target_transform='log1p',
                           pixel_embed_dim=64,
                           use_pixel_bias=True,
                           l2_output=1e-5,
                           lr_schedule=None,
                           grad_accum_steps=1,
                           target_h=None, target_w=None,
                           flat_indices=None,
                           lat_coords=None, lon_coords=None,
                           coord_embed_dim=32,
                           spatial_refine=False,
                           smooth_weight=0.0,
                           corr_weight=0.0,
                           extreme_boost=1.0,
                           n_attn_heads=4,
                           seed=42):
    inputs  = []
    towers  = []       # Salida de cada torre (vector temporal comprimido)
    tower_stats = []   # Estadísticas temporales auxiliares (media, std, max)

    for i, (n_feat, vname) in enumerate(predictor_sizes):
        # ======== Torre i: procesa el predictor i ========
        inp = Input(shape=(input_days, n_feat), name=f'pred_{i}')
        inputs.append(inp)

        # Reducción espacial: proyecta n_feat features a spatial_embed dims
        x = TimeDistributed(Dense(spatial_embed, activation='relu'),
                            name=f'tower_{i}_{vname}_sp1')(inp)
        x = TimeDistributed(BatchNormalization(),
                            name=f'tower_{i}_{vname}_bn1')(x)
        x = TimeDistributed(Dropout(0.2),
                            name=f'tower_{i}_{vname}_do1')(x)
        # Segunda capa de reducción: spatial_embed → spatial_embed//2
        x = TimeDistributed(Dense(spatial_embed // 2, activation='relu'),
                            name=f'tower_{i}_{vname}_sp2')(x)
        x = TimeDistributed(BatchNormalization(),
                            name=f'tower_{i}_{vname}_bn2')(x)

        # BiLSTM con conexión residual: captura dependencias temporales
        x = Bidirectional(
            LSTM(temporal_dim // 2, return_sequences=True),
            name=f'tower_{i}_{vname}_bilstm1'
        )(x)
        x = BatchNormalization(name=f'tower_{i}_{vname}_lstm1_bn')(x)
        x_res = x  # Guardar para conexión residual
        x = Bidirectional(
            LSTM(temporal_dim // 2, return_sequences=True),
            name=f'tower_{i}_{vname}_bilstm2'
        )(x)
        x = Add(name=f'tower_{i}_{vname}_lstm_res')([x, x_res])

        # Multi-Head Attention manual:
        # Cada cabeza calcula pesos de atención sobre la secuencia temporal,
        # luego proyecta a head_dim dimensiones. Se concatenan las cabezas.
        head_outputs = []
        head_dim = temporal_dim // n_attn_heads  # Dims por cabeza
        for h in range(n_attn_heads):
            att_score_h = TimeDistributed(
                Dense(1, activation='tanh'),
                name=f'tower_{i}_{vname}_att_score_h{h}'
            )(x)
            att_score_h = Flatten(name=f'tower_{i}_{vname}_att_flat_h{h}')(att_score_h)
            att_w_h     = Softmax(name=f'tower_{i}_{vname}_att_softmax_h{h}')(att_score_h)
            att_w_h     = RepeatVector(temporal_dim)(att_w_h)
            att_w_h     = Permute((2, 1), name=f'tower_{i}_{vname}_att_perm_h{h}')(att_w_h)
            x_att_h     = Multiply(name=f'tower_{i}_{vname}_att_mul_h{h}')([x, att_w_h])
            x_h = Lambda(
                lambda t: tf.reduce_sum(t, axis=1),
                name=f'tower_{i}_{vname}_att_sum_h{h}'
            )(x_att_h)
            x_h = Dense(head_dim, activation='relu',
                        name=f'tower_{i}_{vname}_head_proj_h{h}')(x_h)
            head_outputs.append(x_h)

        if len(head_outputs) > 1:
            x = Concatenate(name=f'tower_{i}_{vname}_mha_concat')(head_outputs)
        else:
            x = head_outputs[0]

        # Estadísticas temporales: capturan información global de la secuencia
        # que la atención podría perder (e.g. variabilidad, extremos)
        t_mean = Lambda(lambda t: tf.reduce_mean(t, axis=1),
                        name=f'tower_{i}_{vname}_tmean')(x_res)
        t_std  = Lambda(lambda t: tf.sqrt(tf.reduce_mean(
                     tf.square(t - tf.reduce_mean(t, axis=1, keepdims=True)),
                     axis=1) + 1e-8),
                        name=f'tower_{i}_{vname}_tstd')(x_res)
        t_max  = Lambda(lambda t: tf.reduce_max(t, axis=1),
                        name=f'tower_{i}_{vname}_tmax')(x_res)
        tower_stats.extend([t_mean, t_std, t_max])  # Se usarán en la fusión

        x = BatchNormalization(name=f'tower_{i}_{vname}_out_bn')(x)
        x = Dropout(0.3, name=f'tower_{i}_{vname}_out_do')(x)
        towers.append(x)

    # ======== Fusión de torres ========
    # Gated fusion: una "puerta" sigmoide aprende qué información
    # de qué torre es más relevante para cada muestra.
    if len(towers) > 1:
        fused = Concatenate(name='fusion')(towers)
        gate  = Dense(fused.shape[-1], activation='sigmoid',
                      name='fusion_gate')(fused)
        fused = Multiply(name='fusion_gated')([fused, gate])
    else:
        fused = towers[0]

    # Añadir estadísticas temporales como features complementarias
    if tower_stats:
        stats_concat = Concatenate(name='stats_concat')(tower_stats)
        stats_proj   = Dense(hidden_dim // 4, activation='relu',
                             name='stats_proj')(stats_concat)
        fused = Concatenate(name='fusion_with_stats')([fused, stats_proj])

    # ======== Decodificador residual ========
    # 4 capas Dense con 2 conexiones residuales (skip connections)
    # para facilitar el flujo de gradientes y prevenir degradación
    x      = Dense(hidden_dim, activation='relu', name='dec_1')(fused)
    x      = BatchNormalization(name='dec_bn1')(x)
    x      = Dropout(0.2, name='dec_do1')(x)
    x_res  = x  # Punto de conexión residual 1
    x      = Dense(hidden_dim, activation='relu', name='dec_2')(x)
    x      = BatchNormalization(name='dec_bn2')(x)
    x      = Dropout(0.15, name='dec_do2')(x)
    x      = Add(name='dec_res1')([x, x_res])   # Residual 1: x + skip
    x_res2 = x  # Punto de conexión residual 2
    x      = Dense(hidden_dim, activation='relu', name='dec_3')(x)
    x      = BatchNormalization(name='dec_bn3')(x)
    x      = Dropout(0.1, name='dec_do3')(x)
    x      = Add(name='dec_res2')([x, x_res2])  # Residual 2: x + skip
    x      = Dense(hidden_dim, activation='relu', name='dec_4')(x)
    x      = BatchNormalization(name='dec_bn4')(x)

    l2_reg   = tf.keras.regularizers.l2(l2_output) if l2_output > 0 else None

    # ======== Capa de salida ========
    # softplus garantiza predicciones ≥ 0 (natural para precipitación)
    out_main = Dense(n_land, activation='softplus',
                     kernel_regularizer=l2_reg,
                     name='pixel_output')(x)

    # ======== Pixel embedding (ajuste fino por píxel) ========
    # Proyecta la representación a un espacio de baja dimensión (pixel_embed_dim),
    # calcula un ajuste tanh escalado por pixel, y lo suma a la salida principal.
    # adj_scale empieza pequeño (bias=-2.3 ≈ sigmoid(-2.3)≈0.09) para no
    # perturbar la predicción principal al inicio del entrenamiento.
    if pixel_embed_dim and pixel_embed_dim > 0:
        rep_proj = Dense(pixel_embed_dim, activation='relu',
                         name='rep_proj')(x)
        out_adj_raw = Dense(n_land, use_bias=False, name='pixel_adj')(rep_proj)
        adj_scale = Dense(n_land, activation='softplus',
                          kernel_initializer=tf.keras.initializers.Constant(0.0),
                          bias_initializer=tf.keras.initializers.Constant(-2.3),
                          name='pixel_adj_scale')(
            Lambda(lambda t: tf.ones_like(t[:, :1]), name='adj_ones')(out_adj_raw)
        )
        out_adj = Multiply(name='pixel_adj_scaled')([
            Lambda(lambda t: tf.nn.tanh(t), name='pixel_adj_tanh')(out_adj_raw),
            adj_scale
        ])
        out = Add(name='pixel_out_add')([out_main, out_adj])
    else:
        out = out_main

    if use_pixel_bias:
        out = PixelBias(n_land, name='pixel_bias_layer')(out)

    if (coord_embed_dim and coord_embed_dim > 0
            and lat_coords is not None and lon_coords is not None
            and flat_indices is not None):
        lats_2d, lons_2d = np.meshgrid(lat_coords, lon_coords, indexing='ij')
        land_lats = lats_2d.ravel()[flat_indices].astype(np.float32)
        land_lons = lons_2d.ravel()[flat_indices].astype(np.float32)
        lat_mean  = float(land_lats.mean())
        lat_std   = max(float(land_lats.std()), 1e-6)
        lon_mean  = float(land_lons.mean())
        lon_std   = max(float(land_lons.std()), 1e-6)
        coords_norm = np.stack([
            (land_lats - lat_mean) / lat_std,
            (land_lons - lon_mean) / lon_std
        ], axis=-1)
        out = CoordinateModulation(
            coords_norm, embed_dim=coord_embed_dim,
            name='coord_modulation'
        )(out)

    if (spatial_refine
            and target_h is not None and target_w is not None
            and flat_indices is not None):
        out = SpatialRefinement(
            target_h, target_w, flat_indices,
            n_filters=16, name='spatial_refinement'
        )(out)

    if grad_accum_steps > 1:
        model = GradientAccumulationModel(
            accum_steps=grad_accum_steps,
            inputs=inputs, outputs=out,
            name='multitower_pixel_v3fixed',
        )
    else:
        model = Model(inputs, out, name='multitower_pixel_v3fixed')

    neighbor_pairs = None
    if (smooth_weight > 0
            and flat_indices is not None
            and target_h is not None and target_w is not None):
        neighbor_pairs = compute_neighbor_pairs(
            flat_indices, target_h, target_w)

    loss_fn, _train_ph = make_combined_loss(
        huber_delta        = huber_delta,
        quantile_weight    = quantile_weight,
        quantile_tau       = quantile_tau,
        low_threshold      = low_threshold,
        high_threshold     = high_threshold,
        tweedie_weight     = tweedie_weight,
        tweedie_power      = tweedie_power,
        tweedie_scale_norm = tweedie_scale_norm,
        target_transform   = target_transform,
        neighbor_pairs     = neighbor_pairs,
        smooth_weight      = smooth_weight,
        corr_weight        = corr_weight,
        extreme_boost      = extreme_boost,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule if lr_schedule is not None else 1e-3,
        clipnorm=1.0
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
    return model, _train_ph


# ════════════════════════════════════════════════════════════════════
#  Predicción por chunks (LOW-RAM)
# ════════════════════════════════════════════════════════════════════

def predict_chunked(model, dataset, n_samples, batch_size, n_land):
    """Predice en chunks pequeños para no acumular todo en RAM.
    Devuelve un memmap con las predicciones."""
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), f'_pred_tmp_{os.getpid()}.npy')
    out = np.lib.format.open_memmap(
        tmp_path, mode='w+', dtype=np.float32,
        shape=(n_samples, n_land)
    )

    n_batches = (n_samples + batch_size - 1) // batch_size
    idx = 0
    for batch_x, batch_y in dataset:
        if idx >= n_samples:
            break
        pred = model(batch_x, training=False).numpy()
        end = min(idx + pred.shape[0], n_samples)
        out[idx:end] = pred[:end - idx]
        idx = end
        if idx % (batch_size * 10) == 0:
            gc.collect()
        print(f"\r   Prediciendo: {idx}/{n_samples}", end='', flush=True)

    out.flush()
    print()
    gc.collect()
    return out, tmp_path


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_global_seeds(args.seed)

    # ── Inicializar monitor de RAM ────────────────────────────────
    global _RAM
    _RAM = RAMMonitor(max_ram_gb=args.max_ram_gb)
    _RAM.report("inicio")

    # ── Actualizar constante de clip si el usuario la ajustó ─────
    # El clip físico limita predicciones al rango real de precipitación.
    # Se recalcula si precip_max_mm cambió respecto al valor por defecto.
    global _PRECIP_MAX_MM, CLIP_RANGES
    if args.precip_max_mm != _PRECIP_MAX_MM:
        _PRECIP_MAX_MM = args.precip_max_mm
        CLIP_RANGES = {
            'log1p':    (-0.5,  float(np.log1p(_PRECIP_MAX_MM))),
            'cbrt':     (0.0,   float(_PRECIP_MAX_MM ** (1/3))),
            'pow025':   (0.0,   float(_PRECIP_MAX_MM ** 0.25)),
            'pow05':    (0.0,   float(np.sqrt(_PRECIP_MAX_MM))),
            'standard': (-10.0, 10.0),
            'none':     (0.0,   _PRECIP_MAX_MM),
        }

    huber_weight = 1.0 - args.quantile_weight - args.tweedie_weight - args.corr_weight
    if huber_weight <= 0.0:
        raise ValueError(
            f"quantile_weight ({args.quantile_weight}) + "
            f"tweedie_weight ({args.tweedie_weight}) + "
            f"corr_weight ({args.corr_weight}) >= 1.0. "
            f"Peso de Huber resultante: {huber_weight:.4f}."
        )

    print("=" * 70)
    print("  MODELO")
    print("  Pérdida: Huber focal + Quantil + Tweedie + Correlación")
    print("  BiLSTM | Multi-Head Attention | Data Augmentation | SWA")
    print("  RAM monitoreada dinámicamente — nunca excede el límite")
    print("=" * 70)
    print(f"\n  Semilla global: {args.seed}")
    print(f"  RAM máxima: {args.max_ram_gb} GB (safety margin 20%)")

    predictor_configs = []
    for ps in args.predictor:
        predictor_configs.extend(parse_predictor_str(ps))
    n_preds = len(predictor_configs)
    print(f"\n  {n_preds} predictores configurados:")
    for fp, vn, dom in predictor_configs:
        print(f"    {vn} ({dom}) ← {os.path.basename(fp)}")

    if args.pca_components == -1:
        print(f"\n  PCA: DESHABILITADO — usando TODA la información espacial")
        print(f"    Cada predictor conservará todos sus píxeles válidos")
        print(f"    Batch size y gradient accumulation se ajustarán automáticamente")
    elif args.pca_components == 0:
        if not HAS_SKLEARN:
            print("\n  sklearn no disponible → PCA deshabilitado, usando features crudas")
            args.pca_components = -1
        else:
            print(f"\n  PCA: AUTO (retiene {args.pca_variance*100:.1f}% varianza)")
            print(f"    Esto captura toda la variabilidad relevante sin exceder la RAM")
    elif args.pca_components > 0:
        if not HAS_SKLEARN:
            print("\n  sklearn no disponible → PCA deshabilitado")
            args.pca_components = -1
        else:
            print(f"\n  PCA: {args.pca_components} componentes por predictor")

    if not os.path.isfile(args.precip) and os.path.isfile(args.precip + '.nc'):
        args.precip = args.precip + '.nc'

    effective_jobs = (max(1, os.cpu_count() // 2)
                      if args.n_jobs == -1 else max(args.n_jobs, 1))

    print(f"\n  Workers P4: {effective_jobs} | RAM máx: {args.max_ram_gb} GB")

    clip_lo, clip_hi = _get_clip_range(args.target_transform)
    print(f"\n  Función de pérdida v4:")
    print(f"    Huber focal(δ={args.huber_delta}, boost={args.extreme_boost}) × {huber_weight:.2f}")
    print(f"    Quantil adaptativo × {args.quantile_weight:.2f}  (tau={args.quantile_tau})")
    print(f"    Tweedie(p={args.tweedie_power})    × {args.tweedie_weight:.2f}")
    print(f"    Correlación Pearson × {args.corr_weight:.2f}")
    print(f"  Regiones quantil (en mm reales):")
    print(f"    < {args.low_threshold:.0f}   → tau=0.50")
    print(f"    {args.low_threshold:.0f}–{args.high_threshold:.0f} → tau=0.60")
    print(f"    > {args.high_threshold:.0f}  → tau={args.quantile_tau}")
    print(f"  Clip físico ({args.target_transform}):")
    print(f"    [{clip_lo:.3f}, {clip_hi:.3f}] ≡ [0, {_PRECIP_MAX_MM:.0f}] mm")
    print(f"  Mejoras: BiLSTM, {args.n_attn_heads}-head attention, "
          f"noise_std={args.noise_std}, mixup_alpha={args.mixup_alpha}"
          f"{', SWA' if args.use_swa else ''}")

    # ── [1] Preprocesamiento ──────────────────────────────────────
    # Fase intensiva en disco: lee predictores por chunks, calcula PCA,
    # y genera archivos .npy memmap con las ventanas temporales pre-procesadas.
    # Si ya existe un cache válido y no se usa --force_preprocess, lo reutiliza.
    preprocessor = DataPreprocessor(args.cache_dir, args.dtype)
    
    # Determinar seed_split (si no se especifica, usa seed)
    seed_split_actual = args.seed_split if args.seed_split is not None else args.seed
    
    # Lógica inteligente de cache de dos niveles:
    # 1. Base cache (P1-P4): stats + PCA + target — reutilizable si parámetros no cambian
    # 2. Split cache (P5): división de años (train/val) — depende de seed_split únicamente
    
    base_config_hash = preprocessor._get_base_config_hash(
        args.precip, predictor_configs, args.input_days,
        args.pca_components, args.pca_samples, args.target_transform,
        args.pca_variance, args.seed
    )
    
    # Decidir: ¿regenerar todo? ¿resplit rápido? ¿reutilizar?
    has_base = preprocessor.is_base_cached()
    has_full = preprocessor.is_cached()
    
    if not has_base or args.force_preprocess:
        # No existe base o fuerza regeneración — preprocessing completo
        print(f"\n  Generando cache base (P1-P4: stats + PCA + target)...")
        metadata = preprocessor.preprocess(
            precip_path       = args.precip,
            precip_var        = args.varname,
            predictor_configs = predictor_configs,
            input_days        = args.input_days,
            split_frac        = args.split_frac,
            pca_components    = args.pca_components,
            pca_samples       = args.pca_samples,
            target_transform  = args.target_transform,
            stats_chunk_time  = args.stats_chunk_time,
            pca_chunk_time    = args.pca_chunk_time,
            n_jobs            = effective_jobs,
            max_ram_gb        = args.max_ram_gb,
            seed              = args.seed,
            seed_split        = seed_split_actual,
            pca_variance_target = args.pca_variance,
        )
    else:
        # Existe base cache — verificar si solo cambió seed_split
        with open(os.path.join(args.cache_dir, 'base_config.json')) as f:
            base_config = json.load(f)
        
        cached_base_hash = base_config.get('base_config_hash', 'unknown')
        
        if cached_base_hash == base_config_hash:
            # ✓ Base cache válido (mismos parámetros P1-P4)
            # Verificar si cambió seed_split
            if has_full:
                with open(os.path.join(args.cache_dir, 'metadata.json')) as f:
                    metadata = json.load(f)
                cached_seed_split = metadata.get('seed_split', metadata.get('seed'))
            else:
                cached_seed_split = None
            
            if cached_seed_split != seed_split_actual:
                # Cambió seed_split — quick resplit (P5 simplificado)
                print(f"\n  ✓ Base cache válido, seed_split cambió: {cached_seed_split} → {seed_split_actual}")
                print(f"  Ejecutando RAPIDISPLIT (mucho más rápido que preprocessing completo)...")
                metadata = preprocessor._quick_resplit(
                    base_config, args.split_frac, seed_split_actual
                )
            else:
                # seed_split igual — reutilizar metadata completo
                with open(os.path.join(args.cache_dir, 'metadata.json')) as f:
                    metadata = json.load(f)
                print(f"\n  Usando cache existente: {args.cache_dir}/")
                print(f"    {metadata['n_months']} meses, "
                      f"{metadata['n_predictors']} predictores, "
                      f"seed_split={seed_split_actual}")
        else:
            # Base hash cambió — parámetros P1-P4 distintos, regenerar todo
            print(f"\n  Parámetros de base (PCA/stats) cambiaron — regenerando...")
            metadata = preprocessor.preprocess(
                precip_path       = args.precip,
                precip_var        = args.varname,
                predictor_configs = predictor_configs,
                input_days        = args.input_days,
                split_frac        = args.split_frac,
                pca_components    = args.pca_components,
                pca_samples       = args.pca_samples,
                target_transform  = args.target_transform,
                stats_chunk_time  = args.stats_chunk_time,
                pca_chunk_time    = args.pca_chunk_time,
                n_jobs            = effective_jobs,
                max_ram_gb        = args.max_ram_gb,
                seed              = args.seed,
                seed_split        = seed_split_actual,
                pca_variance_target = args.pca_variance,
            )

    gc.collect()
    _RAM.report("después preprocessing")

    n_train       = metadata['n_train']
    n_val         = metadata['n_val']
    n_land        = metadata['n_land']

    # Compatibilidad con caches antiguos que no tienen train_indices/val_indices
    if 'train_indices' in metadata and 'val_indices' in metadata:
        train_indices = metadata['train_indices']
        val_indices   = metadata['val_indices']
    else:
        # Reconstruir desde años si están disponibles, sino desde n_train/n_val
        if 'train_years' in metadata and 'val_years' in metadata and 'month_timestamps' in metadata:
            import pandas as _pd
            val_year_set = set(metadata['val_years'])
            timestamps = [_pd.Timestamp(t) for t in metadata['month_timestamps']]
            train_indices = [i for i, t in enumerate(timestamps) if t.year not in val_year_set]
            val_indices   = [i for i, t in enumerate(timestamps) if t.year in val_year_set]
        else:
            n_total = n_train + n_val
            train_indices = list(range(n_train))
            val_indices   = list(range(n_train, n_total))
        print(" Cache antiguo sin train_indices/val_indices — reconstruidos. "
              "Considere --force_preprocess para regenerar.")

    # Compatibilidad con caches antiguos que no tienen train_end
    if 'train_end' not in metadata:
        if train_indices and 'month_timestamps' in metadata:
            train_end_idx = max(train_indices)
            metadata['train_end'] = metadata['month_timestamps'][train_end_idx]
        else:
            metadata['train_end'] = 'unknown'

    # ── Auto batch size — calcula batch seguro según features y RAM ──
    # Estima la RAM por muestra durante entrenamiento considerando:
    # - Input data: input_days × total_features × 4 bytes
    # - TF overhead: ~4× (input + activaciones + gradientes_out + gradientes_in)
    # Si el batch solicitado excede la RAM, se reduce automáticamente
    # y se compensa con gradient accumulation.
    feat_sizes_list = metadata['predictor_feat_sizes']
    total_feat_size = sum(feat_sizes_list)

    # Estimar RAM por muestra durante entrenamiento:
    #   Input: input_days × total_feat_size × 4 bytes
    #   TF necesita ~4× para: input + activaciones + gradients_out + gradients_in
    bytes_per_sample_train = args.input_days * total_feat_size * 4 * 4

    # Costos fijos: pesos primera capa + gradientes + rest del modelo + TF overhead
    first_layer_weight_bytes = sum(
        fs * args.spatial_embed * 4 * 2  # peso + gradiente
        for fs in feat_sizes_list
    )
    # Modelo restante (LSTMs, attention, decoder) + overhead TF
    fixed_overhead_bytes = 3 * 1024**3  # ~3 GB estimado
    total_fixed = first_layer_weight_bytes + fixed_overhead_bytes

    # RAM disponible para batches (60% de lo reportado por el monitor)
    available_for_batch = max(
        _RAM.available_bytes() * 0.60 - total_fixed,
        256 * 1024**2  # mínimo 256 MB
    )
    safe_batch = max(1, int(available_for_batch / max(bytes_per_sample_train, 1)))

    original_batch = args.batch_size
    if safe_batch < args.batch_size:
        args.batch_size = safe_batch
        # Auto gradient accumulation para mantener batch efectivo similar
        auto_accum = max(1, original_batch // args.batch_size)
        args.grad_accum_steps = max(args.grad_accum_steps, auto_accum)

    if args.batch_size != original_batch:
        print(f"\n  [AUTO-BATCH] Features totales por muestra: {total_feat_size:,}")
        print(f"    RAM primera capa (pesos+grads): "
              f"{first_layer_weight_bytes / (1024**2):.0f} MB")
        print(f"    RAM por muestra (train): "
              f"{bytes_per_sample_train / (1024**2):.1f} MB")
        print(f"    Batch size ajustado: {original_batch} → {args.batch_size}")
        print(f"    Gradient accumulation: {args.grad_accum_steps} pasos")
        print(f"    Batch efectivo: "
              f"{args.batch_size * args.grad_accum_steps}")
    else:
        print(f"\n  [BATCH] Features totales: {total_feat_size:,} | "
              f"batch_size={args.batch_size} cabe en RAM")

    # ── Estimar espacio en disco del cache ────────────────────────
    cache_disk_bytes = sum(
        metadata['n_months'] * args.input_days * fs * 4
        for fs in feat_sizes_list
    )
    cache_disk_gb = cache_disk_bytes / (1024**3)
    if cache_disk_gb > 5.0:
        print(f"\n Cache en disco estimado: {cache_disk_gb:.1f} GB")
        print(f"    Asegúrate de tener espacio suficiente en {args.cache_dir}/")

    # ── [2] Modelo ────────────────────────────────────────────────
    predictor_sizes = list(zip(
        metadata['predictor_feat_sizes'], metadata['predictor_names']
    ))
    print(f"\n[2] Construyendo modelo ({len(predictor_sizes)} torres)...")
    for i, (fs, nm) in enumerate(predictor_sizes):
        orig = metadata['predictor_n_valid'][i]
        red  = f" (sin PCA, {orig} píxeles)" if fs == orig else f" (reducido de {orig})"
        print(f"     Torre {i} ({nm}): ({args.input_days}, {fs}){red}")
    print(f"     Output: ({n_land},)  [activación: softplus]")

    steps_per_epoch_est = max(1, (n_train + args.batch_size - 1) // args.batch_size)
    total_steps  = steps_per_epoch_est * args.epochs
    warmup_steps = max(1, total_steps // 10)
    t0_steps     = steps_per_epoch_est * 50
    lr_schedule  = WarmupCosineDecayWarmRestarts(
        initial_lr=1e-3, warmup_steps=warmup_steps, total_steps=total_steps,
        t0_steps=t0_steps, t_mult=2.0, min_lr=1e-6
    )
    print(f"     LR schedule: warmup {warmup_steps} steps → cosine warm restarts "
          f"(T₀={t0_steps} steps, T_mult=2)")
    if args.grad_accum_steps > 1:
        print(f"     Gradient accumulation: {args.grad_accum_steps} pasos "
              f"(batch efectivo = {args.batch_size * args.grad_accum_steps})")

    flat_idx_arr = np.array(metadata['flat_indices'], dtype=np.int32)
    lat_arr      = np.array(metadata['lat_coords'])
    lon_arr      = np.array(metadata['lon_coords'])

    model, _train_ph = build_multitower_model(
        input_days         = metadata['input_days'],
        predictor_sizes    = predictor_sizes,
        n_land             = n_land,
        spatial_embed      = args.spatial_embed,
        temporal_dim       = args.temporal_dim,
        hidden_dim         = args.hidden_dim,
        huber_delta        = args.huber_delta,
        quantile_weight    = args.quantile_weight,
        quantile_tau       = args.quantile_tau,
        low_threshold      = args.low_threshold,
        high_threshold     = args.high_threshold,
        tweedie_weight     = args.tweedie_weight,
        tweedie_power      = args.tweedie_power,
        tweedie_scale_norm = args.tweedie_scale_norm,
        target_transform   = args.target_transform,
        pixel_embed_dim    = args.pixel_embed_dim,
        l2_output          = args.l2_output,
        lr_schedule        = lr_schedule,
        grad_accum_steps   = args.grad_accum_steps,
        target_h           = metadata['target_h'],
        target_w           = metadata['target_w'],
        flat_indices       = flat_idx_arr,
        lat_coords         = lat_arr,
        lon_coords         = lon_arr,
        coord_embed_dim    = args.coord_embed_dim,
        spatial_refine     = args.spatial_refine,
        smooth_weight      = args.smooth_weight,
        corr_weight        = args.corr_weight,
        extreme_boost      = args.extreme_boost,
        n_attn_heads       = args.n_attn_heads,
        seed               = args.seed,
    )
    model.summary(print_fn=lambda s: print(f"  {s}"))
    total_params = model.count_params()
    print(f"\n  Parámetros: {total_params:,} ({total_params*4/(1024**2):.1f} MB)")
    gc.collect()
    _RAM.report("después modelo")

    # ── [3] Datasets ──────────────────────────────────────────────
    print(f"\nCreando datasets tf.data (prefetch={args.prefetch}"
          f", noise_std={args.noise_std}, mixup_alpha={args.mixup_alpha})...")
    train_ds, _, _ = create_tf_dataset(
        args.cache_dir, metadata, train_indices,
        args.batch_size, shuffle=True, prefetch_n=args.prefetch,
        seed=args.seed,
        noise_std=args.noise_std,
        mixup_alpha=args.mixup_alpha,
    )
    val_ds, _, _ = create_tf_dataset(
        args.cache_dir, metadata,
        val_indices,
        args.batch_size, shuffle=False, prefetch_n=args.prefetch,
        seed=args.seed + 1,
    )

    # ── [4] Entrenamiento ─────────────────────────────────────────
    steps_per_epoch  = max(1, (n_train + args.batch_size - 1) // args.batch_size)
    validation_steps = (max(1, (n_val + args.batch_size - 1) // args.batch_size)
                        if n_val > 0 else None)

    print(f"\nEntrenando ({args.epochs} epochs, "
          f"batch={args.batch_size}, steps/epoch={steps_per_epoch})...")
    _RAM.report("antes entrenamiento")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.save, monitor='val_loss',
            save_best_only=True, verbose=0
        ),
    ]
    callbacks.append(TrainingPhaseCallback(_train_ph))
    callbacks.append(RAMWatchdogCallback(check_every=50))
    if args.use_swa:
        callbacks.append(SWACallback(swa_start_frac=0.75))
        print("     SWA activado: promedio de pesos en último 25% de epochs")

    fit_kwargs = dict(
        epochs=args.epochs, callbacks=callbacks, verbose=1,
        steps_per_epoch=steps_per_epoch
    )
    if n_val > 0:
        fit_kwargs['validation_data']  = val_ds
        fit_kwargs['validation_steps'] = validation_steps

    history = model.fit(train_ds, **fit_kwargs)
    gc.collect()
    _RAM.report("después entrenamiento")

    # ── [5] Guardar ───────────────────────────────────────────────
    save_path = args.save
    if save_path.endswith('.h5'):
        save_path = save_path[:-3] + '.keras'

    print(f"\nGuardando modelo en {save_path}...")
    try:
        model.save(save_path)
    except Exception:
        fallback = os.path.splitext(save_path)[0] + '_savedmodel'
        model.save(fallback)
        print(f"   (SavedModel: {fallback})")

    norm_file = os.path.splitext(save_path)[0] + '_norm_stats.json'
    clip_lo_save, clip_hi_save = _get_clip_range(args.target_transform)
    with open(norm_file, 'w') as f:
        json.dump({
            'predictor_stats':      metadata['predictor_stats'],
            'predictor_feat_sizes': metadata['predictor_feat_sizes'],
            'predictor_names':      metadata['predictor_names'],
            'target_transform':     metadata['target_transform'],
            'target_mean':          metadata['target_mean'],
            'target_std':           metadata['target_std'],
            'train_end':            metadata['train_end'],
            'n_train_months':       n_train,
            'n_val_months':         n_val,
            'pca_components':       metadata['pca_components'],
            'pca_variance_target':  metadata.get('pca_variance_target', 0.999),
            'cache_dir':            args.cache_dir,
            'huber_delta':          args.huber_delta,
            'huber_weight':         huber_weight,
            'quantile_weight':      args.quantile_weight,
            'quantile_tau':         args.quantile_tau,
            'low_threshold':        args.low_threshold,
            'high_threshold':       args.high_threshold,
            'tweedie_weight':       args.tweedie_weight,
            'tweedie_power':        args.tweedie_power,
            'tweedie_scale_norm':   args.tweedie_scale_norm,
            'corr_weight':          args.corr_weight,
            'extreme_boost':        args.extreme_boost,
            'precip_max_mm':        _PRECIP_MAX_MM,
            'clip_lo_transformed':  clip_lo_save,
            'clip_hi_transformed':  clip_hi_save,
            'output_activation':    'softplus',
            'coord_embed_dim':      args.coord_embed_dim,
            'spatial_refine':       args.spatial_refine,
            'smooth_weight':        args.smooth_weight,
            'n_attn_heads':         args.n_attn_heads,
            'use_swa':              args.use_swa,
            'noise_std':            args.noise_std,
            'mixup_alpha':          args.mixup_alpha,
            'seed':                 args.seed,
            'low_ram_edition':      True,
        }, f, indent=2, default=str)
    print(f"   Stats: {norm_file}")

    # ── [6] Predicciones — LOW-RAM: por chunks ───────────────────
    print("\nGenerando predicciones sobre validación...")
    _RAM.report("antes predicciones")

    target_transform = metadata['target_transform']
    target_mean      = metadata['target_mean']
    target_std       = metadata['target_std']

    # Funciones inversas: deshacen la transformación del target
    # para recuperar precipitación en mm reales.
    if target_transform == 'log1p':
        inv_fn = np.expm1            # expm1(x) = e^x - 1
    elif target_transform == 'cbrt':
        inv_fn = lambda x: np.power(np.maximum(x, 0.0), 3.0)   # x³
    elif target_transform == 'pow025':
        inv_fn = lambda x: np.power(np.maximum(x, 0.0), 4.0)   # x⁴
    elif target_transform == 'pow05':
        inv_fn = lambda x: np.power(np.maximum(x, 0.0), 2.0)   # x²
    elif target_transform == 'standard' and target_mean is not None:
        inv_fn = lambda x: x * target_std + target_mean         # desnormalizar
    else:
        inv_fn = lambda x: x  # identidad (sin transformación)

    pred_ds, _, _ = create_tf_dataset(
        args.cache_dir, metadata,
        val_indices,
        args.batch_size, shuffle=False, prefetch_n=args.prefetch,
        seed=args.seed + 1,
    )

    # Predicción chunked en lugar de model.predict() que acumularía
    # todas las predicciones en un solo array gigante
    Y_pred_mm, pred_tmp_path = predict_chunked(
        model, pred_ds, n_val, args.batch_size, n_land
    )
    gc.collect()

    # Cargar target (observaciones) desde memmap en modo lectura
    target_mmap = np.load(
        os.path.join(args.cache_dir, 'target.npy'), mmap_mode='r'
    )
    Y_true_mmap = target_mmap[val_indices]  # Slice de validación

    n_val_actual = min(Y_pred_mm.shape[0], Y_true_mmap.shape[0])

    if n_val_actual == 0:
        print("   ADVERTENCIA: no hay muestras de validación.")
    else:
        clip_lo, clip_hi = _get_clip_range(target_transform)
        print(f"\n   FIX A1: clip en espacio {target_transform}: "
              f"[{clip_lo:.3f}, {clip_hi:.3f}] ≡ [0, {_PRECIP_MAX_MM:.0f} mm]")

        # Procesar por chunks para no acumular arrays gigantes
        chunk_months = max(1, _RAM.safe_chunk_rows(n_land, 4, 0.2))
        chunk_months = min(chunk_months, n_val_actual)

        # Para diagnósticos, necesitamos estadísticas acumuladas
        # Se calculan incrementalmente por chunks para no guardar todo en RAM
        acc_pred_min = np.inf     # Mínimo predicho
        acc_pred_max = -np.inf    # Máximo predicho
        acc_pred_sum = 0.0        # Suma para media
        acc_real_min = np.inf     # Mínimo observado
        acc_real_max = -np.inf    # Máximo observado
        acc_real_sum = 0.0        # Suma para media
        acc_count = 0             # Total de píxeles procesados
        acc_err_sum = 0.0         # Suma de errores (para bias)
        acc_err_sq_sum = 0.0      # Suma de errores² (para RMSE)
        acc_abs_err_sum = 0.0     # Suma de |error| (para MAE)

        # Acumuladores para correlación de Pearson incremental:
        # r = (n*Σxy - Σx*Σy) / sqrt((n*Σx² - (Σx)²) * (n*Σy² - (Σy)²))
        acc_xy = 0.0   # Σ(pred * obs)
        acc_x = 0.0    # Σ(pred)
        acc_y = 0.0    # Σ(obs)
        acc_x2 = 0.0   # Σ(pred²)
        acc_y2 = 0.0   # Σ(obs²)
        n_finite = 0    # Conteo de pares finitos

        # Muestreo para percentiles: no guardamos todos los valores,
        # solo submuestras aleatorias de cada chunk para estimar cuantiles
        all_pred_vals = []
        all_real_vals = []

        # NetCDF: crear memmaps temporales para la grilla 2D completa
        # Se escriben píxel a píxel y luego se vuelcan al archivo final
        target_h   = metadata['target_h']
        target_w   = metadata['target_w']
        flat_idx   = np.array(metadata['flat_indices'], dtype=np.int32)
        lat_coords = np.array(metadata['lat_coords'])
        lon_coords = np.array(metadata['lon_coords'])
        lat_name   = metadata['lat_name']
        lon_name   = metadata['lon_name']
        month_ts   = metadata['month_timestamps']
        val_ts = np.array([np.datetime64(month_ts[val_indices[t]]) for t in range(n_val_actual)])

        # Crear arrays NetCDF por chunks también
        import tempfile
        nc_pred_path = os.path.join(tempfile.gettempdir(), f'_ncpred_{os.getpid()}.npy')
        nc_real_path = os.path.join(tempfile.gettempdir(), f'_ncreal_{os.getpid()}.npy')
        nc_pred_mm = np.lib.format.open_memmap(
            nc_pred_path, mode='w+', dtype=np.float32,
            shape=(n_val_actual, target_h, target_w)
        )
        nc_real_mm = np.lib.format.open_memmap(
            nc_real_path, mode='w+', dtype=np.float32,
            shape=(n_val_actual, target_h, target_w)
        )
        nc_pred_mm[:] = np.nan
        nc_real_mm[:] = np.nan

        print(f"   Procesando {n_val_actual} meses en chunks de {chunk_months}...")
        n_outlier_px = 0
        n_outlier_mon = 0
        outlier_thresh = _PRECIP_MAX_MM * 0.75

        for ci in range(0, n_val_actual, chunk_months):
            ce = min(ci + chunk_months, n_val_actual)
            # Leer chunk de predicciones (desde memmap)
            yp_chunk = np.array(Y_pred_mm[ci:ce])
            yt_chunk = np.array(Y_true_mmap[ci:ce])

            # Clip
            yp_chunk = np.clip(yp_chunk, clip_lo, clip_hi)

            # Invertir
            yp_inv = np.maximum(inv_fn(yp_chunk), 0.0).astype(np.float32)
            yt_inv = inv_fn(yt_chunk).astype(np.float32)

            # Estadísticas acumuladas
            acc_pred_min = min(acc_pred_min, float(yp_inv.min()))
            acc_pred_max = max(acc_pred_max, float(yp_inv.max()))
            acc_pred_sum += float(yp_inv.sum())
            acc_real_min = min(acc_real_min, float(yt_inv.min()))
            acc_real_max = max(acc_real_max, float(yt_inv.max()))
            acc_real_sum += float(yt_inv.sum())
            acc_count += yp_inv.size

            # Outliers
            out_mask = yp_inv > outlier_thresh
            n_outlier_px += int(out_mask.sum())
            n_outlier_mon += int(out_mask.any(axis=1).sum())

            # Error stats
            diff = yp_inv.flatten() - yt_inv.flatten()
            mask_fin = np.isfinite(diff) & np.isfinite(yp_inv.flatten()) & np.isfinite(yt_inv.flatten())
            diff_fin = diff[mask_fin]
            yp_fin = yp_inv.flatten()[mask_fin]
            yt_fin = yt_inv.flatten()[mask_fin]

            acc_err_sum += float(diff_fin.sum())
            acc_err_sq_sum += float((diff_fin ** 2).sum())
            acc_abs_err_sum += float(np.abs(diff_fin).sum())

            # Correlación parcial
            acc_xy += float((yp_fin * yt_fin).sum())
            acc_x += float(yp_fin.sum())
            acc_y += float(yt_fin.sum())
            acc_x2 += float((yp_fin ** 2).sum())
            acc_y2 += float((yt_fin ** 2).sum())
            n_finite += len(yp_fin)

            # Percentiles: muestrear para no guardar todo
            rng_p = np.random.default_rng(args.seed + ci)
            n_sample_p = min(10000, len(yp_fin))
            if n_sample_p > 0:
                idx_s = rng_p.choice(len(yp_fin), size=n_sample_p, replace=False)
                all_pred_vals.append(yp_fin[idx_s])
                all_real_vals.append(yt_fin[idx_s])

            # NetCDF grid
            for t_local in range(ce - ci):
                t_global = ci + t_local
                nc_pred_mm[t_global].ravel()[flat_idx] = yp_inv[t_local]
                nc_real_mm[t_global].ravel()[flat_idx] = yt_inv[t_local]

            del yp_chunk, yt_chunk, yp_inv, yt_inv, diff, mask_fin, diff_fin
            gc.collect()

        # ── Diagnósticos ──────────────────────────────────────────
        print("\nDiagnósticos")
        pred_mean = acc_pred_sum / max(acc_count, 1)
        real_mean = acc_real_sum / max(acc_count, 1)
        print(f"Predicciones — min: {acc_pred_min:.2f}, "
              f"max: {acc_pred_max:.2f}, mean: {pred_mean:.2f}")
        print(f"Reales       — min: {acc_real_min:.2f}, "
              f"max: {acc_real_max:.2f}, mean: {real_mean:.2f}")

        # FIX B2: outliers
        print("\n--- FIX B2: Diagnóstico de outliers ---")
        pct_px  = 100.0 * n_outlier_px / max(acc_count, 1)
        pct_mon = 100.0 * n_outlier_mon / max(n_val_actual, 1)
        print(f"  Umbral outlier: {outlier_thresh:.0f} mm")
        print(f"  Píxeles con pred > {outlier_thresh:.0f} mm: "
              f"{n_outlier_px} ({pct_px:.2f}%)")
        print(f"  Meses con ≥1 píxel outlier: "
              f"{n_outlier_mon}/{n_val_actual} ({pct_mon:.1f}%)")
        if n_outlier_px > 0:
            if pct_mon > 50:
                print("  → Outliers distribuidos en muchos meses.")
                print("    Recomendación: aumentar --smooth_weight.")
            elif n_outlier_mon < 5:
                print("  → Outliers concentrados en pocos meses.")
                print("    Recomendación: revisar --input_days.")
            else:
                print("  → Patrón mixto.")
        else:
            print("  ✓ Sin outliers detectados.")

        # Percentiles
        if all_pred_vals and all_real_vals:
            yp_d = np.concatenate(all_pred_vals)
            yr_d = np.concatenate(all_real_vals)
            print("\nPercentiles (Pred vs Real):")
            for pc in [10, 25, 50, 75, 90, 95, 99]:
                pp = np.percentile(yp_d, pc)
                pr = np.percentile(yr_d, pc)
                ratio = (pp / pr) if abs(pr) > 1e-8 else np.nan
                flag  = ('↑ sobreestima' if ratio > 1.15
                         else '↓ subestima' if ratio < 0.85 else '✓')
                print(f"  P{pc:2d}%  Pred: {pp:8.2f}  Real: {pr:8.2f}"
                      f"  Ratio: {ratio:.2f}  {flag}")
            del yp_d, yr_d, all_pred_vals, all_real_vals

        # Umbrales
        # Para esto necesitaríamos datos completos; usamos las estadísticas acumuladas
        # Las métricas principales se calculan desde las sumas acumuladas
        print()

        # ── [7] NetCDF ────────────────────────────────────────────
        print("Creando NetCDF con predicciones...")
        ds_out = xr.Dataset({
            'predicted': xr.DataArray(
                np.array(nc_pred_mm),  # lee de memmap
                dims=['time', lat_name, lon_name],
                coords={'time': val_ts, lat_name: lat_coords, lon_name: lon_coords},
                attrs={'long_name': 'Predicted monthly precipitation', 'units': 'mm'},
            ),
            'observed': xr.DataArray(
                np.array(nc_real_mm),
                dims=['time', lat_name, lon_name],
                coords={'time': val_ts, lat_name: lat_coords, lon_name: lon_coords},
                attrs={'long_name': 'Observed monthly precipitation', 'units': 'mm'},
            ),
        })
        ds_out.attrs.update({
            'title':            'Multi-tower model predictions (v4 LOW-RAM)',
            'model':            'Multi-torre Dense+LSTM (v4 LOW-RAM)',
            'huber_weight':     huber_weight,
            'quantile_weight':  args.quantile_weight,
            'quantile_tau':     args.quantile_tau,
            'tweedie_weight':   args.tweedie_weight,
            'tweedie_power':    args.tweedie_power,
            'output_activation':'softplus',
            'precip_max_mm':    _PRECIP_MAX_MM,
            'clip_transformed': f'[{clip_lo:.3f}, {clip_hi:.3f}]',
            'seed':             args.seed,
            'creation_date':    pd.Timestamp.now().isoformat(),
        })

        base_nc = os.path.splitext(args.save)[0] + '_predictions.nc'
        alt_nc  = (os.path.splitext(args.save)[0]
                   + f"_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.nc")
        output_nc = None
        for cand in [base_nc, alt_nc,
                     os.path.join(args.cache_dir, os.path.basename(alt_nc))]:
            try:
                if os.path.exists(cand):
                    os.remove(cand)
                try:
                    ds_out.to_netcdf(cand, mode='w', format='NETCDF4', engine='netcdf4')
                except Exception:
                    ds_out.to_netcdf(cand, mode='w', engine='scipy')
                output_nc = cand
                print(f"   ✓ {output_nc}")
                break
            except Exception as exc:
                print(f"   ✗ {cand}: {exc}")

        if output_nc is None:
            raise RuntimeError("No se pudo guardar NetCDF.")

        del ds_out, nc_pred_mm, nc_real_mm
        # Limpiar temporales
        for tmp in [nc_pred_path, nc_real_path, pred_tmp_path]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        gc.collect()

        # Métricas 
        print("\nMétricas de validación...")
        if n_finite >= 2:
            # Pearson desde sumas acumuladas:
            # r = (n*Σxy - Σx*Σy) / sqrt((n*Σx² - (Σx)²) * (n*Σy² - (Σy)²))
            N = float(n_finite)
            num = N * acc_xy - acc_x * acc_y
            den = np.sqrt(max((N * acc_x2 - acc_x**2) * (N * acc_y2 - acc_y**2), 1e-16))
            corr = num / den if den > 0 else 0.0
            bias = acc_err_sum / N
            rmse = np.sqrt(acc_err_sq_sum / N)
            mae  = acc_abs_err_sum / N
            print(f"   Pearson: {corr:.4f}")
            print(f"   Bias:    {bias:.4f} mm")
            print(f"   RMSE:    {rmse:.4f} mm")
            print(f"   MAE:     {mae:.4f} mm")

    _RAM.report("final")

    # Resumen final 
    print('\n' + '=' * 70)
    print(' Entrenamiento completado')
    print('  Pérdida: Huber focal + Quantil + Tweedie + Correlación')
    print('  BiLSTM | Multi-Head Attention | Data Augmentation | SWA')
    print('=' * 70)
    print(f"  Modelo:   {args.save}")
    try:
        print(f"  NetCDF:   {output_nc}")
    except NameError:
        pass
    print(f"  Stats:    {norm_file}")
    print(f"  Cache:    {args.cache_dir}/")
    print(f"  Semilla:  {args.seed}")
    print(f"\n  Pérdida v4:")
    print(f"    Huber focal(δ={args.huber_delta}, boost={args.extreme_boost}) × {huber_weight:.2f}")
    print(f"    Quantil adapt. × {args.quantile_weight:.2f}  "
          f"(tau={args.quantile_tau} para >{args.high_threshold:.0f} mm)")
    print(f"    Tweedie(p={args.tweedie_power})  × {args.tweedie_weight:.2f}")
    print(f"    Correlación      × {args.corr_weight:.2f}")
    clip_lo_f, clip_hi_f = _get_clip_range(args.target_transform)
    print(f"\n  Clip [{clip_lo_f:.3f}, {clip_hi_f:.3f}] "
          f"(máx. {_PRECIP_MAX_MM:.0f} mm físico)")
    print(f"  Mejoras v4: BiLSTM, {args.n_attn_heads}-head attention, "
          f"noise_std={args.noise_std}, mixup={args.mixup_alpha}"
          f"{', SWA' if args.use_swa else ''}")
    feat_total = sum(metadata['predictor_feat_sizes'])
    orig_total = sum(metadata['predictor_n_valid'])
    if feat_total != orig_total:
        print(f"\n  PCA:      {orig_total:,} → {feat_total:,} feats "
              f"({100*feat_total/orig_total:.1f}%)")
    else:
        print(f"\n  SIN PCA: {feat_total:,} features totales (100% información)")
    if args.batch_size != original_batch:
        print(f"  Auto-batch: {args.batch_size} × {args.grad_accum_steps} accum "
              f"= {args.batch_size * args.grad_accum_steps} efectivo")
    print(f"\n  LOW-RAM: Procesamiento siempre desde disco, "
          f"RAM máx={args.max_ram_gb} GB")
    print('=' * 70)


if __name__ == '__main__':
    main()
