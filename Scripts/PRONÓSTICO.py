#!/usr/bin/env python
"""
predict.py — Predicción de precipitación mensual con modelo multi-torre entrenado
==================================================================================

Usa un modelo entrenado con num1_opt.py para generar predicciones de
precipitación mensual acumulada (mm) a partir de archivos de predictores nuevos.

Requisitos:
  - Modelo entrenado (.keras o directorio SavedModel)
  - Directorio de cache del entrenamiento (metadata.json + PCA .joblib)
  - Archivos NetCDF de predictores con al menos 31 días antes del primer mes objetivo

Uso:
  python predict.py \\
    --model modelo_pixel.keras \\
    --cache_dir ./preprocessing_cache \\
    --predictor sst_new.nc:sst \\
    --predictor sp_new.nc:sp \\
    --predictor rmm1_new.nc:RMM1 \\
    --predictor rmm2_new.nc:RMM2 \\
    --target_months 2025-01:2025-12 \\
    --output predicciones_2025.nc

Los predictores DEBEN proporcionarse en el MISMO ORDEN que durante
el entrenamiento.
"""

import argparse
import os
import json
import gc
import pickle
from datetime import timedelta

import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# ───── GPU memory growth ─────────────────────────────────────────
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


# ═════════════════════════════════════════════════════════════════
#  Constantes de clip físico
# ═════════════════════════════════════════════════════════════════

_PRECIP_MAX_MM = 2000.0

CLIP_RANGES = {
    'log1p':    (-0.5,  float(np.log1p(_PRECIP_MAX_MM))),
    'cbrt':     (0.0,   float(_PRECIP_MAX_MM ** (1/3))),
    'pow025':   (0.0,   float(_PRECIP_MAX_MM ** 0.25)),
    'pow05':    (0.0,   float(np.sqrt(_PRECIP_MAX_MM))),
    'standard': (-10.0, 10.0),
    'none':     (0.0,   _PRECIP_MAX_MM),
}


# ═════════════════════════════════════════════════════════════════
#  Objetos custom necesarios para cargar el modelo
# ═════════════════════════════════════════════════════════════════

class PixelBias(tf.keras.layers.Layer):
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
        h     = self.coord_dense2(self.coord_dense1(self.coords))
        scale = tf.squeeze(self.scale_layer(h), -1)
        shift = tf.squeeze(self.shift_layer(h), -1)
        return x * scale + shift

    def get_config(self):
        cfg = super().get_config()
        cfg['coords_data'] = self.coords_data.tolist()
        cfg['embed_dim']   = self.embed_dim
        return cfg


class SpatialRefinement(tf.keras.layers.Layer):
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
        batch_idx  = tf.repeat(tf.range(batch_size), self.n_land)
        flat_idx   = tf.tile(self.flat_indices_tf, [batch_size])
        indices    = tf.stack([batch_idx, flat_idx], axis=1)
        values     = tf.reshape(x, [-1])
        grid       = tf.scatter_nd(indices, values, tf.stack([batch_size, hw]))
        grid_2d    = tf.reshape(grid, [-1, self.target_h, self.target_w, 1])
        refined    = self.conv2(self.bn1(self.conv1(grid_2d), training=training))
        refined_fl = tf.reshape(refined, [-1, hw])
        adjustment = tf.gather(refined_fl, self.flat_indices_tf, axis=1)
        return x + adjustment

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'target_h':     self.target_h,
            'target_w':     self.target_w,
            'flat_indices': self.flat_indices_np.tolist(),
            'n_filters':    self.n_filters,
        })
        return cfg


class GradientAccumulationModel(Model):
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

    def get_config(self):
        cfg = super().get_config()
        cfg['accum_steps'] = self.accum_steps
        return cfg


class WarmupCosineDecayWarmRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-3, warmup_steps=100, total_steps=1000,
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
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        t         = step - self.warmup_steps
        if self.t_mult == 1.0:
            cycle_len  = self.t0_steps
            t_in_cycle = tf.math.floormod(t, cycle_len)
        else:
            n = tf.math.floor(
                tf.math.log(t / self.t0_steps * (self.t_mult - 1.0) + 1.0)
                / tf.math.log(self.t_mult)
            )
            n          = tf.maximum(n, 0.0)
            cumul      = self.t0_steps * (tf.pow(self.t_mult, n) - 1.0) / (self.t_mult - 1.0)
            cycle_len  = self.t0_steps * tf.pow(self.t_mult, n)
            t_in_cycle = tf.maximum(t - cumul, 0.0)
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


# ═════════════════════════════════════════════════════════════════
#  Funciones de utilidad (replicadas de num1_opt.py)
# ═════════════════════════════════════════════════════════════════

def _normalize_time_dim(ds):
    for tname in ['T', 'valid_time', 'time_counter']:
        if tname in ds.dims or tname in ds.coords:
            try:
                ds = ds.rename({tname: 'time'})
            except Exception:
                pass
            break
    return ds


def _normalize_spatial_dims(da):
    rename = {}
    for name in ['latitude', 'y', 'Y']:
        if name in da.dims or name in da.coords:
            rename[name] = 'lat'
            break
    for name in ['longitude', 'x', 'X']:
        if name in da.dims or name in da.coords:
            rename[name] = 'lon'
            break
    if rename:
        try:
            da = da.rename(rename)
        except Exception:
            pass
    if 'lat' not in da.dims and 'lon' not in da.dims:
        da = da.expand_dims({'lat': [0.0], 'lon': [0.0]})
    elif 'lat' not in da.dims:
        da = da.expand_dims({'lat': [0.0]})
    elif 'lon' not in da.dims:
        da = da.expand_dims({'lon': [0.0]})
    try:
        if all(d in da.dims for d in ('time', 'lat', 'lon')):
            da = da.transpose('time', 'lat', 'lon')
    except Exception:
        pass
    return da


def create_validity_mask(da, n_sample=50):
    nt = min(n_sample, len(da['time'])) if 'time' in da.dims else 1
    if 'time' in da.dims:
        sample = da.isel(time=slice(0, nt)).values
    else:
        sample = da.values[np.newaxis, ...]
    return np.any(np.isfinite(sample), axis=0)


# ═════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Predicción de precipitación mensual con modelo multi-torre",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ejemplo:
  python predict.py \\
    --model modelo_pixel.keras \\
    --cache_dir ./preprocessing_cache \\
    --predictor sst_new.nc:sst \\
    --predictor sp_new.nc:sp \\
    --predictor rmm1_new.nc:RMM1 \\
    --predictor rmm2_new.nc:RMM2 \\
    --target_months 2025-01:2025-12 \\
    --output predicciones_2025.nc

Los predictores DEBEN ir en el MISMO ORDEN que durante el entrenamiento.
Los archivos deben cubrir al menos 31 días antes del primer mes objetivo.
        """,
    )
    p.add_argument('--model', required=True,
                   help='Ruta al modelo entrenado (.keras o directorio SavedModel)')
    p.add_argument('--cache_dir', default='./preprocessing_cache',
                   help='Directorio con metadata.json y PCA del entrenamiento')
    p.add_argument('--predictor', action='append', required=True,
                   help='"archivo.nc:variable" — en el mismo orden que el entrenamiento')
    p.add_argument('--target_months', required=True,
                   help='Mes(es) objetivo: "2025-03" o rango "2025-01:2025-12"')
    p.add_argument('--output', default='predictions.nc',
                   help='Ruta del NetCDF de salida (default: predictions.nc)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--precip_max_mm', type=float, default=None,
                   help='Override del máximo físico en mm. '
                        'Si no se da, usa el valor del entrenamiento.')
    return p.parse_args()


def parse_predictor_arg(pred_str):
    """Parsea 'archivo.nc:var1[,var2,...]' → lista de (filepath, varname)."""
    parts = pred_str.split(':')
    if len(parts) < 2:
        raise ValueError(
            f"Formato inválido: '{pred_str}' → debe ser archivo.nc:variable"
        )
    filepath = parts[0]
    varnames = parts[1].split(',')
    if not os.path.isfile(filepath) and os.path.isfile(filepath + '.nc'):
        filepath = filepath + '.nc'
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    return [(filepath, vn.strip()) for vn in varnames]


def parse_target_months(s):
    """Parsea '2025-03' o '2025-01:2025-12' a lista de (month_start, month_end)."""
    if ':' in s:
        parts    = s.split(':')
        start_ts = pd.Timestamp(parts[0])
        end_ts   = pd.Timestamp(parts[1])
    else:
        start_ts = end_ts = pd.Timestamp(s)

    months = pd.date_range(start_ts, end_ts, freq='MS')
    result = []
    for m in months:
        m_end = m + pd.offsets.MonthEnd(0)
        result.append((m, m_end))
    return result


# ═════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── [1] Cargar metadata del entrenamiento ─────────────────────
    meta_path = os.path.join(args.cache_dir, 'metadata.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"metadata.json no encontrado en {args.cache_dir}/\n"
            f"Asegúrate de que el directorio de cache del entrenamiento "
            f"esté disponible."
        )

    print("=" * 60)
    print("  PREDICCIÓN CON MODELO MULTI-TORRE")
    print("=" * 60)

    print("\n[1] Cargando metadata del entrenamiento...")
    with open(meta_path) as f:
        metadata = json.load(f)

    n_preds          = metadata['n_predictors']
    input_days       = metadata['input_days']
    n_land           = metadata['n_land']
    feat_sizes       = metadata['predictor_feat_sizes']
    pred_names       = metadata['predictor_names']
    pred_stats       = metadata['predictor_stats']
    n_valid_expected = metadata['predictor_n_valid']
    target_transform = metadata['target_transform']
    target_h         = metadata['target_h']
    target_w         = metadata['target_w']
    flat_idx         = np.array(metadata['flat_indices'], dtype=np.int32)
    lat_coords       = np.array(metadata['lat_coords'])
    lon_coords       = np.array(metadata['lon_coords'])
    lat_name         = metadata['lat_name']
    lon_name         = metadata['lon_name']

    # Clip físico — intentar leer de norm_stats si existe
    global _PRECIP_MAX_MM, CLIP_RANGES
    norm_stats_path = os.path.splitext(args.model)[0] + '_norm_stats.json'
    if os.path.isfile(norm_stats_path):
        with open(norm_stats_path) as f:
            ns = json.load(f)
        if 'precip_max_mm' in ns and args.precip_max_mm is None:
            _PRECIP_MAX_MM = ns['precip_max_mm']

    if args.precip_max_mm is not None:
        _PRECIP_MAX_MM = args.precip_max_mm

    CLIP_RANGES = {
        'log1p':    (-0.5,  float(np.log1p(_PRECIP_MAX_MM))),
        'cbrt':     (0.0,   float(_PRECIP_MAX_MM ** (1/3))),
        'pow025':   (0.0,   float(_PRECIP_MAX_MM ** 0.25)),
        'pow05':    (0.0,   float(np.sqrt(_PRECIP_MAX_MM))),
        'standard': (-10.0, 10.0),
        'none':     (0.0,   _PRECIP_MAX_MM),
    }

    clip_lo, clip_hi = CLIP_RANGES.get(target_transform, (-10.0, 1e6))

    print(f"   Grid: {target_h}×{target_w}, tierra: {n_land} px")
    print(f"   Predictores esperados: {n_preds} ({', '.join(pred_names)})")
    print(f"   Feat sizes (post-PCA): {feat_sizes}")
    print(f"   Ventana de entrada: {input_days} días")
    print(f"   Transform: {target_transform}")
    print(f"   Clip: [{clip_lo:.3f}, {clip_hi:.3f}] → [0, {_PRECIP_MAX_MM:.0f}] mm")

    # ── [2] Validar predictores ───────────────────────────────────
    print(f"\n[2] Validando predictores...")
    predictor_configs = []
    for ps in args.predictor:
        predictor_configs.extend(parse_predictor_arg(ps))

    if len(predictor_configs) != n_preds:
        raise ValueError(
            f"Se esperan {n_preds} predictores ({', '.join(pred_names)}), "
            f"pero se proporcionaron {len(predictor_configs)}."
        )

    for i, (fp, vn) in enumerate(predictor_configs):
        match_str = "✓" if vn == pred_names[i] else f"⚠ esperada: '{pred_names[i]}'"
        print(f"   pred_{i}: {vn} {match_str} ← {os.path.basename(fp)}")

    # ── [3] Meses objetivo ────────────────────────────────────────
    print(f"\n[3] Meses objetivo...")
    target_months = parse_target_months(args.target_months)
    n_months = len(target_months)
    print(f"   {n_months} mes(es): "
          f"{target_months[0][0].strftime('%Y-%m')} → "
          f"{target_months[-1][0].strftime('%Y-%m')}")

    # ── [4] Cargar modelos PCA ────────────────────────────────────
    print(f"\n[4] Cargando modelos PCA...")
    pcas = []
    for i in range(n_preds):
        if feat_sizes[i] < n_valid_expected[i]:
            pca_joblib = os.path.join(args.cache_dir, f'pca_{i}.joblib')
            pca_pkl    = os.path.join(args.cache_dir, f'pca_{i}.pkl')
            if HAS_JOBLIB and os.path.isfile(pca_joblib):
                pcas.append(joblib.load(pca_joblib))
                print(f"   pca_{i} ({pred_names[i]}): "
                      f"{n_valid_expected[i]} → {feat_sizes[i]} componentes")
            elif os.path.isfile(pca_pkl):
                with open(pca_pkl, 'rb') as f:
                    pcas.append(pickle.load(f))
                print(f"   pca_{i} ({pred_names[i]}): "
                      f"{n_valid_expected[i]} → {feat_sizes[i]} componentes")
            else:
                raise FileNotFoundError(
                    f"PCA para predictor {i} ({pred_names[i]}) no encontrado "
                    f"en {args.cache_dir}/"
                )
        else:
            pcas.append(None)
            print(f"   pca_{i} ({pred_names[i]}): "
                  f"sin PCA (feat_size={feat_sizes[i]} == n_valid)")

    # ── [5] Cargar modelo ─────────────────────────────────────────
    print(f"\n[5] Cargando modelo: {args.model}...")

    # Las funciones Lambda serializadas en .keras referencian 'tf' y 'np'
    # que no están en globals al deserializar. Inyectar en builtins.
    import builtins
    builtins.tf = tf
    builtins.np = np

    # Monkey-patch Lambda.compute_output_shape para Keras 3 que no
    # puede inferir la forma de salida automáticamente en lambdas
    # con tf.reduce_sum/mean/max etc.
    _original_lambda_cos = tf.keras.layers.Lambda.compute_output_shape

    def _patched_lambda_cos(self, input_shape):
        try:
            return _original_lambda_cos(self, input_shape)
        except NotImplementedError:
            concrete = tuple(s if s is not None else 2 for s in input_shape)
            dummy = tf.zeros(concrete)
            out = self.function(dummy)
            result = list(out.shape)
            if result:
                result[0] = None
            return tuple(result)

    tf.keras.layers.Lambda.compute_output_shape = _patched_lambda_cos

    custom_objects = {
        'PixelBias':                    PixelBias,
        'CoordinateModulation':         CoordinateModulation,
        'SpatialRefinement':            SpatialRefinement,
        'GradientAccumulationModel':    GradientAccumulationModel,
        'WarmupCosineDecayWarmRestarts': WarmupCosineDecayWarmRestarts,
    }

    try:
        model = tf.keras.models.load_model(
            args.model, custom_objects=custom_objects,
            compile=False, safe_mode=False
        )
    except TypeError:
        # safe_mode no disponible en versiones antiguas de TF
        model = tf.keras.models.load_model(
            args.model, custom_objects=custom_objects, compile=False
        )

    print(f"   ✓ Modelo cargado ({model.count_params():,} parámetros)")

    # ── [6] Procesar predictores ──────────────────────────────────
    print(f"\n[6] Procesando ventanas de {input_days} días "
          f"para {n_months} mes(es)...")

    inputs = {
        f'pred_{i}': np.zeros(
            (n_months, input_days, feat_sizes[i]), dtype=np.float32
        )
        for i in range(n_preds)
    }

    for i, (filepath, varname) in enumerate(predictor_configs):
        print(f"\n   Predictor {i} ({varname}): {os.path.basename(filepath)}")

        ds = xr.open_dataset(filepath)
        ds = _normalize_time_dim(ds)
        da = ds[varname]
        da = _normalize_spatial_dims(da)
        if not np.issubdtype(da['time'].dtype, np.datetime64):
            da['time'] = pd.to_datetime(da['time'].values)

        # Detectar píxeles válidos (misma lógica que entrenamiento)
        validity  = create_validity_mask(da)
        valid_idx = np.where(validity.ravel())[0].astype(np.int32)
        n_valid   = len(valid_idx)

        if n_valid != n_valid_expected[i]:
            raise ValueError(
                f"Predictor {i} ({varname}): {n_valid} píxeles válidos "
                f"vs {n_valid_expected[i]} en entrenamiento.\n"
                f"La grilla espacial o el patrón de NaN no coincide con "
                f"los datos de entrenamiento. Usa archivos con la misma "
                f"grilla y cobertura."
            )

        p_mean = pred_stats[i]['mean']
        p_std  = max(pred_stats[i]['std'], 1e-8)
        pca    = pcas[i]

        # Verificar cobertura temporal
        da_times = da['time'].values
        first_month_start = target_months[0][0]
        first_input_start = first_month_start - timedelta(days=input_days)
        last_month_start  = target_months[-1][0]
        last_input_end    = last_month_start - timedelta(days=1)

        if np.datetime64(first_input_start) < da_times[0]:
            print(f"   ⚠ AVISO: el archivo empieza en "
                  f"{pd.Timestamp(da_times[0]).strftime('%Y-%m-%d')}, "
                  f"pero se necesitan datos desde "
                  f"{first_input_start.strftime('%Y-%m-%d')}")
        if np.datetime64(last_input_end) > da_times[-1]:
            print(f"   ⚠ AVISO: el archivo termina en "
                  f"{pd.Timestamp(da_times[-1]).strftime('%Y-%m-%d')}, "
                  f"pero se necesitan datos hasta "
                  f"{last_input_end.strftime('%Y-%m-%d')}")

        for mi, (m_start, m_end) in enumerate(target_months):
            input_end   = m_start - timedelta(days=1)
            input_start = input_end - timedelta(days=input_days - 1)

            # Extraer ventana temporal
            try:
                window = da.sel(time=slice(input_start, input_end)).values
            except Exception:
                i_s = int(np.argmin(
                    np.abs(da_times - np.datetime64(input_start))))
                i_e = int(np.argmin(
                    np.abs(da_times - np.datetime64(input_end))))
                window = da.isel(time=slice(i_s, i_e + 1)).values

            if window.ndim == 2:
                window = np.repeat(window[np.newaxis], input_days, axis=0)

            T = window.shape[0]

            if T == 0:
                # Sin datos → llenar con ceros normalizados
                inputs[f'pred_{i}'][mi] = np.zeros(
                    (input_days, feat_sizes[i]), dtype=np.float32
                )
                continue

            # Padding si no hay suficientes días
            if T < input_days:
                pad = np.full(
                    (input_days - T,) + window.shape[1:],
                    p_mean, dtype=np.float32
                )
                window = np.concatenate([window, pad], axis=0)
            elif T > input_days:
                window = window[:input_days]

            # Aplanar y seleccionar píxeles válidos
            flat = window.reshape(window.shape[0], -1)[:, valid_idx]
            flat = flat.astype(np.float32)
            flat = np.nan_to_num(flat, nan=p_mean, posinf=p_mean, neginf=p_mean)

            # Normalizar con stats de entrenamiento
            flat = (flat - p_mean) / p_std

            # Aplicar PCA si corresponde
            if pca is not None:
                flat = pca.transform(flat).astype(np.float32)
                # Truncar a feat_size (el entrenamiento puede guardar más
                # componentes de los que realmente usa)
                if flat.shape[1] > feat_sizes[i]:
                    flat = flat[:, :feat_sizes[i]]

            inputs[f'pred_{i}'][mi] = flat

        ds.close()
        gc.collect()
        print(f"   ✓ {n_months} ventanas procesadas "
              f"→ shape ({n_months}, {input_days}, {feat_sizes[i]})")

    # ── [7] Ejecutar predicciones ─────────────────────────────────
    print(f"\n[7] Ejecutando predicciones...")
    Y_pred = model.predict(inputs, batch_size=args.batch_size, verbose=1)

    # Clip en espacio transformado
    Y_pred = np.clip(Y_pred, clip_lo, clip_hi)

    # Transformada inversa → mm
    target_mean = metadata.get('target_mean')
    target_std  = metadata.get('target_std')

    if target_transform == 'log1p':
        Y_mm = np.expm1(Y_pred)
    elif target_transform == 'cbrt':
        Y_mm = np.power(np.maximum(Y_pred, 0.0), 3.0)
    elif target_transform == 'pow025':
        Y_mm = np.power(np.maximum(Y_pred, 0.0), 4.0)
    elif target_transform == 'pow05':
        Y_mm = np.power(np.maximum(Y_pred, 0.0), 2.0)
    elif target_transform == 'standard' and target_mean is not None:
        Y_mm = Y_pred * (target_std or 1.0) + (target_mean or 0.0)
    else:
        Y_mm = Y_pred.copy()

    Y_mm = np.maximum(Y_mm, 0.0).astype(np.float32)

    print(f"\n   Predicciones (mm):")
    print(f"     min:  {Y_mm.min():.2f}")
    print(f"     max:  {Y_mm.max():.2f}")
    print(f"     mean: {Y_mm.mean():.2f}")

    # ── [8] Reconstruir grilla 2D ─────────────────────────────────
    print(f"\n[8] Reconstruyendo grilla 2D ({target_h}×{target_w})...")
    Y_grid = np.full((n_months, target_h, target_w), np.nan, dtype=np.float32)
    for t in range(n_months):
        Y_grid[t].ravel()[flat_idx] = Y_mm[t]

    time_coords = np.array([
        np.datetime64(m_start) for m_start, _ in target_months
    ])

    # ── [9] Guardar NetCDF ────────────────────────────────────────
    print(f"\n[9] Guardando NetCDF: {args.output}...")
    ds_out = xr.Dataset({
        'predicted': xr.DataArray(
            Y_grid,
            dims=['time', lat_name, lon_name],
            coords={
                'time':  time_coords,
                lat_name: lat_coords,
                lon_name: lon_coords,
            },
            attrs={
                'long_name': 'Predicted monthly precipitation',
                'units':     'mm',
            },
        ),
    })
    ds_out.attrs.update({
        'title':             'Multi-tower model predictions',
        'source':            f'predict.py + {os.path.basename(args.model)}',
        'target_transform':  target_transform,
        'precip_max_mm':     _PRECIP_MAX_MM,
        'clip_transformed':  f'[{clip_lo:.3f}, {clip_hi:.3f}]',
        'input_days':        input_days,
        'n_predictors':      n_preds,
        'predictor_names':   ', '.join(pred_names),
        'creation_date':     pd.Timestamp.now().isoformat(),
    })

    output_nc = None
    alt_nc = (os.path.splitext(args.output)[0]
              + f"_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.nc")

    for cand in [args.output, alt_nc]:
        try:
            if os.path.exists(cand):
                os.remove(cand)
            try:
                ds_out.to_netcdf(cand, mode='w', format='NETCDF4',
                                 engine='netcdf4')
            except Exception:
                ds_out.to_netcdf(cand, mode='w', engine='scipy')
            output_nc = cand
            print(f"   ✓ {output_nc}")
            break
        except Exception as exc:
            print(f"   ✗ {cand}: {exc}")

    if output_nc is None:
        raise RuntimeError(
            "No se pudo guardar el NetCDF en ninguna ruta candidata."
        )

    # ── Resumen ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREDICCIÓN COMPLETADA")
    print("=" * 60)
    print(f"  Modelo:      {args.model}")
    print(f"  Output:      {output_nc}")
    print(f"  Meses:       {n_months}")
    print(f"  Transform:   {target_transform} → mm")
    print(f"  Rango pred:  {Y_mm.min():.1f} – {Y_mm.max():.1f} mm")
    print(f"  Mean pred:   {Y_mm.mean():.1f} mm")
    print("=" * 60)


if __name__ == '__main__':
    main()
