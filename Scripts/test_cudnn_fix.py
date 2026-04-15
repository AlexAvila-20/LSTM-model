#!/usr/bin/env python
"""
test_cudnn_fix.py — Verifica que la configuración de cuDNN es estable
Prueba la compilación del modelo LSTM sin entrenar datos completos
"""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# CRÍTICO: Desahabilitar autotuning de cuDNN que causa status 5003
os.environ['TF_FORCE_CUDNN_USE_AUTOTUNE'] = '0'
# Desahabilitar XLA/JIT
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, BatchNormalization,
    TimeDistributed, Dropout
)

# ─── Wrapper LSTM sin cuDNN ──────────────────────────────────────────
class LSTMNoCuDNN(LSTM):
    """LSTM que desahabilita explícitamente cuDNN para evitar status 5003."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_cudnn_kernel = False

print("=" * 70)
print("TEST: Validación de configuración cuDNN post-cambio de usuario")
print("=" * 70)

# Verificar GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"✓ GPUs detectadas: {len(gpus)}")
print(f"✓ TensorFlow: {tf.__version__}")

# Configuración de teste
n_timesteps = 31
n_features = 256
lstm_units = 96
batch_size = 64

print(f"\nConfigurando modelo de teste:")
print(f"  - Timesteps: {n_timesteps}")
print(f"  - Features: {n_features}")
print(f"  - LSTM units: {lstm_units}")
print(f"  - Batch size: {batch_size}")

try:
    # Construir modelo minimal similar al original
    inp = Input(shape=(n_timesteps, n_features), batch_size=batch_size)
    x = TimeDistributed(Dense(128, activation='relu'))(inp)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    
    # BiLSTM 1 - con activaciones explícitas para mejor estabilidad cuDNN
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True,
             activation='tanh', recurrent_activation='sigmoid',
             use_bias=True, unit_forget_bias=True),
        name='bilstm_1'
    )(x)
    x = BatchNormalization()(x)
    
    # BiLSTM 2
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True,
             activation='tanh', recurrent_activation='sigmoid',
             use_bias=True, unit_forget_bias=True),
        name='bilstm_2'
    )(x)
    
    out = Dense(1, activation='relu')(x)
    model = Model(inp, out)
    
    print("\n✓ Modelo construido exitosamente")
    print(model.summary())
    
    # Compilar con jit_compile=False
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'],
        jit_compile=False  # ← CRÍTICO: Evita conflictos cuDNN
    )
    print("\n✓ Modelo compilado exitosamente (jit_compile=False)")
    
    # Realizar forward pass
    print("\nRealizando forward pass de prueba...")
    dummy_input = tf.random.normal((batch_size, n_timesteps, n_features))
    output = model(dummy_input, training=False)
    print(f"✓ Forward pass exitoso. Output shape: {output.shape}")
    
    # Intentar un mini-batch training
    print("\nEntrenando 1 época con datos dummy...")
    dummy_target = tf.random.normal((batch_size, n_timesteps, 1))
    dummy_dataset = tf.data.Dataset.from_tensors(
        (dummy_input, dummy_target)
    ).repeat(2).batch(32)
    
    history = model.fit(dummy_dataset, epochs=1, verbose=1)
    print("\n✓ Entrenamiento completado sin errores cuDNN")
    
    print("\n" + "=" * 70)
    print("✅ ÉXITO: Configuración de cuDNN es estable")
    print("=" * 70)
    print("\nYa puedes ejecutar MODELO.py con confianza.")
    
except Exception as e:
    print(f"\n❌ ERROR durante validación: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
