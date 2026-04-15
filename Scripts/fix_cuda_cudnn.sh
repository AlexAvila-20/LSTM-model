#!/bin/bash
# fix_cuda_cudnn.sh — Limpia e reinstala CUDA/cuDNN para arreglar error status 5003

set -e

echo "════════════════════════════════════════════════════════════"
echo "Reinstalando TensorFlow + CUDA/cuDNN"
echo "════════════════════════════════════════════════════════════"

# Obtener el nombre del env conda actual
CONDA_ENV=$(conda info --json | python -c "import sys, json; print(json.load(sys.stdin)['active_prefix'].split('/')[-1])")

echo ""
echo "🔍 Ambiente Conda actual: $CONDA_ENV"
echo ""

# Limpiar pip cache
echo "Limpiando caché de pip..."
pip cache purge 2>/dev/null || true

# Desinstalar tensorflow y CUDA/cuDNN
echo ""
echo "Desinstalando tensorflow y CUDA/cuDNN..."
pip uninstall -y tensorflow tensorflow-gpu tensorflow-macos tensorflow-metal 2>/dev/null || true
pip uninstall -y nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-toolkit 2>/dev/null || true
conda remove -y cudatoolkit cudnn 2>/dev/null || true

# Limpiar caché de conda
echo "Limpiando caché de conda..."
conda clean -a -y

# Limpiar caché local de tensorflow/keras
echo "Limpiando caché local de TensorFlow/Keras..."
rm -rf ~/.cache/pip ~/.cache/keras ~/.keras 2>/dev/null || true
rm -rf /tmp/xla* /tmp/tf* /tmp/keras* 2>/dev/null || true

# Reinstalar tensorflow con CUDA 12.x backend
echo ""
echo "Reinstalando TensorFlow 2.21.0 con CUDA 12..."
pip install --upgrade pip setuptools

pip install \
  tensorflow==2.21.0 \
  nvidia-cudnn-cu12==9.2.0.82 \
  nvidia-cuda-runtime-cu12==12.9.79 \
  --no-cache-dir

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Reinstalación completada"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Verificando instalación..."
python << 'EOF'
import os
# Deshabilitar determinismo extremo
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detectadas: {len(gpus)}")
if gpus:
    print(f"GPU: {gpus[0]}")
print("✓ TensorFlow está funcionando correctamente")
EOF

echo ""
echo "Ya puedes ejecutar: python MODELO.py"
