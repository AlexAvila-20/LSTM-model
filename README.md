# LSTM Precipitation Prediction Model

Modelo de **predicción de precipitación mensual** acumulada (mm) basado en redes LSTM multi-torre, entrenado con múltiples variables predictoras (SST, presión, humedad, velocidad del viento, etc.). El proyecto incluye scripts para entrenamiento, predicción, diagnósticos y visualización geoespacial.

---

## Estructura de Archivos

### Scripts Principales

#### **MODELO.py**
Script de **entrenamiento del modelo LSTM**. Construye y entrena una arquitectura de red neuronal con:
- **Múltiples torres de entrada** (una por cada predictor) con embeddings espaciales
- **Capas LSTM** para capturar dependencias temporales
- **Atención multi-cabeza** para ponderación de características
- **Optimización de RAM** mediante lectura por chunks y PCA dinámico
- Genera archivo `.keras` con los pesos entrenados

**Uso:**
```bash
python MODELO.py --precip ENACTS2_prcp.nc --varname rfe \
  --predictor "era5.sst.nc:sst" \
  --predictor "era5.1.nc:z,r,q" \
  --n_jobs 4 --max_ram_gb 8.0 --epochs 4000 --pca_components 0
```

#### **PRONÓSTICO.py**
Script de **predicción de precipitación** para nuevos datos. Usa el modelo entrenado para generar pronósticos a partir de predictores nuevos.
- Lee el modelo `.keras` y la metadata de entrenamiento del cache
- Aplica las transformaciones PCA pre-calculadas
- Genera predicciones por chunks para eficiencia
- Exporta resultados en formato NetCDF

**Uso:**
```bash
python PRONÓSTICO.py \
  --model modelo_pixel.keras \
  --cache_dir ./preprocessing_cache \
  --predictor sst_new.nc:sst \
  --target_months 2025-01:2025-12 \
  --output predicciones_2025.nc
```

#### **DIAGNÓSTICOS.py**
Script de **evaluación y visualización de predicciones**. Genera ~15 figuras con:
- Mapas de predicciones vs observaciones
- Series temporales comparativas
- Métricas estadísticas (RMSE, MAE, R², correlación)
- Análisis de errores extremos
- Diagramas de dispersión

**Uso:**
```bash
python DIAGNÓSTICOS.py modelo_pixel_predictions.nc --save
```

#### **DEP.py**
Script de **mapeo y agregación por departamentos**. Utiliza:
- Shapefile de límites administrativos de Guatemala
- Carga predicciones del archivo NetCDF
- Calcula estadísticas (media, mediana, percentiles) por departamento
- Mapeo visual de resultados por región

**Uso:**
```bash
python DEP.py
# Lee "modelo_pixel_predictions.nc" por defecto
```

#### **MAPA.py**
Script de **visualización cartográfica simple**. Genera:
- Mapas raster de datos NetCDF
- Selección por fecha
- Opcional: acumulación temporal
- Exportación de mapas estáticos

**Uso:**
```bash
python MAPA.py
# Lee "predicciones_2026.nc" por defecto
```

---

### Archivos de Datos

#### **modelo_pixel.keras**
Modelo entrenado con la arquitectura LSTM compilada y pesos. Se genera después de ejecutar `MODELO.py` y se usa en `PRONÓSTICO.py` para hacer nuevas predicciones.

#### **modelo_pixel_predictions.nc**
Archivo NetCDF con predicciones generadas. Contiene variables dimensionadas `(time, lat, lon)`:
- `predicted`: precipitación predicha (mm)
- `observed`: precipitación observada (mm, si disponible)
- Coordenadas: `time`, `lat`, `lon`

#### **modelo_pixel_norm_stats.json**
Estadísticas de normalización del objetivo (precipitación):
- Media y desviación estándar
- Parámetros de transformación logarítmica (si aplica)
Se usa para desnormalizar predicciones

#### **ENTRADAS.txt**
Documento de **referencia con comandos de ejecución**. Contiene:
- Combinaciones de predictores probadas (6 variantes)
- Comandos listos para copiar y ejecutar
- Parámetros recomendados por combinación
- Notas sobre resultados de diagnósticos

#### **README.md**
Este archivo. Documentación del proyecto.

---

### Carpeta: `preprocessing_cache/`

Contiene datos de preprocesamiento reutilizables entre entrenamientos y predicciones:

| Archivo | Descripción |
|---------|-------------|
| **metadata.json** | Metadatos del preprocesamiento (forma de datos, coordenadas, variables) |
| **base_config.json** | Configuración base (variables, ranges temporales, grid) |
| **pca_0.joblib** | Componentes PCA para predictor 1 (reconstrucción de 99.9% varianza) |
| **pca_1.joblib** | Componentes PCA para predictor 2 |
| **pca_2.joblib** | Componentes PCA para predictor 3 |
| **pred_0.npy** | Array de predictor 1 preprocesado (formato comprimido/memmap) |
| **pred_1.npy** | Array de predictor 2 |
| **pred_2.npy** | Array de predictor 3 |
| **pred_3.npy** | Array de predictor 4 |
| **pred_4.npy** | Array de predictor 5 |
| **target.npy** | Array de variable objetivo (precipitación observada) |

Estos archivos acelaran entrenamientos posteriores y aseguran que la predicción use exactamente las mismas transformaciones del entrenamiento.

---

## Flujo de Trabajo Típico

```
1. ENTRENAMIENTO
   ├─ Cargar predictores (NetCDF)
   ├─ Preprocesar → cache/ (.npy, .joblib, .json)
   ├─ Entrenar LSTM → modelo_pixel.keras
   └─ Evaluar métricas

2. PRONÓSTICO
   ├─ Cargar modelo_pixel.keras
   ├─ Usar transformaciones de cache/
   ├─ Predicción sobre nuevos datos NetCDF
   └─ Exportar → modelo_pixel_predictions.nc

3. DIAGNÓSTICOS
   ├─ Leer predicciones + observaciones
   ├─ Calcular métricas (RMSE, R², correlación)
   └─ Generar 15 figuras de análisis

4. VISUALIZACIÓN POR DEPARTAMENTOS
   ├─ Mapear a límites administrativos
   ├─ Agregar estadísticas por región
   └─ Mapas temáticos de resultados
```

---

## Parámetros Clave

Los scripts usan parámetros comunes como:
- `--pca_components 0`: Auto-PCA con umbral de varianza
- `--spatial_embed 512`: Dimensionalidad de embedding espacial
- `--hidden_dim 256`: Unidades LSTM escondidas
- `--batch_size 64`: Tamaño de lotes para entrenamiento
- `--epochs 4000`: Número de iteraciones
- `--max_ram_gb 8.0`: Límite de memoria para optimización

Ver `ENTRADAS.txt` para ejemplos completos.

---

## Dependencias

- **TensorFlow/Keras**: Construcción y entrenamiento del modelo
- **xarray/NetCDF4**: Manejo de datos geoespaciales
- **scikit-learn**: PCA incremental y métricas
- **gtMapTools**: Visualización y operaciones raster geoespaciales
- **shapefile**: Lectura de límites administrativos
- **numpy, pandas, scipy**: Computación numérica y estadística

