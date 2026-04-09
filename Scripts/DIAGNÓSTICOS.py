#!/usr/bin/env python3
"""
diagnosticos.py  –  Diagnósticos completos de predicción vs observación
=======================================================================
Genera ~15 figuras y un resumen estadístico en consola a partir de un
archivo NetCDF con variables "observed" y "predicted" dimensionadas
(time, lat, lon).

Uso:
    python diagnosticos.py                              # usa archivo por defecto
    python diagnosticos.py  mi_prediccion.nc            # archivo personalizado
    python diagnosticos.py  mi_prediccion.nc  --save    # guarda PNGs en disco
"""

import sys, os, warnings
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Configuración ────────────────────────────────────────────────────
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
ARCHIVO = _args[0] if _args else "modelo_pixel_predictions.nc"
GUARDAR = "--save" in sys.argv
OUTDIR  = "diagnosticos_figs"
if GUARDAR:
    os.makedirs(OUTDIR, exist_ok=True)

MESES_ES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
             7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}


# ─── Funciones auxiliares ─────────────────────────────────────────────
def guardar_o_mostrar(fig, nombre):
    if GUARDAR:
        fig.savefig(os.path.join(OUTDIR, nombre), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [guardado] {nombre}")
    # Si no se guarda, se muestra al final con plt.show()


def clean(obs, pred):
    """Aplana y elimina NaN de ambos arrays."""
    o = obs.ravel().astype(np.float64)
    p = pred.ravel().astype(np.float64)
    mask = np.isfinite(o) & np.isfinite(p)
    return o[mask], p[mask]


def nse(obs, pred):
    """Nash-Sutcliffe Efficiency."""
    return 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)


def kge(obs, pred):
    """Kling-Gupta Efficiency."""
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta  = np.mean(pred) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def pbias(obs, pred):
    """Porcentaje de sesgo."""
    return 100 * np.sum(pred - obs) / np.sum(obs)


# ─── Cargar datos ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f" Diagnósticos – {ARCHIVO}")
print(f"{'='*60}\n")

ds = xr.open_dataset(ARCHIVO)
obs_da  = ds["observed"]
pred_da = ds["predicted"]
obs_flat, pred_flat = clean(obs_da.values, pred_da.values)

times = ds["time"].values


# =====================================================================
# 1. ESTADÍSTICAS GLOBALES
# =====================================================================
print("─── 1. Estadísticas globales ───")
rmse_val   = np.sqrt(mean_squared_error(obs_flat, pred_flat))
mae_val    = mean_absolute_error(obs_flat, pred_flat)
r2_val     = r2_score(obs_flat, pred_flat)
r_val, p_r = pearsonr(obs_flat, pred_flat)
rho, p_rho = spearmanr(obs_flat, pred_flat)
nse_val    = nse(obs_flat, pred_flat)
kge_val    = kge(obs_flat, pred_flat)
bias_val   = np.mean(pred_flat - obs_flat)
pbias_val  = pbias(obs_flat, pred_flat)
std_err    = np.std(pred_flat - obs_flat)

stats = {
    "RMSE":              f"{rmse_val:.4f}",
    "MAE":               f"{mae_val:.4f}",
    "Bias (media)":      f"{bias_val:.4f}",
    "% Bias":            f"{pbias_val:.2f} %",
    "Std(error)":        f"{std_err:.4f}",
    "R²":                f"{r2_val:.4f}",
    "Pearson r":         f"{r_val:.4f}  (p={p_r:.2e})",
    "Spearman ρ":        f"{rho:.4f}  (p={p_rho:.2e})",
    "Nash-Sutcliffe":    f"{nse_val:.4f}",
    "Kling-Gupta (KGE)": f"{kge_val:.4f}",
}
for k, v in stats.items():
    print(f"  {k:25s} = {v}")
print()


# =====================================================================
# 2. DISPERSIÓN MEJORADA (hexbin + densidad)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Dispersión Observado vs Predicho", fontsize=13)

# hexbin
ax = axes[0]
hb = ax.hexbin(obs_flat, pred_flat, gridsize=60, cmap="YlOrRd",
               mincnt=1, linewidths=0.2)
fig.colorbar(hb, ax=ax, label="Conteo")
mx = max(obs_flat.max(), pred_flat.max()) * 1.05
ax.plot([0, mx], [0, mx], "k--", lw=1, label="1:1")
ax.set(xlabel="Observado", ylabel="Predicho", title="Hexbin",
       xlim=(0, mx), ylim=(0, mx))
ax.legend(fontsize=9)

# scatter con alpha
ax = axes[1]
ax.scatter(obs_flat, pred_flat, s=3, alpha=0.15, edgecolors="none")
ax.plot([0, mx], [0, mx], "r--", lw=1, label="1:1")
# línea de regresión
coef = np.polyfit(obs_flat, pred_flat, 1)
ax.plot([0, mx], [coef[1], coef[0]*mx + coef[1]], "b-", lw=1,
        label=f"Regresión (m={coef[0]:.2f})")
ax.set(xlabel="Observado", ylabel="Predicho", title="Scatter + regresión",
       xlim=(0, mx), ylim=(0, mx))
ax.legend(fontsize=9)
fig.tight_layout()
guardar_o_mostrar(fig, "01_dispersion.png")

# =====================================================================
# 4. MAPAS ESPACIALES DE BIAS Y RMSE POR PÍXEL
# =====================================================================
dif = pred_da - obs_da

# Bias medio por píxel
bias_map = dif.mean(dim="time")
# RMSE por píxel
rmse_map = np.sqrt((dif**2).mean(dim="time"))
# Correlación por píxel
corr_map = xr.corr(obs_da, pred_da, dim="time")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Métricas espaciales por píxel", fontsize=13)

# Bias
ax = axes[0]
lim = max(abs(float(bias_map.min())), abs(float(bias_map.max())))
im = bias_map.plot(ax=ax, cmap="RdBu_r", vmin=-lim, vmax=lim,
                   add_colorbar=True, cbar_kwargs={"label": "mm"})
ax.set_title("Bias medio (pred − obs)")

# RMSE
ax = axes[1]
rmse_map.plot(ax=ax, cmap="YlOrRd", add_colorbar=True,
              cbar_kwargs={"label": "mm"})
ax.set_title("RMSE")

# Correlación
ax = axes[2]
corr_map.plot(ax=ax, cmap="RdYlGn", vmin=0, vmax=1, add_colorbar=True,
              cbar_kwargs={"label": "r"})
ax.set_title("Correlación Pearson")

fig.tight_layout()
guardar_o_mostrar(fig, "03_mapas_metricas.png")


# =====================================================================
# 5. SERIES TEMPORALES – PROMEDIO ESPACIAL
# =====================================================================
obs_ts  = obs_da.mean(dim=["lat", "lon"])
pred_ts = pred_da.mean(dim=["lat", "lon"])

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.suptitle("Series temporales (promedio espacial)", fontsize=13)

ax = axes[0]
ax.plot(times, obs_ts, label="Observado", color="k")
ax.plot(times, pred_ts, label="Predicho", color="dodgerblue", ls="--")
ax.fill_between(times,
                (pred_da.quantile(0.25, dim=["lat","lon"])).values,
                (pred_da.quantile(0.75, dim=["lat","lon"])).values,
                alpha=0.2, color="dodgerblue", label="IQR predicho")
ax.set(ylabel="Precipitación (mm)", title="Observado vs Predicho")
ax.legend(fontsize=9)

ax = axes[1]
dif_ts = pred_ts - obs_ts
colors = ["firebrick" if v > 0 else "steelblue" for v in dif_ts.values]
ax.bar(times, dif_ts, color=colors, width=20)
ax.axhline(0, color="k", lw=0.5)
ax.set(ylabel="Diferencia (mm)", xlabel="Tiempo",
       title="Diferencia predicho − observado")
fig.tight_layout()
guardar_o_mostrar(fig, "04_series_temporales.png")


# =====================================================================
# 6. CICLO ANUAL (promedio mensual)
# =====================================================================
obs_month  = obs_da.groupby("time.month").mean(dim="time").mean(dim=["lat","lon"])
pred_month = pred_da.groupby("time.month").mean(dim="time").mean(dim=["lat","lon"])

fig, ax = plt.subplots(figsize=(8, 5))
meses = obs_month.month.values
labels = [MESES_ES.get(m, m) for m in meses]
ax.plot(meses, obs_month, "ko-", label="Observado")
ax.plot(meses, pred_month, "s--", color="dodgerblue", label="Predicho")
ax.fill_between(meses, obs_month, pred_month, alpha=0.15, color="gray")
ax.set(xticks=meses, xlabel="Mes", ylabel="Precipitación (mm)",
       title="Ciclo anual medio")
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
guardar_o_mostrar(fig, "05_ciclo_anual.png")


# =====================================================================
# 7. MÉTRICAS POR MES
# =====================================================================
import pandas as pd
t_pd = pd.DatetimeIndex(times)

monthly_stats = []
for m in sorted(t_pd.month.unique()):
    sel = t_pd.month == m
    o, p = clean(obs_da.values[sel], pred_da.values[sel])
    if len(o) < 10:
        continue
    monthly_stats.append({
        "Mes": MESES_ES[m],
        "RMSE":  np.sqrt(mean_squared_error(o, p)),
        "MAE":   mean_absolute_error(o, p),
        "Bias":  np.mean(p - o),
        "r":     np.corrcoef(o, p)[0,1],
        "NSE":   nse(o, p),
        "KGE":   kge(o, p),
    })
df_m = pd.DataFrame(monthly_stats)

print("─── 7. Métricas por mes ───")
print(df_m.to_string(index=False, float_format="%.3f"))
print()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Métricas mensuales", fontsize=13)
for ax, col, color in zip(axes.flat,
                          ["RMSE", "Bias", "r", "KGE"],
                          ["firebrick","steelblue","seagreen","darkorange"]):
    ax.bar(df_m["Mes"], df_m[col], color=color, alpha=0.8)
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=45)
    if col == "Bias":
        ax.axhline(0, color="k", lw=0.5)
fig.tight_layout()
guardar_o_mostrar(fig, "06_metricas_mes.png")


# =====================================================================
# 8. MÉTRICAS POR AÑO
# =====================================================================
yearly_stats = []
for y in sorted(t_pd.year.unique()):
    sel = t_pd.year == y
    o, p = clean(obs_da.values[sel], pred_da.values[sel])
    if len(o) < 10:
        continue
    yearly_stats.append({
        "Año":  y,
        "RMSE": np.sqrt(mean_squared_error(o, p)),
        "Bias": np.mean(p - o),
        "r":    np.corrcoef(o, p)[0,1],
        "NSE":  nse(o, p),
        "KGE":  kge(o, p),
    })
df_y = pd.DataFrame(yearly_stats)

print("─── 8. Métricas por año ───")
print(df_y.to_string(index=False, float_format="%.3f"))
print()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_y["Año"], df_y["KGE"], "o-", label="KGE", color="darkorange")
ax.plot(df_y["Año"], df_y["NSE"], "s-", label="NSE", color="seagreen")
ax.plot(df_y["Año"], df_y["r"],   "^-", label="r",   color="royalblue")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set(xlabel="Año", ylabel="Valor", title="Métricas anuales")
ax.legend()
fig.tight_layout()
guardar_o_mostrar(fig, "07_metricas_anual.png")


# =====================================================================
# 11. ANÁLISIS DE EXTREMOS  (percentiles altos)
# =====================================================================
print("─── 11. Análisis de extremos ───")

pcts = [90, 95, 99]
for p in pcts:
    umb = np.percentile(obs_flat, p)
    sel = obs_flat >= umb
    o_e, p_e = obs_flat[sel], pred_flat[sel]
    r_e = np.corrcoef(o_e, p_e)[0, 1]
    rmse_e = np.sqrt(mean_squared_error(o_e, p_e))
    bias_e = np.mean(p_e - o_e)
    print(f"  P{p} (>= {umb:.1f} mm): n={sel.sum()}, "
          f"r={r_e:.3f}, RMSE={rmse_e:.1f}, Bias={bias_e:.1f}")

# QQ-plot
fig, ax = plt.subplots(figsize=(6, 6))
quantiles = np.linspace(0, 100, 200)
obs_q  = np.percentile(obs_flat, quantiles)
pred_q = np.percentile(pred_flat, quantiles)
ax.plot(obs_q, pred_q, "o", ms=3, color="steelblue")
mx = max(obs_q.max(), pred_q.max()) * 1.05
ax.plot([0, mx], [0, mx], "r--", lw=1)
ax.set(xlabel="Quantiles observados", ylabel="Quantiles predichos",
       title="QQ-plot", xlim=(0, mx), ylim=(0, mx))
ax.set_aspect("equal")
fig.tight_layout()
guardar_o_mostrar(fig, "10_qqplot.png")
print()


# =====================================================================
# 13. DISTRIBUCIÓN ACUMULADA (CDF)
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 5))
obs_sorted = np.sort(obs_flat)
pred_sorted = np.sort(pred_flat)
n = len(obs_sorted)
cdf = np.arange(1, n+1) / n
ax.plot(obs_sorted, cdf, label="Observado", color="k")
ax.plot(pred_sorted, cdf, label="Predicho", color="dodgerblue", ls="--")
ax.set(xlabel="Precipitación (mm)", ylabel="CDF",
       title="Distribución acumulada")
ax.legend()
fig.tight_layout()
guardar_o_mostrar(fig, "12_cdf.png")


# =====================================================================
# 14. MAPA DE NSE Y KGE POR PÍXEL
# =====================================================================
nlat = len(ds.lat)
nlon = len(ds.lon)
nse_grid = np.full((nlat, nlon), np.nan)
kge_grid = np.full((nlat, nlon), np.nan)

obs_vals  = obs_da.values   # (time, lat, lon)
pred_vals = pred_da.values

for i in range(nlat):
    for j in range(nlon):
        o = obs_vals[:, i, j].astype(np.float64)
        p = pred_vals[:, i, j].astype(np.float64)
        mask = np.isfinite(o) & np.isfinite(p)
        if mask.sum() < 10:
            continue
        o, p = o[mask], p[mask]
        nse_grid[i, j] = nse(o, p)
        kge_grid[i, j] = kge(o, p)

nse_da = xr.DataArray(nse_grid, coords=[ds.lat, ds.lon], dims=["lat","lon"])
kge_da = xr.DataArray(kge_grid, coords=[ds.lat, ds.lon], dims=["lat","lon"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Eficiencia por píxel", fontsize=13)

nse_da.plot(ax=axes[0], cmap="RdYlGn", vmin=-0.5, vmax=1,
            add_colorbar=True, cbar_kwargs={"label":""})
axes[0].set_title("Nash-Sutcliffe (NSE)")

kge_da.plot(ax=axes[1], cmap="RdYlGn", vmin=-0.5, vmax=1,
            add_colorbar=True, cbar_kwargs={"label":""})
axes[1].set_title("Kling-Gupta (KGE)")

fig.tight_layout()
guardar_o_mostrar(fig, "13_nse_kge_mapa.png")


# =====================================================================
# 15. TENDENCIA TEMPORAL DEL ERROR
# =====================================================================
rmse_ts = []
bias_ts = []
for t_idx in range(len(times)):
    o, p = clean(obs_vals[t_idx], pred_vals[t_idx])
    if len(o) < 10:
        rmse_ts.append(np.nan)
        bias_ts.append(np.nan)
        continue
    rmse_ts.append(np.sqrt(mean_squared_error(o, p)))
    bias_ts.append(np.mean(p - o))

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle("Evolución temporal del error", fontsize=13)

ax = axes[0]
ax.plot(times, rmse_ts, "o-", ms=3, color="firebrick")
z = np.polyfit(np.arange(len(rmse_ts)), rmse_ts, 1)
ax.plot(times, np.polyval(z, np.arange(len(rmse_ts))), "k--", lw=1,
        label=f"tendencia: {z[0]:.2f}/paso")
ax.set(ylabel="RMSE (mm)", title="RMSE por paso temporal")
ax.legend(fontsize=9)

ax = axes[1]
colors_ts = ["firebrick" if b > 0 else "steelblue" for b in bias_ts]
ax.bar(times, bias_ts, color=colors_ts, width=20)
ax.axhline(0, color="k", lw=0.5)
ax.set(ylabel="Bias (mm)", xlabel="Tiempo", title="Bias por paso temporal")
fig.tight_layout()
guardar_o_mostrar(fig, "14_tendencia_error.png")


# =====================================================================
# 16. RESUMEN DE PERCENTILES CRUZADOS
# =====================================================================
print("─── 16. Percentiles cruzados ───")
for p in [10, 25, 50, 75, 90, 95]:
    print(f"  P{p:2d}  obs={np.percentile(obs_flat,p):8.2f}   "
          f"pred={np.percentile(pred_flat,p):8.2f}   "
          f"ratio={np.percentile(pred_flat,p)/np.percentile(obs_flat,p):.3f}")
print()


# =====================================================================
# 17. EXPORTAR MÉTRICAS ESPACIALES A NETCDF
# =====================================================================
ds_out = xr.Dataset({
    "bias":  bias_map,
    "rmse":  rmse_map,
    "corr":  corr_map,
    "nse":   nse_da,
    "kge":   kge_da,
})
ds_out.attrs["description"] = "Métricas espaciales pixel-a-pixel"
ds_out.attrs["source"] = ARCHIVO
outfile = ARCHIVO.replace(".nc", "_diagnosticos.nc")
ds_out.to_netcdf(outfile)
print(f"  Métricas espaciales guardadas en: {outfile}")


# =====================================================================
# MOSTRAR TODAS LAS FIGURAS
# =====================================================================
if not GUARDAR:
    print("\n  Mostrando figuras (cierra las ventanas para continuar)...")
    plt.show()
else:
    print(f"\n  Todas las figuras guardadas en: {OUTDIR}/")

print(f"\n{'='*60}")
print(" Diagnósticos completados")
print(f"{'='*60}\n")
