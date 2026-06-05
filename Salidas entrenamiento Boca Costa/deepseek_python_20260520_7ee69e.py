import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# ------------------------------------------------------------
# Configuración de meses lluviosos y secos (ajustable)
# ------------------------------------------------------------
MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
# Por defecto: lluviosos = Oct-Mar (primavera-verano austral)
MESES_LLUVIOSOS = ['Oct', 'Nov', 'Dic', 'Ene', 'Feb', 'Mar']
MESES_SECOS = ['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep']


def parsear_tabla_mensual(bloque: str) -> pd.DataFrame:
    """
    Extrae la tabla de métricas por mes dentro del bloque de una entrada.
    Retorna un DataFrame con columnas: Mes, RMSE, MAE, Bias, r, NSE, KGE
    """
    # Buscar la sección "7. Métricas por mes"
    patron_seccion = r"─── 7\. Métricas por mes ───\s*\n(.*?)(?:─── 8\.|$)"
    match = re.search(patron_seccion, bloque, re.DOTALL | re.IGNORECASE)
    if not match:
        return pd.DataFrame()  # vacío

    tabla_texto = match.group(1)
    lineas = tabla_texto.strip().split('\n')
    if len(lineas) < 13:
        return pd.DataFrame()  # no hay suficientes líneas

    # La primera línea después del encabezado son los nombres de columna
    # (asumimos que la línea 0 o 1 contiene "Mes    RMSE ...")
    # Buscamos la línea que contiene "Mes"
    idx_encabezado = None
    for i, linea in enumerate(lineas):
        if 'Mes' in linea and 'RMSE' in linea:
            idx_encabezado = i
            break
    if idx_encabezado is None:
        return pd.DataFrame()

    # Extraer nombres de columnas
    encabezado = lineas[idx_encabezado].split()
    # Normalmente: ['Mes', 'RMSE', 'MAE', 'Bias', 'r', 'NSE', 'KGE']
    # Puede haber 'p' (Pearson?) pero en el ejemplo es 'r'
    # Nos quedamos con los que nos interesan
    cols = encabezado[:7]  # asumimos 7 columnas
    if len(cols) != 7:
        # Si no, intentar rellenar con nombres conocidos
        cols = ['Mes', 'RMSE', 'MAE', 'Bias', 'r', 'NSE', 'KGE']

    # Las siguientes 12 líneas son los datos de cada mes
    datos = []
    for linea in lineas[idx_encabezado+1: idx_encabezado+13]:
        if not linea.strip():
            continue
        partes = linea.split()
        if len(partes) >= 7:
            mes = partes[0]
            valores = [float(x) for x in partes[1:7]]
            datos.append([mes] + valores)
    if len(datos) != 12:
        return pd.DataFrame()

    df_mes = pd.DataFrame(datos, columns=cols)
    return df_mes


def parse_archivo_completo(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee el archivo y retorna:
      - df_global: métricas globales por combinación/entrada (igual que antes)
      - df_mensual: métricas mensuales con columnas:
          Combinacion, Entrada, Mes, RMSE, MAE, Bias, r, NSE, KGE
    """
    text = Path(filepath).read_text(encoding="utf-8", errors="ignore")

    combo_pattern = re.compile(
        r"COMBINACIÓN\s+(\d+)(.*?)(?=COMBINACIÓN\s+\d+|$)",
        re.DOTALL | re.IGNORECASE
    )
    entry_pattern = re.compile(
        r"Entrada\s+(\d+)\..*?(?=Entrada\s+\d+\.|$)",
        re.DOTALL | re.IGNORECASE
    )

    # Métricas globales
    metric_patterns = {
        "RMSE": r"RMSE\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "MAE": r"MAE\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "Bias": r"Bias\s*\(media\)\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "PctBias": r"%\s*Bias\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "StdError": r"Std\(error\)\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "R2": r"R²\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "Pearson_r": r"Pearson\s+r\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "Spearman_rho": r"Spearman\s+ρ\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "NSE": r"Nash-Sutcliffe\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "KGE": r"Kling-Gupta\s+\(KGE\)\s*=\s*([-+]?\d+(?:\.\d+)?)",
    }

    rows_global = []
    rows_mensual = []

    for combo_match in combo_pattern.finditer(text):
        combo_id = int(combo_match.group(1))
        combo_block = combo_match.group(2)

        for entry_match in entry_pattern.finditer(combo_block):
            entry_id = int(entry_match.group(1))
            entry_block = entry_match.group(0)

            # Métricas globales
            row_global = {"Combinacion": combo_id, "Entrada": entry_id}
            for metric_name, pattern in metric_patterns.items():
                m = re.search(pattern, entry_block, re.IGNORECASE)
                row_global[metric_name] = float(m.group(1)) if m else np.nan
            rows_global.append(row_global)

            # Métricas mensuales
            df_mes = parsear_tabla_mensual(entry_block)
            if not df_mes.empty:
                df_mes = df_mes.copy()
                df_mes.insert(0, "Combinacion", combo_id)
                df_mes.insert(1, "Entrada", entry_id)
                rows_mensual.append(df_mes)

    df_global = pd.DataFrame(rows_global)
    df_mensual = pd.concat(rows_mensual, ignore_index=True) if rows_mensual else pd.DataFrame()
    return df_global, df_mensual


def resumir_por_combinacion(df_global: pd.DataFrame) -> pd.DataFrame:
    """Promedio de métricas globales por combinación."""
    metric_cols = [c for c in df_global.columns if c not in ["Combinacion", "Entrada"]]
    return df_global.groupby("Combinacion", as_index=False)[metric_cols].mean()


def resumir_mensual_por_combinacion(df_mensual: pd.DataFrame) -> pd.DataFrame:
    """
    Promedia las métricas mensuales sobre las entradas para cada combinación y mes.
    Retorna un DataFrame con columnas: Combinacion, Mes, RMSE, MAE, Bias, r, NSE, KGE
    """
    metric_cols = ["RMSE", "MAE", "Bias", "r", "NSE", "KGE"]
    # Asegurar que existan
    for col in metric_cols:
        if col not in df_mensual.columns:
            df_mensual[col] = np.nan
    return df_mensual.groupby(["Combinacion", "Mes"], as_index=False)[metric_cols].mean()


def ranking_mensual(df_mensual_res: pd.DataFrame, metrica: str = "KGE") -> pd.DataFrame:
    """
    Por cada mes, rankea las combinaciones según la métrica elegida (mayor = mejor).
    Retorna un DataFrame con columnas: Combinacion, Mes, rank, y la métrica.
    """
    # Para cada mes, ordenar por métrica descendente
    rankings = []
    for mes in df_mensual_res["Mes"].unique():
        sub = df_mensual_res[df_mensual_res["Mes"] == mes].copy()
        sub["Rank"] = sub[metrica].rank(ascending=False, method="average")
        rankings.append(sub)
    df_rank = pd.concat(rankings, ignore_index=True)
    return df_rank[["Combinacion", "Mes", metrica, "Rank"]]


def analizar_por_estacion(df_mensual_res: pd.DataFrame,
                          meses_lluviosos: List[str],
                          meses_secos: List[str],
                          metrica: str = "KGE") -> pd.DataFrame:
    """
    Calcula el rendimiento promedio por combinación en los dos grupos de meses.
    Retorna un DataFrame con Combinacion, KGE_lluvioso, KGE_seco, diferencia.
    """
    df_lluv = df_mensual_res[df_mensual_res["Mes"].isin(meses_lluviosos)]
    df_seco = df_mensual_res[df_mensual_res["Mes"].isin(meses_secos)]

    lluv_prom = df_lluv.groupby("Combinacion")[metrica].mean().rename(f"{metrica}_lluvioso")
    seco_prom = df_seco.groupby("Combinacion")[metrica].mean().rename(f"{metrica}_seco")
    resultado = pd.concat([lluv_prom, seco_prom], axis=1).reset_index()
    resultado["diferencia"] = resultado[f"{metrica}_lluvioso"] - resultado[f"{metrica}_seco"]
    return resultado.sort_values(f"{metrica}_lluvioso", ascending=False)


def graficar_kge_mensual(df_mensual_res: pd.DataFrame, savepath: str = "kge_mensual.png"):
    """Gráfico de líneas del KGE por mes para cada combinación."""
    plt.figure(figsize=(12, 6))
    for combo in sorted(df_mensual_res["Combinacion"].unique()):
        sub = df_mensual_res[df_mensual_res["Combinacion"] == combo].sort_values("Mes")
        # Ordenar meses según la lista MESES
        sub = sub.set_index("Mes").reindex(MESES).reset_index()
        plt.plot(sub["Mes"], sub["KGE"], marker='o', label=f"Combo {int(combo)}")
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Mes")
    plt.ylabel("KGE promedio")
    plt.title("Rendimiento mensual por combinación (KGE)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def graficar_ranking_estacional(estacion_df: pd.DataFrame, savepath: str = "ranking_estacional.png"):
    """Barras para comparar KGE en lluviosos vs secos."""
    estacion_df = estacion_df.sort_values("KGE_lluvioso", ascending=False)
    combos = estacion_df["Combinacion"].astype(str).tolist()
    x = np.arange(len(combos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, estacion_df["KGE_lluvioso"], width, label="Meses lluviosos")
    ax.bar(x + width/2, estacion_df["KGE_seco"], width, label="Meses secos")
    ax.set_xlabel("Combinación")
    ax.set_ylabel("KGE promedio")
    ax.set_title("Comparación de desempeño: meses lluviosos vs secos")
    ax.set_xticks(x)
    ax.set_xticklabels(combos)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Etiquetas de valores
    for i, (ll, se) in enumerate(zip(estacion_df["KGE_lluvioso"], estacion_df["KGE_seco"])):
        ax.text(i - width/2, ll + 0.01, f"{ll:.2f}", ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, se + 0.01, f"{se:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    # Cambiar por la ruta real del archivo
    filepath = "ENTRADAS_BOCA_COSTA.txt"

    print("Parsing del archivo...")
    df_global, df_mensual = parse_archivo_completo(filepath)

    if df_global.empty:
        raise ValueError("No se encontraron datos globales.")
    if df_mensual.empty:
        raise ValueError("No se encontraron tablas mensuales. Verifique que el archivo contenga la sección '7. Métricas por mes'.")

    # Resúmenes
    resumen_global = resumir_por_combinacion(df_global)
    resumen_mensual = resumir_mensual_por_combinacion(df_mensual)

    print("\n=== Promedio global por combinación (KGE, RMSE, etc.) ===")
    print(resumen_global[["Combinacion", "KGE", "RMSE", "MAE", "Pearson_r"]].to_string(index=False))

    # Ranking mensual
    ranking_mes = ranking_mensual(resumen_mensual, metrica="KGE")
    # Mostrar ranking promedio por combinación (orden inverso: mejor combinación en promedio mensual)
    ranking_promedio = ranking_mes.groupby("Combinacion")["Rank"].mean().sort_values().reset_index()
    ranking_promedio.columns = ["Combinacion", "RankPromedio"]
    print("\n=== Ranking promedio (menor rank = mejor) ===")
    print(ranking_promedio.to_string(index=False))

    # Análisis estacional
    print(f"\nMeses considerados LLUVIOSOS: {MESES_LLUVIOSOS}")
    print(f"Meses considerados SECOS: {MESES_SECOS}")
    estacional = analizar_por_estacion(resumen_mensual, MESES_LLUVIOSOS, MESES_SECOS, metrica="KGE")
    print("\n=== Rendimiento por estación ===")
    print(estacional.to_string(index=False))

    # Identificar mejor para lluviosos y mejor para secos
    mejor_lluv = estacional.loc[estacional["KGE_lluvioso"].idxmax()]
    mejor_seco = estacional.loc[estacional["KGE_seco"].idxmax()]
    print(f"\n🌟 Mejor combinación para meses lluviosos: Combo {int(mejor_lluv['Combinacion'])} (KGE = {mejor_lluv['KGE_lluvioso']:.3f})")
    print(f"🌵 Mejor combinación para meses secos: Combo {int(mejor_seco['Combinacion'])} (KGE = {mejor_seco['KGE_seco']:.3f})")

    # Gráficos
    graficar_kge_mensual(resumen_mensual, savepath="kge_mensual_comparacion.png")
    graficar_ranking_estacional(estacional, savepath="estacional_comparacion.png")

    # Opcional: mostrar combinación con menor diferencia (más equilibrada)
    estacional["abs_dif"] = estacional["diferencia"].abs()
    mas_balanceada = estacional.loc[estacional["abs_dif"].idxmin()]
    print(f"⚖️ Combinación más equilibrada (menor diferencia lluvioso-seco): Combo {int(mas_balanceada['Combinacion'])} (dif = {mas_balanceada['diferencia']:.3f})")


if __name__ == "__main__":
    main()