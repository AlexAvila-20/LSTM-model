import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_metrics_file(filepath: str) -> pd.DataFrame:
    """
    Lee un archivo de resultados con secciones tipo:
      COMBINACIÓN 1
      Entrada 1
      ...
      RMSE = ...
      MAE = ...
      ...
    y devuelve un DataFrame con una fila por entrada.
    """
    text = Path(filepath).read_text(encoding="utf-8", errors="ignore")

    # Divide por combinaciones
    combo_pattern = re.compile(
        r"COMBINACIÓN\s+(\d+)(.*?)(?=COMBINACIÓN\s+\d+|$)",
        re.DOTALL | re.IGNORECASE
    )

    # Divide por entradas dentro de cada combinación
    entry_pattern = re.compile(
        r"Entrada\s+(\d+)\..*?(?=Entrada\s+\d+\.|$)",
        re.DOTALL | re.IGNORECASE
    )

    # Métricas globales que queremos capturar
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

    rows = []

    for combo_match in combo_pattern.finditer(text):
        combo_id = int(combo_match.group(1))
        combo_block = combo_match.group(2)

        for entry_match in entry_pattern.finditer(combo_block):
            entry_id = int(entry_match.group(1))
            entry_block = entry_match.group(0)

            row = {
                "Combinacion": combo_id,
                "Entrada": entry_id,
            }

            for metric_name, pattern in metric_patterns.items():
                m = re.search(pattern, entry_block, re.IGNORECASE)
                row[metric_name] = float(m.group(1)) if m else np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summarize_by_combination(df: pd.DataFrame) -> pd.DataFrame:
    """
    Promedia métricas por combinación.
    """
    metric_cols = [c for c in df.columns if c not in ["Combinacion", "Entrada"]]
    summary = (
        df.groupby("Combinacion", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values("Combinacion")
    )
    return summary


def rank_combinations(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Crea un ranking simple:
    - Menor es mejor: RMSE, MAE, |Bias|, |PctBias|, StdError
    - Mayor es mejor: R2, Pearson_r, Spearman_rho, NSE, KGE
    """
    df = summary.copy()

    # Magnitudes absolutas para bias
    if "Bias" in df.columns:
        df["AbsBias"] = df["Bias"].abs()
    if "PctBias" in df.columns:
        df["AbsPctBias"] = df["PctBias"].abs()

    lower_better = ["RMSE", "MAE", "AbsBias", "AbsPctBias", "StdError"]
    higher_better = ["R2", "Pearson_r", "Spearman_rho", "NSE", "KGE"]

    # Solo usar columnas existentes
    lower_better = [c for c in lower_better if c in df.columns]
    higher_better = [c for c in higher_better if c in df.columns]

    # Ranking por métrica
    for col in lower_better:
        df[f"rank_{col}"] = df[col].rank(ascending=True, method="average")
    for col in higher_better:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="average")

    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    df["ScorePromedio"] = df[rank_cols].mean(axis=1)

    return df.sort_values("ScorePromedio")


def plot_comparison(summary: pd.DataFrame, savepath: str = "comparacion_combinaciones.png"):
    """
    Genera una figura con varias métricas por combinación.
    """
    combos = summary["Combinacion"].astype(str).tolist()

    metrics_to_plot = [
        ("RMSE", "RMSE"),
        ("MAE", "MAE"),
        ("R2", "R²"),
        ("Pearson_r", "Pearson r"),
        ("NSE", "NSE"),
        ("KGE", "KGE"),
    ]

    metrics_to_plot = [(col, label) for col, label in metrics_to_plot if col in summary.columns]

    n = len(metrics_to_plot)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (col, label) in zip(axes, metrics_to_plot):
        values = summary[col].values
        ax.bar(combos, values)
        ax.set_title(label)
        ax.set_xlabel("Combinación")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)

        # Etiquetas numéricas
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Apagar ejes sobrantes
    for ax in axes[len(metrics_to_plot):]:
        ax.axis("off")

    fig.suptitle("Comparación de combinaciones de predictores", fontsize=14)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def plot_single_score(ranked: pd.DataFrame, savepath: str = "ranking_combinaciones.png"):
    """
    Gráfica de score promedio de ranking.
    Menor score = mejor combinación global.
    """
    df = ranked.sort_values("ScorePromedio")
    combos = df["Combinacion"].astype(str).tolist()
    values = df["ScorePromedio"].values

    plt.figure(figsize=(10, 5))
    plt.bar(combos, values)
    plt.xlabel("Combinación")
    plt.ylabel("Score promedio de ranking")
    plt.title("Ranking global de combinaciones (menor es mejor)")
    plt.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Cambia esto por el nombre real de tu archivo
    filepath = "ENTRADAS.txt"

    df = parse_metrics_file(filepath)

    if df.empty:
        raise ValueError("No se encontraron combinaciones o entradas en el archivo.")

    print("\n=== Datos por entrada ===")
    print(df.to_string(index=False))

    summary = summarize_by_combination(df)

    print("\n=== Promedio por combinación ===")
    print(summary.to_string(index=False))

    ranked = rank_combinations(summary)

    print("\n=== Ranking global ===")
    cols_to_show = ["Combinacion", "ScorePromedio"]
    extra = [c for c in ["RMSE", "MAE", "R2", "Pearson_r", "NSE", "KGE"] if c in ranked.columns]
    print(ranked[cols_to_show + extra].to_string(index=False))

    plot_comparison(summary)
    plot_single_score(ranked)
