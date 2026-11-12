"""Funciones de visualización reutilizables para el EDA de Predict Future Sales."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from predict_future_sales.config import FIGURES_DIR

#: Paleta base del proyecto (consistente en todas las gráficas).
COLOR_MAIN = "#2a6f97"
COLOR_ACCENT = "#e07a5f"
COLOR_MUTED = "#9aa5b1"


def set_style() -> None:
    """Estilo común de matplotlib para todo el notebook."""
    plt.rcParams.update(
        {
            "figure.figsize": (12, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
        }
    )


def save_fig(fig: plt.Figure, name: str, figures_dir: Path = FIGURES_DIR) -> Path:
    """Guarda la figura como PNG en ``reports/figures`` y devuelve la ruta."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def thousands_formatter(decimals: int = 0) -> FuncFormatter:
    """Formateador de ejes con sufijos legibles: 1.2M, 350k."""

    def _fmt(x: float, _pos: int) -> str:
        if abs(x) >= 1e6:
            return f"{x / 1e6:.{max(decimals, 1)}f}M"
        if abs(x) >= 1e3:
            return f"{x / 1e3:.{decimals}f}k"
        return f"{x:.0f}"

    return FuncFormatter(_fmt)


MONTHS_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def block_label(date_block_num: int) -> str:
    """Etiqueta en español de un ``date_block_num`` (0 -> 'Ene-13')."""
    year = 2013 + date_block_num // 12
    return f"{MONTHS_ES[date_block_num % 12]}-{str(year)[2:]}"


def date_formatter_es() -> FuncFormatter:
    """Formateador de ejes de fecha de matplotlib con meses en español ('Ene-13')."""
    import matplotlib.dates as mdates

    def _fmt(x: float, _pos: int) -> str:
        d = mdates.num2date(x)
        return f"{MONTHS_ES[d.month - 1]}-{str(d.year)[2:]}"

    return FuncFormatter(_fmt)


def plot_acf_pacf(
    series: pd.Series, title: str, lags: int, axes: tuple[plt.Axes, plt.Axes] | None = None
) -> tuple[plt.Axes, plt.Axes]:
    """Dibuja ACF y PACF lado a lado para una serie.

    La PACF usa el método ``ywm`` y requiere ``lags < len(series) / 2``; si se pide más se
    recorta automáticamente.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    pacf_lags = min(lags, len(series) // 2 - 1)
    plot_acf(series, lags=lags, ax=axes[0], color=COLOR_MAIN, vlines_kwargs={"colors": COLOR_MAIN})
    plot_pacf(
        series,
        lags=pacf_lags,
        ax=axes[1],
        method="ywm",
        color=COLOR_ACCENT,
        vlines_kwargs={"colors": COLOR_ACCENT},
    )
    axes[0].set_title(f"ACF · {title}")
    axes[1].set_title(f"PACF · {title}")
    for ax in axes:
        ax.set_xlabel("Rezago")
    return axes
