"""Ingeniería de características para el forecasting mensual de Predict Future Sales.

Contiene la construcción del dataset base (grid mensual tienda-producto con ceros) y los
transformadores personalizados del pipeline de sklearn. Los transformadores viven en este
módulo (y no en un notebook) para que ``joblib``/``pickle`` pueda deserializar los ``.pkl``
desde cualquier notebook o script: ``pickle`` guarda la ruta de la clase, no su código.

Transformadores:

- :class:`SalesHistoryFeatures`: convierte filas ``(date_block_num, shop_id, item_id)`` en
  variables de historia (lags, precios, antigüedad, banderas). Solo usa meses **anteriores**
  al mes de cada fila, por lo que no hay fuga de información.
- :class:`Mapper`: mapea valores con diccionarios (misma interfaz que el ``QualityOrdinalMapper``
  visto en clase: ``variables`` + mapeo; aquí el mapeo es un parámetro y puede escribir en una
  columna nueva).
- :class:`ColumnSelector` / :class:`DtypeDowncaster`: utilitarios de selección de columnas y de
  reducción de memoria.
"""

import itertools
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import typer

from predict_future_sales.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

#: Llaves de una fila del problema: mes, tienda y producto.
KEYS = ["date_block_num", "shop_id", "item_id"]
#: Variable objetivo y su rango de evaluación en Kaggle.
TARGET = "item_cnt_month"
TARGET_CLIP = (0, 20)
#: Mes que hay que predecir (noviembre 2015).
TEST_BLOCK = 34


# ---------------------------------------------------------------------------
# Dataset base
# ---------------------------------------------------------------------------
def build_monthly_grid(monthly: pd.DataFrame, clip: tuple[int, int] = TARGET_CLIP) -> pd.DataFrame:
    """Construye el grid mensual tienda × producto con ceros en los meses sin ventas.

    Para cada mes se toma el producto cartesiano entre las tiendas y los productos que tuvieron
    alguna venta ese mes (el mismo criterio con el que Kaggle armó ``test.csv``: tiendas y
    productos activos). Los pares sin venta reciben ``item_cnt_month = 0``.

    Args:
        monthly: ventas mensuales por par (``date_block_num, shop_id, item_id, item_cnt_month``),
            p. ej. ``data/processed/monthly_shop_item.csv`` del notebook 01.
        clip: rango al que se recorta el objetivo (Kaggle evalúa en [0, 20]).

    Returns:
        DataFrame ``date_block_num, shop_id, item_id, item_cnt_month`` con tipos compactos.
    """
    parts = []
    for block, grp in monthly.groupby("date_block_num"):
        combos = itertools.product([block], grp["shop_id"].unique(), grp["item_id"].unique())
        parts.append(np.array(list(combos), dtype=np.int32))
    grid = pd.DataFrame(np.vstack(parts), columns=KEYS)
    grid = grid.merge(monthly[KEYS + [TARGET]], on=KEYS, how="left")
    grid[TARGET] = grid[TARGET].fillna(0).clip(*clip).astype(np.float32)
    grid["date_block_num"] = grid["date_block_num"].astype(np.int8)
    grid["shop_id"] = grid["shop_id"].astype(np.int8)
    grid["item_id"] = grid["item_id"].astype(np.int16)
    return grid.sort_values(KEYS).reset_index(drop=True)


def build_mappings(
    items: pd.DataFrame, categories: pd.DataFrame, shops: pd.DataFrame
) -> dict[str, dict]:
    """Diccionarios de mapeo que usa el ``Mapper`` del pipeline.

    - ``city``: ``shop_id`` -> ciudad (extraída del nombre de la tienda en el EDA).
    - ``shop_type``: ``shop_id`` -> tipo de tienda (centro comercial, calle, online...).
    - ``category_group``: ``item_category_id`` -> grupo de categoría (Juegos, Cine, Libros...).
    """
    from predict_future_sales.dataset import SHOP_CITY_ES, build_catalogs

    items_full, shops_full = build_catalogs(items, categories, shops)
    cats = items_full.drop_duplicates("item_category_id")
    return {
        "city": dict(SHOP_CITY_ES),
        "shop_type": shops_full.set_index("shop_id")["shop_type"].to_dict(),
        "category_group": cats.set_index("item_category_id")["category_group"].to_dict(),
    }


# ---------------------------------------------------------------------------
# Transformadores
# ---------------------------------------------------------------------------
class SalesHistoryFeatures(BaseEstimator, TransformerMixin):
    """Genera variables de historia de ventas para filas ``(date_block_num, shop_id, item_id)``.

    El transformador guarda la historia mensual de ventas (``history``) y, para cada fila del mes
    ``m``, consulta **solo los meses < m**. Así, el mismo objeto sirve para entrenar (meses
    0-33), validar y predecir el mes 34 de ``test.csv`` sin fuga de información: para el mes 34
    los lags salen de los meses 33, 32, 31, ...

    Variables generadas:

    - ``item_category_id``: categoría del producto (catálogo ``items``).
    - ``cnt_lag_k`` (par tienda-producto), ``item_cnt_lag_k`` (producto, todas las tiendas),
      ``cat_cnt_lag_k`` (categoría), ``shop_cnt_lag_k`` (tienda), ``shop_cat_cnt_lag_k``
      (tienda × categoría): ventas (recortadas a ``clip`` a nivel par) del mes ``m - k``.
      Los niveles agregados son el **promedio por par** del grupo, para que la escala no
      dependa del número de tiendas/productos. ``NaN`` cuando ``m - k < 0`` (antes del inicio de
      los datos: se imputa en el pipeline).
    - ``item_price_last``: último precio medio mensual conocido del producto (``NaN`` si el
      producto nunca se ha vendido).
    - ``item_age`` / ``pair_age``: meses desde la primera venta del producto / del par (0 si es
      nuevo); ``item_is_new`` / ``pair_is_new``: banderas de producto o par sin historia.
    - ``month``: mes del año (0 = enero) y ``days_in_month``.

    Args:
        history: ventas mensuales por par (``date_block_num, shop_id, item_id, item_cnt_month,
            revenue``). Puede incluir meses posteriores a los de ``X``: nunca se consultan.
        items: catálogo ``items.csv`` (``item_id, item_category_id``).
        pair_lags, item_lags, cat_lags, shop_lags, shop_cat_lags: rezagos por nivel.
        clip: recorte de la venta mensual del par (igual que el objetivo).
    """

    def __init__(
        self,
        history: pd.DataFrame,
        items: pd.DataFrame,
        pair_lags: tuple[int, ...] = (1, 2, 3, 6, 12),
        item_lags: tuple[int, ...] = (1, 2, 3),
        cat_lags: tuple[int, ...] = (1,),
        shop_lags: tuple[int, ...] = (1,),
        shop_cat_lags: tuple[int, ...] = (1,),
        clip: tuple[int, int] = TARGET_CLIP,
    ):
        self.history = history
        self.items = items
        self.pair_lags = pair_lags
        self.item_lags = item_lags
        self.cat_lags = cat_lags
        self.shop_lags = shop_lags
        self.shop_cat_lags = shop_cat_lags
        self.clip = clip

    # -- preparación de tablas de consulta (no depende de y: solo de la historia) ----------
    def fit(self, X: pd.DataFrame, y=None):
        h = self.history[KEYS + [TARGET, "revenue"]].copy()
        h[TARGET] = h[TARGET].clip(*self.clip)
        cat_map = self.items.set_index("item_id")["item_category_id"]
        h["item_category_id"] = h["item_id"].map(cat_map)

        # número de tiendas / productos activos por mes (denominador de los promedios)
        n_shops = h.groupby("date_block_num")["shop_id"].nunique()
        n_items = h.groupby("date_block_num")["item_id"].nunique()
        n_items_cat = h.groupby(["date_block_num", "item_category_id"])["item_id"].nunique()

        self.cat_map_ = cat_map.astype(np.int16)
        self.pair_cnt_ = h.set_index(KEYS)[TARGET].astype(np.float32)
        item_sum = h.groupby(["date_block_num", "item_id"])[TARGET].sum()
        self.item_cnt_ = (item_sum / n_shops.reindex(item_sum.index, level=0)).astype(np.float32)
        cat_sum = h.groupby(["date_block_num", "item_category_id"])[TARGET].sum()
        denom = n_items_cat * n_shops.reindex(n_items_cat.index, level=0)
        self.cat_cnt_ = (cat_sum / denom).astype(np.float32)
        shop_sum = h.groupby(["date_block_num", "shop_id"])[TARGET].sum()
        self.shop_cnt_ = (shop_sum / n_items.reindex(shop_sum.index, level=0)).astype(np.float32)
        sc_sum = h.groupby(["date_block_num", "shop_id", "item_category_id"])[TARGET].sum()
        sc_den = n_items_cat.reindex(sc_sum.droplevel("shop_id").index).to_numpy()
        self.shop_cat_cnt_ = pd.Series(sc_sum.to_numpy() / sc_den, index=sc_sum.index).astype(
            np.float32
        )

        # precio medio mensual del producto (ingreso / unidades, solo meses con venta neta > 0)
        hp = h[h[TARGET] > 0].groupby(["item_id", "date_block_num"])[["revenue", TARGET]].sum()
        price = (hp["revenue"] / hp[TARGET]).rename("price").reset_index()
        self.item_price_ = price.sort_values(["date_block_num", "item_id"])

        self.item_first_ = h.groupby("item_id")["date_block_num"].min()
        self.pair_first_ = h.groupby(["shop_id", "item_id"])["date_block_num"].min()
        self.max_block_ = int(h["date_block_num"].max())
        return self

    @staticmethod
    def _lookup(keys: pd.DataFrame, table: pd.Series, lag: int) -> np.ndarray:
        """Valor de ``table`` en el mes ``date_block_num - lag`` para cada fila de ``keys``.

        Fila sin registro en un mes válido -> 0 (no hubo ventas). Mes anterior al inicio de los
        datos (``m - lag < 0``) -> NaN (sin información).
        """
        k = keys.copy()
        k["date_block_num"] = k["date_block_num"] - lag
        idx = pd.MultiIndex.from_frame(k[list(table.index.names)])
        values = table.reindex(idx).to_numpy(dtype=np.float32, na_value=np.nan)
        values = np.where(np.isnan(values), 0.0, values)
        values[(k["date_block_num"] < 0).to_numpy()] = np.nan
        return values.astype(np.float32)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["date_block_num"] = X["date_block_num"].astype(np.int16)
        X["item_category_id"] = X["item_id"].map(self.cat_map_).fillna(-1).astype(np.int16)
        base = X[["date_block_num", "shop_id", "item_id", "item_category_id"]]

        levels = [
            ("cnt", self.pair_cnt_, self.pair_lags),
            ("item_cnt", self.item_cnt_, self.item_lags),
            ("cat_cnt", self.cat_cnt_, self.cat_lags),
            ("shop_cnt", self.shop_cnt_, self.shop_lags),
            ("shop_cat_cnt", self.shop_cat_cnt_, self.shop_cat_lags),
        ]
        for name, table, lags in levels:
            for lag in lags:
                X[f"{name}_lag_{lag}"] = self._lookup(base, table, lag)

        # último precio conocido del producto (estrictamente antes del mes de la fila)
        order = np.argsort(X["date_block_num"].to_numpy(), kind="stable")
        left = X.iloc[order][["date_block_num", "item_id"]].reset_index()
        left["item_id"] = left["item_id"].astype(np.int64)
        left["date_block_num"] = left["date_block_num"].astype(np.int64)
        right = self.item_price_.astype({"item_id": np.int64, "date_block_num": np.int64})
        merged = pd.merge_asof(
            left,
            right,
            on="date_block_num",
            by="item_id",
            allow_exact_matches=False,  # solo meses < m
        )
        X["item_price_last"] = (
            merged.set_index("index")["price"].reindex(X.index).astype(np.float32)
        )

        # antigüedad y banderas de novedad (exógenas de producto y de par)
        item_first = X["item_id"].map(self.item_first_)
        pair_idx = pd.MultiIndex.from_frame(X[["shop_id", "item_id"]])
        pair_first = pd.Series(self.pair_first_.reindex(pair_idx).to_numpy(), index=X.index)
        m = X["date_block_num"]
        item_known = item_first < m
        pair_known = pair_first < m
        X["item_is_new"] = (~item_known).astype(np.int8)
        X["pair_is_new"] = (~pair_known).astype(np.int8)
        X["item_age"] = np.where(item_known, m - item_first, 0).astype(np.int16)
        X["pair_age"] = np.where(pair_known, m - pair_first, 0).astype(np.int16)

        # calendario
        X["month"] = (m % 12).astype(np.int8)
        X["days_in_month"] = (
            pd.to_datetime({"year": 2013 + m // 12, "month": m % 12 + 1, "day": 1})
            .dt.days_in_month.astype(np.int8)
            .to_numpy()
        )
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Mapea los valores de variables con un diccionario (operador ``Mapper`` de clase).

    Misma interfaz que el ``QualityOrdinalMapper`` del curso (``variables`` + mapeo, ``fit`` sin
    estado, ``transform`` sobre una copia), generalizada: el mapeo se recibe como parámetro y,
    opcionalmente, el resultado se escribe en una columna nueva.

    Args:
        variables: columnas de entrada a mapear.
        mappings: diccionario ``{valor_original: valor_nuevo}`` (el mismo para todas las
            ``variables``).
        new_variables: nombres de salida (misma longitud que ``variables``). Si es ``None`` se
            sobrescriben las columnas originales. Los valores sin mapeo quedan como ``NaN``
            (los atiende el imputador categórico del pipeline).
    """

    def __init__(
        self, variables: list[str], mappings: dict, new_variables: list[str] | None = None
    ):
        self.variables = variables
        self.mappings = mappings
        self.new_variables = new_variables

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Transformador sin estado: siempre está 'ajustado' (convención de sklearn)."""
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        outputs = self.new_variables or self.variables
        for src, dst in zip(self.variables, outputs):
            X[dst] = X[src].map(self.mappings)
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Conserva solo las columnas indicadas (en ese orden)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Transformador sin estado: siempre está 'ajustado' (convención de sklearn)."""
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()


class DtypeDowncaster(BaseEstimator, TransformerMixin):
    """Convierte todas las columnas numéricas a ``float32`` (la mitad de memoria que float64)."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Transformador sin estado: siempre está 'ajustado' (convención de sklearn)."""
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Definición de variables (decididas en notebooks/02_feature_exploration.ipynb)
# ---------------------------------------------------------------------------
#: Rezagos por nivel de agregación.
PAIR_LAGS = (1, 2, 3, 6, 12)
ITEM_LAGS = (1, 2, 3, 6, 12)
GROUP_LAGS = (1, 2, 12)  # categoría, tienda y tienda × categoría

#: Variables de rezago generadas por SalesHistoryFeatures con los rezagos anteriores.
LAG_VARS = (
    [f"cnt_lag_{k}" for k in PAIR_LAGS]
    + [f"item_cnt_lag_{k}" for k in ITEM_LAGS]
    + [f"{lvl}_lag_{k}" for lvl in ("cat_cnt", "shop_cnt", "shop_cat_cnt") for k in GROUP_LAGS]
)
#: i. Imputación numérica: NaN = mes anterior al inicio de los datos / producto sin precio.
NUM_IMPUTE_VARS = LAG_VARS + ["item_price_last"]
#: Indicadores de faltante (agregados antes de imputar).
MISSING_INDICATOR_VARS = [f"cnt_lag_{k}" for k in PAIR_LAGS] + ["item_price_last"]
#: ii. Imputación categórica (salidas del Mapper).
CAT_VARS = ["city", "shop_type", "category_group"]
#: iii. Codificación.
ONEHOT_VARS = ["shop_type"]
MEAN_ENCODE_VARS = ["city", "category_group"]
#: iv. Outliers: capping del precio en el percentil 99.
OUTLIER_VARS = ["item_price_last"]
#: v. Transformación log(x + 1).
LOG_VARS = LAG_VARS + ["item_price_last", "pair_age"]
#: Llaves que no entran al modelo (el mes absoluto no se extrapola al mes 34).
DROP_VARS = ["date_block_num", "item_id"]


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "monthly_shop_item.csv",
    output_path: Path = PROCESSED_DATA_DIR / "monthly_grid.pkl",
) -> None:
    """Genera el grid mensual tienda × producto con ceros (meses 0-33)."""
    monthly = pd.read_csv(input_path)
    grid = build_monthly_grid(monthly)
    test = pd.read_csv(RAW_DATA_DIR / "test.csv")
    logger.info(f"Grid: {len(grid):,} filas; test (mes {TEST_BLOCK}): {len(test):,} filas")
    grid.to_pickle(output_path)
    logger.success(f"Grid guardado en {output_path}")


if __name__ == "__main__":
    app()
