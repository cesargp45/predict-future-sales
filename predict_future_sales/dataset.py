"""Carga, enriquecimiento y limpieza del dataset Predict Future Sales (Kaggle, 1C Company).

Este módulo concentra la lógica reutilizable que el EDA (notebook 01) usa y que las etapas
siguientes (feature engineering / forecasting) van a necesitar:

- ``load_raw_data``: lee los 5 CSV originales de ``data/raw``.
- ``CATEGORY_NAMES_ES`` / ``SHOP_CITY_ES``: traducciones al español de los catálogos (vienen en ruso).
- ``add_catalog_info``: une ventas con catálogos y agrega columnas derivadas (fecha, ingreso, etc.).
- ``clean_sales``: aplica las decisiones de limpieza justificadas en el EDA (sección 4).

Uso por línea de comandos (genera ``data/interim/sales_clean.csv``)::

    python -m predict_future_sales.dataset
"""

from pathlib import Path
import re

from loguru import logger
import pandas as pd
import typer

from predict_future_sales.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# ---------------------------------------------------------------------------------------------
# Constantes de negocio (decisiones documentadas en el notebook 01_exploratory_data_analysis)
# ---------------------------------------------------------------------------------------------

#: Tiendas duplicadas: mismo nombre (o casi) con distinto ID y SIN ventas en meses simultáneos.
#: Se mapea el ID "viejo" al ID que sigue activo y que aparece en test.csv.
SHOP_DUPLICATES: dict[int, int] = {0: 57, 1: 58, 11: 10}

#: Umbrales para eliminar transacciones extremas puntuales (ver sección 4b / 7c del EDA).
MAX_ITEM_PRICE = 100_000  # una sola venta de 307,980 (licencia corporativa de software)
MAX_ITEM_CNT_DAY = 1_000  # una sola fila de 2,169 unidades de un servicio de envío

#: Traducción al español de las 84 categorías (item_categories.csv viene en ruso).
CATEGORY_NAMES_ES: dict[int, str] = {
    0: "PC - Auriculares/Headsets",
    1: "Accesorios - PS2",
    2: "Accesorios - PS3",
    3: "Accesorios - PS4",
    4: "Accesorios - PSP",
    5: "Accesorios - PSVita",
    6: "Accesorios - XBOX 360",
    7: "Accesorios - XBOX ONE",
    8: "Entradas (Digital)",
    9: "Envío de productos",
    10: "Consolas - PS2",
    11: "Consolas - PS3",
    12: "Consolas - PS4",
    13: "Consolas - PSP",
    14: "Consolas - PSVita",
    15: "Consolas - XBOX 360",
    16: "Consolas - XBOX ONE",
    17: "Consolas - Otras",
    18: "Juegos - PS2",
    19: "Juegos - PS3",
    20: "Juegos - PS4",
    21: "Juegos - PSP",
    22: "Juegos - PSVita",
    23: "Juegos - XBOX 360",
    24: "Juegos - XBOX ONE",
    25: "Juegos - Accesorios para juegos",
    26: "Juegos Android - Digital",
    27: "Juegos MAC - Digital",
    28: "Juegos PC - Ediciones adicionales",
    29: "Juegos PC - Ediciones de colección",
    30: "Juegos PC - Ediciones estándar",
    31: "Juegos PC - Digital",
    32: "Tarjetas de pago (Cine, Música, Juegos)",
    33: "Tarjetas de pago - Live!",
    34: "Tarjetas de pago - Live! (Digital)",
    35: "Tarjetas de pago - PSN",
    36: "Tarjetas de pago - Windows (Digital)",
    37: "Cine - Blu-Ray",
    38: "Cine - Blu-Ray 3D",
    39: "Cine - Blu-Ray 4K",
    40: "Cine - DVD",
    41: "Cine - Coleccionables",
    42: "Libros - Artbooks, enciclopedias",
    43: "Libros - Audiolibros",
    44: "Libros - Audiolibros (Digital)",
    45: "Libros - Audiolibros 1C",
    46: "Libros - Negocios",
    47: "Libros - Cómics, manga",
    48: "Libros - Computación",
    49: "Libros - Material metodológico 1C",
    50: "Libros - Postales",
    51: "Libros - Divulgación",
    52: "Libros - Guías de viaje",
    53: "Libros - Ficción",
    54: "Libros - Digital",
    55: "Música - CD producción local",
    56: "Música - CD producción de marca",
    57: "Música - MP3",
    58: "Música - Vinilo",
    59: "Música - Video musical",
    60: "Música - Ediciones de regalo",
    61: "Regalos - Artículos promocionales",
    62: "Regalos - Gadgets, robots, deporte",
    63: "Regalos - Peluches",
    64: "Regalos - Juegos de mesa",
    65: "Regalos - Juegos de mesa (compactos)",
    66: "Regalos - Postales, stickers",
    67: "Regalos - Didácticos",
    68: "Regalos - Certificados, servicios",
    69: "Regalos - Souvenirs",
    70: "Regalos - Souvenirs (colgantes)",
    71: "Regalos - Bolsas, álbumes, mousepads",
    72: "Regalos - Figuras",
    73: "Programas - 1C:Empresa 8",
    74: "Programas - MAC (Digital)",
    75: "Programas - Hogar y oficina",
    76: "Programas - Hogar y oficina (Digital)",
    77: "Programas - Educativos",
    78: "Programas - Educativos (Digital)",
    79: "Servicios",
    80: "Servicios - Entradas",
    81: "Medios vírgenes (spindle)",
    82: "Medios vírgenes (unidad)",
    83: "Baterías",
}

#: Ciudad (en español) de cada tienda; el nombre original empieza con la ciudad en ruso.
SHOP_CITY_ES: dict[int, str] = {
    0: "Yakutsk", 1: "Yakutsk", 2: "Adigueya", 3: "Balashikha", 4: "Volzhski", 5: "Vólogda",
    6: "Vorónezh", 7: "Vorónezh", 8: "Vorónezh", 9: "Venta itinerante", 10: "Zhukovski",
    11: "Zhukovski", 12: "Tienda online", 13: "Kazán", 14: "Kazán", 15: "Kaluga", 16: "Kolomna",
    17: "Krasnoyarsk", 18: "Krasnoyarsk", 19: "Kursk", 20: "Moscú", 21: "Moscú", 22: "Moscú",
    23: "Moscú", 24: "Moscú", 25: "Moscú", 26: "Moscú", 27: "Moscú", 28: "Moscú", 29: "Moscú",
    30: "Moscú", 31: "Moscú", 32: "Moscú", 33: "Mytishchi", 34: "N. Nóvgorod", 35: "N. Nóvgorod",
    36: "Novosibirsk", 37: "Novosibirsk", 38: "Omsk", 39: "Rostov del Don", 40: "Rostov del Don",
    41: "Rostov del Don", 42: "San Petersburgo", 43: "San Petersburgo", 44: "Samara",
    45: "Samara", 46: "Sérguiev Posad", 47: "Surgut", 48: "Tomsk", 49: "Tiumén", 50: "Tiumén",
    51: "Tiumén", 52: "Ufá", 53: "Ufá", 54: "Jimki", 55: "Almacén digital", 56: "Chéjov",
    57: "Yakutsk", 58: "Yakutsk", 59: "Yaroslavl",
}  # fmt: skip

#: Tipo de tienda según su naturaleza (derivado del nombre original).
SHOP_TYPE_SPECIAL: dict[int, str] = {
    9: "Itinerante",
    12: "Online",
    20: "Liquidación",
    55: "Online",
}

RAW_FILES = {
    "sales": "sales_train.csv",
    "items": "items.csv",
    "categories": "item_categories.csv",
    "shops": "shops.csv",
    "test": "test.csv",
}


def load_raw_data(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Lee los CSV originales de Kaggle.

    Args:
        raw_dir: carpeta con los archivos de Kaggle.

    Returns:
        Diccionario ``{"sales", "items", "categories", "shops", "test"}`` -> DataFrame.

    Raises:
        FileNotFoundError: si falta alguno de los archivos.
    """
    data = {}
    for key, fname in RAW_FILES.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {path}. Descárgalo de Kaggle (ver README).")
        data[key] = pd.read_csv(path)
    return data


def category_group(category_name_es: str) -> str:
    """Grupo de alto nivel de una categoría: la parte antes de ' - ' sin paréntesis."""
    group = category_name_es.split(" - ")[0]
    return re.sub(r"\s*\(.*\)", "", group).strip()


def shop_type(shop_id: int, shop_name: str) -> str:
    """Clasifica la tienda en Centro comercial / Calle / Online / Itinerante / Liquidación."""
    if shop_id in SHOP_TYPE_SPECIAL:
        return SHOP_TYPE_SPECIAL[shop_id]
    if re.search(r"\b(ТЦ|ТРЦ|ТРК|МТРЦ|ТК)\b", shop_name):
        return "Centro comercial"
    return "Calle"


def build_catalogs(
    items: pd.DataFrame, categories: pd.DataFrame, shops: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye catálogos enriquecidos de productos y tiendas (con traducciones).

    Returns:
        ``(items_full, shops_full)``: productos con su categoría (ES) y grupo; tiendas con
        ciudad, tipo y etiqueta legible para gráficas.
    """
    cats = categories.copy()
    cats["category_es"] = cats["item_category_id"].map(CATEGORY_NAMES_ES)
    cats["category_group"] = cats["category_es"].map(category_group)
    items_full = items.merge(cats, on="item_category_id", how="left")

    shops_full = shops.copy()
    shops_full["city"] = shops_full["shop_id"].map(SHOP_CITY_ES)
    shops_full["shop_type"] = [
        shop_type(i, n) for i, n in zip(shops_full["shop_id"], shops_full["shop_name"])
    ]
    shops_full["shop_label"] = shops_full["shop_id"].astype(str) + " · " + shops_full["city"]
    return items_full, shops_full


def add_catalog_info(
    sales: pd.DataFrame, items_full: pd.DataFrame, shops_full: pd.DataFrame
) -> pd.DataFrame:
    """Une ventas con catálogos y agrega columnas derivadas.

    Columnas nuevas: ``date`` (datetime), ``year``, ``month``, ``day_of_week``, ``revenue``
    (``item_price * item_cnt_day``), ``is_return`` y los atributos de producto/tienda.
    """
    df = sales.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_of_week  # 0 = lunes
    df["revenue"] = df["item_price"] * df["item_cnt_day"]
    df["is_return"] = df["item_cnt_day"] < 0
    df = df.merge(
        items_full[["item_id", "item_category_id", "category_es", "category_group"]],
        on="item_id",
        how="left",
    )
    df = df.merge(
        shops_full[["shop_id", "city", "shop_type", "shop_label"]], on="shop_id", how="left"
    )
    return df


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Aplica las decisiones de limpieza del EDA sobre ``sales_train`` crudo.

    1. Convierte ``date`` (texto dd.mm.aaaa) a datetime.
    2. Elimina filas duplicadas exactas.
    3. Imputa precios <= 0 con la mediana del mismo producto-tienda-mes (o del producto).
    4. Elimina transacciones extremas puntuales (precio > ``MAX_ITEM_PRICE``,
       cantidad diaria > ``MAX_ITEM_CNT_DAY``).
    5. Unifica tiendas duplicadas según ``SHOP_DUPLICATES``.

    Las devoluciones (``item_cnt_day < 0``) se conservan: la venta mensual objetivo es neta.

    Returns:
        Copia limpia del DataFrame de ventas.
    """
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df = df.drop_duplicates().reset_index(drop=True)

    bad_price = df["item_price"] <= 0
    if bad_price.any():
        valid = df.loc[~bad_price]
        med_ism = valid.groupby(["item_id", "shop_id", "date_block_num"])["item_price"].median()
        med_i = valid.groupby("item_id")["item_price"].median()
        for idx in df.index[bad_price]:
            key = (df.at[idx, "item_id"], df.at[idx, "shop_id"], df.at[idx, "date_block_num"])
            df.at[idx, "item_price"] = med_ism.get(key, med_i.get(key[0]))

    df = df[(df["item_price"] <= MAX_ITEM_PRICE) & (df["item_cnt_day"] <= MAX_ITEM_CNT_DAY)]
    df = df.assign(shop_id=df["shop_id"].replace(SHOP_DUPLICATES)).reset_index(drop=True)
    return df


@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "sales_clean.csv",
) -> None:
    """Genera el dataset de ventas limpio en ``data/interim``."""
    logger.info(f"Leyendo datos crudos de {raw_dir}")
    data = load_raw_data(raw_dir)
    clean = clean_sales(data["sales"])
    logger.info(f"Filas: {len(data['sales']):,} -> {len(clean):,}")
    clean.to_csv(output_path, index=False)
    logger.success(f"Dataset limpio guardado en {output_path}")


if __name__ == "__main__":
    app()
