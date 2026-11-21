"""Genera las predicciones de ``test.csv`` con el pipeline completo, desde la terminal.

Reproduce ``notebooks/05_inference_calculation.ipynb``: carga ``models/full_pipeline.pkl``,
predice noviembre 2015 (``date_block_num = 34``) para cada par de ``test.csv``, recorta a
[0, 20] y escribe el archivo de envío de Kaggle (``ID, item_cnt_month``).

Uso::

    python -m predict_future_sales.modeling.predict
"""

from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from predict_future_sales.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
import predict_future_sales.features as pf

app = typer.Typer()


def predict_test(pipeline, test: pd.DataFrame, block: int = pf.TEST_BLOCK) -> pd.DataFrame:
    """Predice ``item_cnt_month`` para los pares de ``test`` en el mes ``block``.

    Args:
        pipeline: pipeline completo ajustado (preprocesamiento + modelo).
        test: DataFrame con columnas ``ID, shop_id, item_id`` (``test.csv`` crudo).
        block: ``date_block_num`` a predecir (34 = noviembre 2015).

    Returns:
        DataFrame ``ID, item_cnt_month`` con predicciones recortadas a [0, 20].
    """
    X = test.assign(date_block_num=block)[pf.KEYS]
    pred = np.clip(pipeline.predict(X), *pf.TARGET_CLIP)
    return pd.DataFrame({"ID": test["ID"].to_numpy(), "item_cnt_month": pred})


@app.command()
def main(
    test_path: Path = RAW_DATA_DIR / "test.csv",
    pipeline_path: Path = MODELS_DIR / "full_pipeline.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "submission.csv",
) -> None:
    """Escribe el archivo de envío con las predicciones del mes 34."""
    pipeline = joblib.load(pipeline_path)
    test = pd.read_csv(test_path)
    submission = predict_test(pipeline, test)
    submission.to_csv(output_path, index=False)
    logger.success(
        f"{len(submission):,} predicciones (media {submission.item_cnt_month.mean():.3f}) "
        f"guardadas en {output_path}"
    )


if __name__ == "__main__":
    app()
