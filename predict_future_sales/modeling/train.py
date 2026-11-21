"""Re-entrena el pipeline completo (preprocesamiento + modelo ganador) desde la terminal.

Reproduce la sección 5 de ``notebooks/04_model_training_n_tunning.ipynb``: toma el pipeline de
preprocesamiento (``models/preprocessing_pipeline.pkl``), le agrega como paso final el modelo
ganador con los hiperparámetros guardados en ``reports/best_model.json`` y lo ajusta con los meses
0-33 del grid mensual. El torneo de 15 modelos se ejecuta en el notebook 04.

Uso::

    python -m predict_future_sales.modeling.train
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
from loguru import logger
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import typer
import xgboost as xgb

from predict_future_sales.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
import predict_future_sales.features as pf

app = typer.Typer()

#: Clases de modelo que pueden resultar ganadoras en el torneo del notebook 04.
MODEL_CLASSES = {
    "Ridge": Ridge,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": xgb.XGBRegressor,
    "LGBMRegressor": lgb.LGBMRegressor,
}


def build_winner(summary: dict):
    """Instancia el modelo ganador a partir del resumen ``best_model.json``."""
    cls = MODEL_CLASSES[summary["clase"]]
    valid = cls().get_params()
    params = {k: v for k, v in summary["hiperparametros"].items() if k in valid}
    return cls(**params)


@app.command()
def main(
    monthly_path: Path = PROCESSED_DATA_DIR / "monthly_shop_item.csv",
    preprocessing_path: Path = MODELS_DIR / "preprocessing_pipeline.pkl",
    summary_path: Path = REPORTS_DIR / "best_model.json",
    output_path: Path = MODELS_DIR / "full_pipeline.pkl",
) -> None:
    """Ajusta preprocesamiento + modelo ganador con los meses 0-33 y guarda el ``.pkl``."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    logger.info(f"Modelo ganador: {summary['modelo']} ({summary['clase']})")

    grid = pf.build_monthly_grid(pd.read_csv(monthly_path))
    preprocessing = joblib.load(preprocessing_path)
    full_pipeline = Pipeline(clone(preprocessing).steps + [("model", build_winner(summary))])

    logger.info(f"Entrenando con {len(grid):,} filas (meses 0-33)...")
    full_pipeline.fit(grid[pf.KEYS], grid[pf.TARGET])
    joblib.dump(full_pipeline, output_path, compress=3)
    logger.success(f"Pipeline completo guardado en {output_path}")


if __name__ == "__main__":
    app()
